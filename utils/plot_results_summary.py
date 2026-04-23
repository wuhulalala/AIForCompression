import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / "logs" / "results"
DEFAULT_CRA5_SUMMARY = Path(__file__).resolve().parents[1] / "models" / "CRA5" / "summary.json"
DEFAULT_MODEL_PARAMS = {
    "CRA5-VAEformer": 404_900_000.0,
}


def finite_float(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def throughput_to_mps(value):
    value = finite_float(value)
    if value is None:
        return None
    if value > 10000:
        return value / 1e6
    return value


def compute_throughput_mps(original_bytes, seconds):
    original_bytes = finite_float(original_bytes)
    seconds = finite_float(seconds)
    if not original_bytes or not seconds or seconds <= 0:
        return None
    return original_bytes / seconds / 1e6


def original_bytes_from_shape(shape):
    if not shape:
        return None
    total = 1
    for dim in shape:
        total *= int(dim)
    return total * 4


def pick_label(source_name, record):
    return (
        record.get("arch")
        or record.get("model")
        or record.get("model_id")
        or source_name
    )


def curve_key(source_name, label, model_id):
    if label == "CAESAR":
        return f"{source_name}/{model_id or 'unknown'}"
    return label


def normalize_record(source_name, record):
    if "error" in record:
        return None

    compress = record.get("compress") or {}
    original_bytes = (
        compress.get("original_bytes")
        or record.get("original_bytes")
        or original_bytes_from_shape(record.get("data_shape"))
    )
    encode_time = (
        compress.get("encode_time")
        or record.get("encode_time")
        or record.get("encode_time_avg")
        or record.get("avg_encode_time")
    )
    decode_time = (
        compress.get("decode_time")
        or record.get("decode_time")
        or record.get("decode_time_avg")
        or record.get("avg_decode_time")
    )

    encode_throughput = throughput_to_mps(record.get("encode_throughput"))
    if encode_throughput is None:
        encode_throughput = compute_throughput_mps(original_bytes, encode_time)

    decode_throughput = throughput_to_mps(record.get("decode_throughput"))
    if decode_throughput is None:
        decode_throughput = compute_throughput_mps(original_bytes, decode_time)

    label = pick_label(source_name, record)
    model_id = record.get("model_id") or record.get("model")
    params = finite_float(record.get("params"))
    if params is None:
        params = DEFAULT_MODEL_PARAMS.get(label) or DEFAULT_MODEL_PARAMS.get(model_id)
    row = {
        "source": source_name,
        "label": label,
        "curve_key": curve_key(source_name, label, model_id),
        "model_id": model_id,
        "compression_ratio": finite_float(
            compress.get("compression_ratio") or record.get("compression_ratio")
        ),
        "psnr": finite_float(compress.get("psnr") or record.get("psnr")),
        "params": params,
        "encode_throughput": encode_throughput,
        "decode_throughput": decode_throughput,
    }
    if row["compression_ratio"] is None:
        return None
    return row


def read_summary(path):
    with path.open("r") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    return [payload]


def collect_records(results_dir=DEFAULT_RESULTS_DIR, cra5_summary=DEFAULT_CRA5_SUMMARY):
    rows = []
    for summary_path in sorted(Path(results_dir).glob("*/summary.json")):
        source_name = summary_path.parent.name
        for record in read_summary(summary_path):
            row = normalize_record(source_name, record)
            if row is not None:
                rows.append(row)

    cra5_summary = Path(cra5_summary)
    if cra5_summary.exists():
        for record in read_summary(cra5_summary):
            row = normalize_record("CRA5", record)
            if row is not None:
                rows.append(row)
    return rows


def write_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "label",
        "curve_key",
        "model_id",
        "compression_ratio",
        "psnr",
        "params",
        "encode_throughput",
        "decode_throughput",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def grouped_rows(rows, metric):
    groups = defaultdict(list)
    for row in rows:
        if row.get(metric) is None:
            continue
        groups[row["curve_key"]].append(row)
    return {
        label: sorted(items, key=lambda item: item["compression_ratio"])
        for label, items in sorted(groups.items())
    }


def model_styles(rows):
    labels = sorted({row["label"] for row in rows})
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    return {
        label: {
            "color": palette[idx % len(palette)],
            "marker": markers[idx % len(markers)],
        }
        for idx, label in enumerate(labels)
    }


def param_name(row):
    if row["label"] == "CAESAR" and row.get("model_id"):
        return f"CAESAR {row['model_id']}"
    if row.get("model_id"):
        return row["model_id"]
    return row["label"]


def param_range_name(row):
    if row["label"] == "CAESAR" and row.get("model_id"):
        return f"CAESAR {row['model_id']}"
    return row["label"]


def model_param_rows(rows):
    values = {}
    missing = set()
    for row in rows:
        name = param_name(row)
        params = row.get("params")
        if params is None:
            missing.add(name)
            continue
        values.setdefault(name, params)
        missing.discard(name)

    param_rows = [
        {"name": name, "params": params, "params_m": params / 1e6}
        for name, params in values.items()
    ]
    param_rows.sort(key=lambda row: row["params"])
    return param_rows, sorted(missing)


def model_param_ranges(rows):
    groups = defaultdict(set)
    missing = set()
    for row in rows:
        name = param_range_name(row)
        params = row.get("params")
        if params is None:
            missing.add(name)
            continue
        groups[name].add(params)
        missing.discard(name)

    range_rows = []
    for name, values in groups.items():
        sorted_values = sorted(values)
        min_params = sorted_values[0]
        max_params = sorted_values[-1]
        range_rows.append(
            {
                "name": name,
                "min_params": min_params,
                "max_params": max_params,
                "min_params_m": min_params / 1e6,
                "max_params_m": max_params / 1e6,
                "params_values": sorted_values,
                "params_values_m": [value / 1e6 for value in sorted_values],
            }
        )
    range_rows.sort(key=lambda row: row["max_params"])
    return range_rows, sorted(missing)


def model_param_summary_rows(rows):
    range_rows, missing = model_param_ranges(rows)
    summary_rows = []
    for row in range_rows:
        if row["min_params"] == row["max_params"]:
            label_text = f"{row['max_params_m']:.1f}M"
        else:
            label_text = f"{row['min_params_m']:.1f}-{row['max_params_m']:.1f}M"
        summary_rows.append(
            {
                "name": row["name"],
                "params": row["max_params"],
                "params_m": row["max_params_m"],
                "label_text": label_text,
                "params_values": row["params_values"],
            }
        )
    return summary_rows, missing


def draw_metric(ax, rows, metric, y_label, title, styles, yscale="linear"):
    groups = grouped_rows(rows, metric)
    used_labels = set()

    for _, items in groups.items():
        label = items[0]["label"]
        style = styles[label]
        legend_label = label if label not in used_labels else "_nolegend_"
        used_labels.add(label)
        x_values = [item["compression_ratio"] for item in items]
        y_values = [item[metric] for item in items]

        if len(items) == 1:
            ax.scatter(
                x_values,
                y_values,
                s=54,
                color=style["color"],
                marker=style["marker"],
                edgecolor="white",
                linewidth=0.8,
                label=legend_label,
                zorder=3,
            )
        else:
            ax.plot(
                x_values,
                y_values,
                color=style["color"],
                marker=style["marker"],
                linewidth=1.8,
                markersize=5,
                label=legend_label,
            )

    ax.set_title(title, fontsize=13, pad=8)
    ax.set_xscale("log")
    ax.set_yscale(yscale)
    ax.set_xlabel("Compression Ratio")
    ax.set_ylabel(y_label)
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.12)


def plot_metric(rows, metric, y_label, title, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = grouped_rows(rows, metric)
    fig, ax = plt.subplots(figsize=(14, 8))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]

    for idx, (label, items) in enumerate(groups.items()):
        x_values = [item["compression_ratio"] for item in items]
        y_values = [item[metric] for item in items]
        ax.plot(
            x_values,
            y_values,
            marker=markers[idx % len(markers)],
            linewidth=1.8,
            markersize=5,
            label=label,
        )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Compression Ratio")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(rows, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
        }
    )

    styles = model_styles(rows)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), sharex=True)
    draw_metric(
        axes[0],
        rows,
        "psnr",
        "PSNR (dB)",
        "Quality",
        styles,
    )
    draw_metric(
        axes[1],
        rows,
        "encode_throughput",
        "Encode Throughput (M/s)",
        "Compression Speed",
        styles,
        yscale="log",
    )
    draw_metric(
        axes[2],
        rows,
        "decode_throughput",
        "Decode Throughput (M/s)",
        "Decompression Speed",
        styles,
        yscale="log",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(5, len(labels)),
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle("ERA5 Compression Trade-off Summary", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_model_params(rows, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    param_rows, missing = model_param_summary_rows(rows)
    if not param_rows:
        return missing

    names = [row["name"] for row in param_rows]
    values = [row["params_m"] for row in param_rows]

    fig_height = max(5.0, 0.5 * len(names) + 1.8)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    colors = plt.cm.tab20(range(len(names)))
    bars = ax.barh(names, values, color=colors, edgecolor="white", linewidth=0.8)

    ax.set_xlabel("Parameters (M)")
    ax.set_title("Model Parameter Count", fontsize=15, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    max_value = max(values)
    ax.set_xlim(0, max_value * 1.16)
    for bar, value, row in zip(bars, values, param_rows):
        ax.text(
            value + max_value * 0.015,
            bar.get_y() + bar.get_height() / 2,
            row["label_text"],
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return missing


def plot_model_param_ranges(rows, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    range_rows, missing = model_param_ranges(rows)
    if not range_rows:
        return missing

    names = [row["name"] for row in range_rows]
    min_values = [row["min_params_m"] for row in range_rows]
    max_values = [row["max_params_m"] for row in range_rows]
    widths = [max_v - min_v for min_v, max_v in zip(min_values, max_values)]
    y_positions = list(range(len(names)))

    fig_height = max(5.0, 0.5 * len(names) + 1.8)
    fig, ax = plt.subplots(figsize=(11.5, fig_height))
    colors = plt.cm.tab20(range(len(names)))
    bars = ax.barh(
        y_positions,
        widths,
        left=min_values,
        height=0.48,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.85,
    )

    for y, row, min_value, max_value in zip(y_positions, range_rows, min_values, max_values):
        point_values = row["params_values_m"]
        ax.scatter(
            point_values,
            [y] * len(point_values),
            color="#1f1f1f",
            s=24,
            zorder=3,
        )
        if min_value == max_value:
            label = f"{max_value:.1f}M"
        else:
            label = f"{min_value:.1f}-{max_value:.1f}M"
        ax.text(
            max_value + max(max_values) * 0.018,
            y,
            label,
            va="center",
            fontsize=9,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(names)
    ax.set_xlabel("Parameters (M)")
    ax.set_title("Model Parameter Range by Quality", fontsize=15, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, max(max_values) * 1.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return missing


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot PSNR and encode/decode throughput against compression ratio."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--cra5-summary", type=Path, default=DEFAULT_CRA5_SUMMARY)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "summary_plots",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = collect_records(args.results_dir, args.cra5_summary)
    if not rows:
        raise SystemExit("No valid records found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "summary_metrics.csv")
    plot_tradeoff(rows, args.output_dir / "summary_tradeoff.png")
    missing_params = plot_model_params(rows, args.output_dir / "model_params_bar.png")
    missing_range_params = plot_model_param_ranges(
        rows, args.output_dir / "model_params_floating_bar.png"
    )
    plot_metric(
        rows,
        "psnr",
        "PSNR (dB)",
        "PSNR vs Compression Ratio",
        args.output_dir / "psnr_vs_compression_ratio.png",
    )
    plot_metric(
        rows,
        "encode_throughput",
        "Encode Throughput (M/s)",
        "Encode Throughput vs Compression Ratio",
        args.output_dir / "encode_throughput_vs_compression_ratio.png",
    )
    plot_metric(
        rows,
        "decode_throughput",
        "Decode Throughput (M/s)",
        "Decode Throughput vs Compression Ratio",
        args.output_dir / "decode_throughput_vs_compression_ratio.png",
    )
    print(f"Wrote {len(rows)} rows and plots to {args.output_dir}")
    missing_all = sorted(set(missing_params or []) | set(missing_range_params or []))
    if missing_all:
        print("Missing params:", ", ".join(missing_all))


if __name__ == "__main__":
    main()
