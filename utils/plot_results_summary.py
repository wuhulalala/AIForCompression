import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path


DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / "logs" / "results"
DEFAULT_CRA5_SUMMARY = Path(__file__).resolve().parents[1] / "models" / "CRA5" / "summary.json"
DEFAULT_MODEL_PARAMS = {
    "CRA5-VAEformer": 404_900_000.0,
}
SCIENTIFIC_LABELS = {"CRA5", "CRA5-VAEformer"}
VIDEO_PREFIXES = ("DCMVC", "DCVC", "FLAVC")


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


def total_values_from_shape(shape):
    if not shape:
        return None
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def channels_per_spatial_position(shape):
    if not shape or len(shape) < 3:
        return 1
    total = 1
    for dim in shape[:-2]:
        total *= int(dim)
    return total


def compute_bpv(original_bytes, bitstream_bytes, compression_ratio):
    original_bytes = finite_float(original_bytes)
    bitstream_bytes = finite_float(bitstream_bytes)
    compression_ratio = finite_float(compression_ratio)
    if original_bytes and bitstream_bytes and original_bytes > 0:
        value_count = original_bytes / 4.0
        if value_count > 0:
            return bitstream_bytes * 8.0 / value_count
    if compression_ratio and compression_ratio > 0:
        return 32.0 / compression_ratio
    return None


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


def format_caesar_mode(model_id):
    if model_id == "caesar_v":
        return "CAESAR-V"
    if model_id == "caesar_d":
        return "CAESAR-D"
    return f"CAESAR-{model_id}" if model_id else "CAESAR"


def caesar_error_bound_label(source_name):
    match = re.search(r"_eb_([0-9p]+)em([0-9]+)$", source_name or "")
    if not match:
        return None
    mantissa = match.group(1).replace("p", ".")
    exponent = match.group(2)
    return f"{mantissa}e-{exponent}"


def caesar_annotation_text(row):
    if plot_group_key(row) != "CAESAR-D":
        return None
    error_bound = caesar_error_bound_label(row.get("source"))
    if not error_bound:
        return None
    return f"eb={error_bound}"


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
    shape = record.get("data_shape")
    compression_ratio = finite_float(
        compress.get("compression_ratio") or record.get("compression_ratio")
    )
    bitstream_bytes = finite_float(
        compress.get("bitstream_bytes") or record.get("bitstream_bytes")
    )
    raw_bpp = finite_float(compress.get("bpp") or record.get("bpp"))

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
    bpv = compute_bpv(original_bytes, bitstream_bytes, compression_ratio)
    bpp = raw_bpp
    if bpp is None and bpv is not None and shape:
        bpp = bpv * channels_per_spatial_position(shape)
    row = {
        "source": source_name,
        "label": label,
        "curve_key": curve_key(source_name, label, model_id),
        "model_id": model_id,
        "compression_ratio": compression_ratio,
        "psnr": finite_float(compress.get("psnr") or record.get("psnr")),
        "bpp": bpp,
        "bpv": bpv,
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
        "bpp",
        "bpv",
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
        groups[plot_group_key(row)].append(row)
    return {
        label: sorted(items, key=lambda item: item["compression_ratio"])
        for label, items in sorted(groups.items())
    }


def model_styles(rows):
    labels = sorted({plot_group_key(row) for row in rows})
    fixed_styles = {
        "CAESAR-V": {"color": "#2563eb", "marker": "o"},
        "CAESAR-D": {"color": "#dc2626", "marker": "D"},
        "CRA5-VAEformer": {"color": "#7c3aed", "marker": "s"},
    }
    family_palettes = {
        "image": ["#0f766e", "#d97706", "#65a30d", "#ea580c", "#16a34a", "#a16207", "#15803d"],
        "video": ["#1d4ed8", "#0891b2", "#4338ca", "#0284c7", "#2563eb", "#0f766e"],
        "caesar": ["#2563eb", "#dc2626"],
        "cra5": ["#7c3aed", "#a78bfa"],
    }
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    styles = {}
    family_indices = defaultdict(int)
    for idx, label in enumerate(labels):
        if label in fixed_styles:
            styles[label] = fixed_styles[label]
            continue
        sample_row = next(row for row in rows if plot_group_key(row) == label)
        family = model_family(sample_row)
        palette = family_palettes.get(family, family_palettes["image"])
        family_idx = family_indices[family]
        styles[label] = {
            "color": palette[family_idx % len(palette)],
            "marker": markers[idx % len(markers)],
        }
        family_indices[family] += 1
    return styles


def plot_label(row):
    if row["label"] == "CAESAR":
        return format_caesar_mode(row.get("model_id"))
    if row["label"] == "RwkvCompress":
        return "LALIC"
    return row["label"]


def plot_group_key(row):
    return plot_label(row)


def metric_axis_scales(metric):
    yscale = "log" if metric in {"encode_throughput", "decode_throughput"} else "linear"
    return "log", yscale


def param_name(row):
    if row["label"] == "CAESAR":
        return plot_group_key(row)
    if row.get("model_id"):
        return row["model_id"]
    return row["label"]


def param_range_name(row):
    if row["label"] == "CAESAR":
        return plot_group_key(row)
    return row["label"]


def model_family(row):
    label = row.get("label") or ""
    source = row.get("source") or ""
    model_id = row.get("model_id") or ""
    if label in SCIENTIFIC_LABELS:
        return "cra5"
    if label == "CAESAR" or source.startswith("CAESAR"):
        return "caesar"
    if (
        label.startswith(VIDEO_PREFIXES)
        or source.startswith(VIDEO_PREFIXES)
        or model_id.startswith(VIDEO_PREFIXES)
        or source == "video_intra"
    ):
        return "video"
    return "image"


def prepare_rows_for_plot(rows):
    prepared = []
    caesar_v_rows = []
    for row in rows:
        if plot_group_key(row) == "CAESAR-V":
            caesar_v_rows.append(row)
            continue
        prepared.append(row)

    if caesar_v_rows:
        caesar_v_rows.sort(
            key=lambda row: (
                0 if row.get("source") == "CAESAR" else 1,
                -(row.get("psnr") or float("-inf")),
                row.get("compression_ratio") or float("inf"),
            )
        )
        prepared.append(caesar_v_rows[0])
    return prepared


def rows_by_group(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[plot_group_key(row)].append(row)
    return {
        name: sorted(items, key=lambda item: item["compression_ratio"])
        for name, items in sorted(groups.items())
    }


def ranked_slot_indices(count, slot_count):
    if count <= 0 or slot_count <= 0:
        return []
    if count == 1:
        return [0]
    indices = []
    for slot_idx in range(slot_count):
        index = int(slot_idx * (count - 1) / (slot_count - 1))
        if index not in indices:
            indices.append(index)
    return indices


def select_ranked_slots(rows, slot_count=3):
    groups = rows_by_group(rows)
    slot_map = {f"slot_{idx + 1}": [] for idx in range(slot_count)}
    for _, items in groups.items():
        for slot_idx, row_idx in enumerate(ranked_slot_indices(len(items), slot_count)):
            slot_map[f"slot_{slot_idx + 1}"].append(items[row_idx])
    return {name: slot_rows for name, slot_rows in slot_map.items() if slot_rows}


def speed_score(row):
    encode = finite_float(row.get("encode_throughput"))
    decode = finite_float(row.get("decode_throughput"))
    if encode and decode and encode > 0 and decode > 0:
        return math.sqrt(encode * decode)
    return encode or decode or float("-inf")


def _minmax_normalize(value, values):
    if value is None:
        return 0.0
    finite_values = [v for v in values if v is not None]
    if not finite_values:
        return 0.0
    min_value = min(finite_values)
    max_value = max(finite_values)
    if math.isclose(min_value, max_value):
        return 1.0
    return (value - min_value) / (max_value - min_value)


def tradeoff_score(row, group_rows):
    psnr_score = _minmax_normalize(row.get("psnr"), [item.get("psnr") for item in group_rows])
    compression_score = _minmax_normalize(
        row.get("compression_ratio"),
        [item.get("compression_ratio") for item in group_rows],
    )
    return psnr_score * compression_score


def select_representative_points(rows):
    groups = rows_by_group(rows)
    selected = {
        "best_cr": [],
        "best_speed": [],
        "best_tradeoff": [],
    }
    for _, items in groups.items():
        selected["best_cr"].append(
            max(items, key=lambda row: (row.get("compression_ratio") or float("-inf"), row.get("psnr") or float("-inf")))
        )
        selected["best_speed"].append(max(items, key=speed_score))
        selected["best_tradeoff"].append(max(items, key=lambda row: tradeoff_score(row, items)))
    return selected


def reference_xticks(rows):
    compression_ratios = sorted(
        {
            row["compression_ratio"]
            for row in rows
            if row.get("compression_ratio") is not None
        }
    )
    if not compression_ratios:
        return []

    min_ratio = compression_ratios[0]
    max_ratio = compression_ratios[-1]
    preferred = [500, 1000, 200, 400, 2000, 4000, 5000]
    ticks = [tick for tick in preferred if min_ratio <= tick <= max_ratio]
    if ticks:
        return sorted(ticks[:2])
    return [compression_ratios[len(compression_ratios) // 2]]


def apply_reference_xticks(ax, rows):
    from matplotlib import ticker

    ticks = reference_xticks(rows)
    if not ticks:
        return
    ax.xaxis.set_major_locator(ticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter([str(int(tick)) for tick in ticks]))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())


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


def _apply_log_yticks(ax, rows, metric):
    from matplotlib import ticker

    values = [r[metric] for r in rows if r.get(metric) is not None and r[metric] > 0]
    if not values:
        return
    vmin, vmax = min(values), max(values)
    decades = math.floor(math.log10(vmin)), math.ceil(math.log10(vmax))
    ticks = []
    for exp in range(decades[0], decades[1] + 1):
        base = 10 ** exp
        ticks.append(base)
        ticks.append(base * 2)
        ticks.append(base * 5)
    ticks = [t for t in ticks if vmin * 0.8 <= t <= vmax * 1.2]
    if ticks:
        ax.yaxis.set_major_locator(ticker.FixedLocator(ticks))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.tick_params(axis="y", which="minor", left=False)


def draw_metric(ax, rows, metric, y_label, title, styles, yscale="linear", compact=False):
    groups = grouped_rows(rows, metric)
    used_labels = set()

    scatter_s = 42 if compact else 54
    scatter_ew = 0.5 if compact else 0.8
    line_lw = 1.4 if compact else 1.8
    line_ms = 4.5 if compact else 5
    title_fs = 11 if compact else 13
    annot_fs = 7 if compact else 8

    for _, items in groups.items():
        label = plot_group_key(items[0])
        style = styles[label]
        legend_label = label if label not in used_labels else "_nolegend_"
        used_labels.add(label)
        x_values = [item["compression_ratio"] for item in items]
        y_values = [item[metric] for item in items]

        if len(items) == 1:
            ax.scatter(
                x_values,
                y_values,
                s=scatter_s,
                color=style["color"],
                marker=style["marker"],
                edgecolor="white",
                linewidth=scatter_ew,
                label=legend_label,
                zorder=3,
            )
        else:
            ax.plot(
                x_values,
                y_values,
                color=style["color"],
                marker=style["marker"],
                linewidth=line_lw,
                markersize=line_ms,
                label=legend_label,
            )

        for item in items:
            annotation = caesar_annotation_text(item)
            if not annotation:
                continue
            ax.annotate(
                annotation,
                (item["compression_ratio"], item[metric]),
                textcoords="offset points",
                xytext=(5, 6),
                fontsize=annot_fs,
                color=style["color"],
            )

    ax.set_title(title, fontsize=title_fs, pad=8)
    ax.set_xscale("log")
    ax.set_yscale(yscale)
    apply_reference_xticks(ax, rows)
    if yscale == "log":
        _apply_log_yticks(ax, rows, metric)
    ax.set_xlabel("Compression Ratio")
    ax.set_ylabel(y_label)
    ax.grid(True, which="major", alpha=0.18, linestyle="-", linewidth=0.5)
    ax.grid(True, which="minor", alpha=0.06, linestyle="-", linewidth=0.3)


def plot_metric(rows, metric, y_label, title, output_path, styles=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if styles is None:
        styles = model_styles(rows)

    groups = grouped_rows(rows, metric)
    fig, ax = plt.subplots(figsize=(14, 8))
    used_labels = set()

    for _, items in groups.items():
        label = plot_group_key(items[0])
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

    ax.set_title(title, fontsize=16)
    xscale, yscale = metric_axis_scales(metric)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    apply_reference_xticks(ax, rows)
    ax.set_xlabel("Compression Ratio")
    ax.set_ylabel(y_label)
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.12)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def grouped_rows_by_x(rows, x_metric, y_metric):
    groups = defaultdict(list)
    for row in rows:
        if row.get(x_metric) is None or row.get(y_metric) is None:
            continue
        groups[plot_group_key(row)].append(row)
    return {
        label: sorted(items, key=lambda item: item[x_metric])
        for label, items in sorted(groups.items())
    }


def plot_metric_vs_bpv(rows, metric, y_label, title, output_path, styles=None, yscale="linear"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if styles is None:
        styles = model_styles(rows)

    groups = grouped_rows_by_x(rows, "bpv", metric)
    fig, ax = plt.subplots(figsize=(14, 8))
    used_labels = set()
    for _, items in groups.items():
        label = plot_group_key(items[0])
        style = styles[label]
        legend_label = label if label not in used_labels else "_nolegend_"
        used_labels.add(label)
        x_values = [item["bpv"] for item in items]
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

    ax.set_xlabel("BPV (bits per value)")
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=16)
    ax.set_yscale(yscale)
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.12)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def throughput_bpv_overview_specs():
    return [
        ("encode_throughput", "Encode Throughput vs BPV", "encode_throughput_vs_bpv.png"),
        ("decode_throughput", "Decode Throughput vs BPV", "decode_throughput_vs_bpv.png"),
    ]


def bpv_stale_plot_names(slug):
    return [
        "encode_throughput_vs_bpv.png",
        "decode_throughput_vs_bpv.png",
        f"{slug}_encode_throughput_vs_bpv.png",
        f"{slug}_decode_throughput_vs_bpv.png",
        f"{slug}_encode_throughput_by_bpv_bar.png",
        f"{slug}_decode_throughput_by_bpv_bar.png",
    ]


def _bar_width_from_sorted_x(x_values):
    if len(x_values) < 2:
        return max(x_values[0] * 0.05, 0.0012) if x_values else 0.006
    diffs = [right - left for left, right in zip(x_values, x_values[1:]) if right > left]
    if not diffs:
        return max(x_values[0] * 0.05, 0.0012)
    return max(min(diffs) * 0.22, 0.0012)


def plot_throughput_bar_by_bpv(rows, metric, title, output_path, styles=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if styles is None:
        styles = model_styles(rows)

    plot_rows = [
        row for row in rows
        if row.get("bpv") is not None and row.get(metric) is not None and row.get(metric) > 0
    ]
    if not plot_rows:
        return
    plot_rows.sort(key=lambda row: row["bpv"])

    fig, ax = plt.subplots(figsize=(18, 8))
    used_labels = set()
    width = _bar_width_from_sorted_x([row["bpv"] for row in plot_rows])
    for row in plot_rows:
        label = plot_group_key(row)
        style = styles[label]
        legend_label = label if label not in used_labels else "_nolegend_"
        used_labels.add(label)
        ax.bar(
            row["bpv"],
            row[metric],
            width=width,
            color=style["color"],
            edgecolor="white",
            linewidth=0.6,
            label=legend_label,
            alpha=0.9,
        )

    ax.set_xlabel("BPV (bits per value)")
    ax.set_ylabel("Throughput (MB/s)")
    ax.set_title(title, fontsize=16)
    ax.set_yscale("log")
    ax.margins(x=0.03)
    ax.grid(True, axis="y", which="major", alpha=0.28)
    ax.grid(True, axis="y", which="minor", alpha=0.1)
    ax.legend(loc="best", fontsize=9, ncol=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_bpv_variants(rows, output_dir, styles=None):
    if styles is None:
        styles = model_styles(rows)
    for subset, slug, title in bpv_plot_specs(rows):
        for stale_name in bpv_stale_plot_names(slug):
            stale_path = output_dir / stale_name
            if stale_path.exists():
                stale_path.unlink()
        plot_metric_vs_bpv(
            subset,
            "psnr",
            "PSNR (dB)",
            "PSNR vs BPV",
            output_dir / f"{slug}_psnr_vs_bpv.png",
            styles=styles,
        )
        plot_average_throughput_bar(
            subset,
            f"{title} Average Throughput",
            output_dir / f"{slug}_average_throughput_bar.png",
        )


def filter_rows(rows, families=None, labels=None):
    filtered = []
    allowed_families = set(families or [])
    allowed_labels = set(labels or [])
    for row in rows:
        if allowed_families and model_family(row) in allowed_families:
            filtered.append(row)
            continue
        if allowed_labels and plot_group_key(row) in allowed_labels:
            filtered.append(row)
    return filtered


def bpv_plot_specs(rows):
    specs = [
        (
            sorted(filter_rows(rows, families={"caesar", "cra5"}), key=plot_group_key),
            "scientific_family",
            "Scientific Compression",
        ),
        (
            sorted(filter_rows(rows, families={"video"}), key=plot_group_key),
            "video_family",
            "Video Compression",
        ),
        (
            sorted(filter_rows(rows, families={"image"}), key=plot_group_key),
            "image_family",
            "Image Compression",
        ),
    ]
    return [spec for spec in specs if spec[0]]


def average_throughput_rows(rows):
    grouped = defaultdict(lambda: {"encode": [], "decode": []})
    for row in rows:
        name = plot_group_key(row)
        if row.get("encode_throughput") is not None:
            grouped[name]["encode"].append(row["encode_throughput"])
        if row.get("decode_throughput") is not None:
            grouped[name]["decode"].append(row["decode_throughput"])
    averages = []
    for name, values in sorted(grouped.items()):
        encode_values = values["encode"]
        decode_values = values["decode"]
        averages.append(
            {
                "name": name,
                "encode_throughput": sum(encode_values) / len(encode_values) if encode_values else None,
                "decode_throughput": sum(decode_values) / len(decode_values) if decode_values else None,
            }
        )
    return averages


def plot_average_throughput_bar(rows, title, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    avg_rows = average_throughput_rows(rows)
    if not avg_rows:
        return

    names = [row["name"] for row in avg_rows]
    encode_values = [row["encode_throughput"] for row in avg_rows]
    decode_values = [row["decode_throughput"] for row in avg_rows]
    x = np.arange(len(names))
    width = 0.36

    fig_width = max(7.2, 0.9 * len(names) + 2.2)
    fig, ax = plt.subplots(figsize=(fig_width, 5.4))
    ax.bar(x - width / 2, encode_values, width=width, color="#4c78a8", label="Encode")
    ax.bar(x + width / 2, decode_values, width=width, color="#f58518", label="Decode")

    ax.set_xticks(x)
    ax.set_xticklabels([format_grouped_bar_label(name) for name in names], rotation=16)
    ax.set_ylabel("Average Throughput (MB/s)")
    ax.set_title(title, fontsize=16)
    ax.set_yscale("log")
    ax.grid(True, axis="y", which="major", alpha=0.28)
    ax.grid(True, axis="y", which="minor", alpha=0.1)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout(pad=0.9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def selection_series_styles(series_names):
    palette = [
        {"color": "#4c78a8", "hatch": "", "edgecolor": "#1f3b5b"},
        {"color": "#f58518", "hatch": "xx", "edgecolor": "#8a4b0f"},
        {"color": "#54a24b", "hatch": "//", "edgecolor": "#2d5a28"},
        {"color": "#e45756", "hatch": "..", "edgecolor": "#8d2f2e"},
        {"color": "#72b7b2", "hatch": "\\\\\\", "edgecolor": "#356764"},
        {"color": "#b279a2", "hatch": "oo", "edgecolor": "#6c4261"},
    ]
    return {
        name: palette[idx % len(palette)]
        for idx, name in enumerate(series_names)
    }


def display_name_for_selection(series_name, strategy):
    if strategy == "ranked_slots":
        labels = {
            "slot_1": "Low CR slot",
            "slot_2": "Mid CR slot",
            "slot_3": "High CR slot",
        }
        return labels.get(series_name, series_name.replace("_", " ").title())
    labels = {
        "best_cr": "Best CR",
        "best_speed": "Best Speed",
        "best_tradeoff": "Best Trade-off",
    }
    return labels.get(series_name, series_name.replace("_", " ").title())


def grouped_bar_xtick_labels(series_map):
    labels = []
    for rows in series_map.values():
        for row in rows:
            label = plot_group_key(row)
            if label not in labels:
                labels.append(label)
    return labels


def grouped_bar_metric_specs():
    return [
        ("psnr", "PSNR (dB)", "PSNR (dB)", "linear"),
        ("compression_ratio", "Compression Ratio", "Compression Ratio", "log"),
        ("encode_throughput", "Compression Throughput (MB/s)", "Compression Throughput (MB/s)", "log"),
        ("decode_throughput", "Decompression Throughput (MB/s)", "Decompression Throughput (MB/s)", "log"),
    ]


def format_grouped_bar_label(label):
    replacements = {
        "CAESAR-V": "CAESAR\nV",
        "CAESAR-D": "CAESAR\nD",
        "CRA5-VAEformer": "CRA5\nVAEformer",
        "DCVC-RT": "DCVC-\nRT",
        "LIC-HPCM-base": "LIC-HPCM\nbase",
        "LIC-HPCM-large": "LIC-HPCM\nlarge",
        "LALIC": "LALIC",
    }
    if label in replacements:
        return replacements[label]
    if len(label) > 10 and "-" in label:
        return label.replace("-", "-\n", 1)
    return label


def plot_grouped_metric_bars(ax, categories, series_map, metric, title, y_label, yscale, styles, strategy):
    import numpy as np

    series_names = list(series_map)
    x = np.arange(len(categories))
    width = min(0.78 / max(len(series_names), 1), 0.26)

    for idx, series_name in enumerate(series_names):
        row_by_category = {plot_group_key(row): row for row in series_map[series_name]}
        values = [row_by_category.get(category, {}).get(metric) for category in categories]
        style = styles[series_name]
        offsets = x + (idx - (len(series_names) - 1) / 2) * width
        plotted_values = [value if value is not None and value > 0 else float("nan") for value in values]
        ax.bar(
            offsets,
            plotted_values,
            width=width,
            label=display_name_for_selection(series_name, strategy),
            color=style["color"],
            edgecolor=style["edgecolor"],
            linewidth=0.9,
            hatch=style["hatch"],
            zorder=3,
        )

    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels([format_grouped_bar_label(label) for label in categories], rotation=24)
    ax.set_yscale(yscale)
    ax.set_ylabel(y_label)
    ax.grid(True, axis="y", which="major", alpha=0.22, linewidth=0.5)
    ax.grid(True, axis="y", which="minor", alpha=0.08, linewidth=0.3)
    ax.set_axisbelow(True)


def plot_grouped_bar_summary(rows, output_path, figure_title, series_map, strategy):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    categories = grouped_bar_xtick_labels(series_map)
    if not categories:
        return

    styles = selection_series_styles(list(series_map))
    metric_specs = grouped_bar_metric_specs()
    fig_width = max(13.0, 1.25 * len(categories) + 7.6)
    fig, axes = plt.subplots(1, len(metric_specs), figsize=(fig_width, 5.8))
    for ax, (metric, title, y_label, yscale) in zip(axes, metric_specs):
        plot_grouped_metric_bars(
            ax,
            categories,
            series_map,
            metric,
            title,
            y_label,
            yscale,
            styles,
            strategy,
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(len(labels), 4),
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        fontsize=9,
    )
    fig.suptitle(figure_title, fontsize=14, fontweight="bold", y=1.06)
    fig.tight_layout(rect=[0, 0.1, 1, 0.9])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_bar_variants(rows, output_dir):
    specs = [
        (
            rows,
            "summary",
            "ERA5 Summary",
        ),
        (
            filter_rows(rows, families={"caesar"}),
            "caesar_family",
            "CAESAR Family",
        ),
        (
            filter_rows(rows, labels={"CAESAR-V", "CAESAR-D"}),
            "caesar_modes",
            "CAESAR V/D Modes",
        ),
        (
            filter_rows(rows, families={"video"}),
            "video_family",
            "Video Compression",
        ),
        (
            filter_rows(rows, families={"image"}),
            "image_family",
            "Image Compression Models",
        ),
    ]
    for subset, slug, title in specs:
        if not subset:
            continue
        ranked_slots = select_ranked_slots(subset, slot_count=3)
        plot_grouped_bar_summary(
            subset,
            output_dir / f"{slug}_ranked_slots_bar.png",
            f"{title} Ranked Slot Bar Summary",
            ranked_slots,
            "ranked_slots",
        )
        representatives = select_representative_points(subset)
        plot_grouped_bar_summary(
            subset,
            output_dir / f"{slug}_representative_bar.png",
            f"{title} Representative Point Bar Summary",
            representatives,
            "representatives",
        )


def plot_tradeoff(rows, output_path, figure_title="ERA5 Compression Trade-off Summary",
                  figsize=None, compact=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if figsize is None:
        figsize = (20, 6.6)

    label_count = len({plot_group_key(r) for r in rows})

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
        }
    )

    styles = model_styles(rows)
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True)
    draw_metric(
        axes[0],
        rows,
        "psnr",
        "PSNR (dB)",
        "Quality",
        styles,
        compact=compact,
    )
    draw_metric(
        axes[1],
        rows,
        "encode_throughput",
        "Encode Throughput (M/s)",
        "Compression Speed",
        styles,
        yscale="log",
        compact=compact,
    )
    draw_metric(
        axes[2],
        rows,
        "decode_throughput",
        "Decode Throughput (M/s)",
        "Decompression Speed",
        styles,
        yscale="log",
        compact=compact,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    legend_fs = 7.5 if compact else 8
    ncol = min(label_count, 6)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=ncol,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
        fontsize=legend_fs,
    )
    suptitle_fs = 14 if compact else 16
    bottom_rect = 0.13 if compact else 0.18
    fig.suptitle(figure_title, fontsize=suptitle_fs, fontweight="bold")
    fig.tight_layout(rect=[0, bottom_rect, 1, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff_variants(rows, output_dir):
    specs = [
        (
            rows,
            output_dir / "summary_tradeoff.png",
            "ERA5 Compression Trade-off Summary",
        ),
        (
            rows,
            output_dir / "scientific_vs_generic_tradeoff.png",
            "Scientific vs Generic Compression",
        ),
        (
            filter_rows(rows, families={"image", "video"}),
            output_dir / "image_vs_video_tradeoff.png",
            "Image vs Video Compression",
        ),
        (
            filter_rows(rows, labels={"CAESAR-V", "CAESAR-D"}),
            output_dir / "caesar_modes_tradeoff.png",
            "CAESAR V/D Modes",
            (15, 4.2),
            True,
        ),
        (
            filter_rows(rows, families={"caesar"}),
            output_dir / "caesar_family_tradeoff.png",
            "CAESAR Family",
            (15, 4.2),
            True,
        ),
        (
            filter_rows(rows, families={"video"}),
            output_dir / "video_family_tradeoff.png",
            "Video Compression (DCMVC / DCVC-RT)",
        ),
        (
            filter_rows(rows, families={"image"}),
            output_dir / "image_family_tradeoff.png",
            "Image Compression Models",
        ),
    ]
    for spec in specs:
        subset, path, title = spec[:3]
        figsize = spec[3] if len(spec) > 3 else None
        compact = spec[4] if len(spec) > 4 else False
        if subset:
            plot_tradeoff(subset, path, figure_title=title, figsize=figsize, compact=compact)


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
    plot_rows = prepare_rows_for_plot(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "summary_metrics.csv")
    plot_tradeoff_variants(plot_rows, args.output_dir)
    plot_grouped_bar_variants(rows, args.output_dir)
    styles = model_styles(plot_rows)
    missing_params = plot_model_params(rows, args.output_dir / "model_params_bar.png")
    missing_range_params = plot_model_param_ranges(
        rows, args.output_dir / "model_params_floating_bar.png"
    )
    plot_metric(
        plot_rows,
        "psnr",
        "PSNR (dB)",
        "PSNR vs Compression Ratio",
        args.output_dir / "psnr_vs_compression_ratio.png",
        styles=styles,
    )
    plot_metric(
        plot_rows,
        "encode_throughput",
        "Encode Throughput (M/s)",
        "Encode Throughput vs Compression Ratio",
        args.output_dir / "encode_throughput_vs_compression_ratio.png",
        styles=styles,
    )
    plot_metric(
        plot_rows,
        "decode_throughput",
        "Decode Throughput (M/s)",
        "Decode Throughput vs Compression Ratio",
        args.output_dir / "decode_throughput_vs_compression_ratio.png",
        styles=styles,
    )
    plot_bpv_variants(plot_rows, args.output_dir, styles=styles)
    print(f"Wrote {len(rows)} rows and plots to {args.output_dir}")
    missing_all = sorted(set(missing_params or []) | set(missing_range_params or []))
    if missing_all:
        print("Missing params:", ", ".join(missing_all))


if __name__ == "__main__":
    main()
