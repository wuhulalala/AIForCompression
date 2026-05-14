"""
Plot Kodak full-dataset results: PSNR vs BPP, Encode Throughput vs BPP, Decode Throughput vs BPP.
Usage: python utils/plot_kodak_results.py
"""
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "unified_results" / "kodak_all_models" / "kodak_avg_summary.csv"
OUTPUT_DIR = PROJECT_ROOT / "unified_results" / "kodak_all_models" / "plots"

MODEL_FAMILIES = {
    "DCAE":          {"color": "#0f766e", "marker": "o", "label": "DCAE"},
    "LIC-HPCM-base": {"color": "#d97706", "marker": "s", "label": "LIC-HPCM base"},
    "LIC-HPCM-large":{"color": "#ea580c", "marker": "D", "label": "LIC-HPCM large"},
    "LIC_TCM":       {"color": "#65a30d", "marker": "^", "label": "LIC-TCM"},
    "RwkvCompress":  {"color": "#1d4ed8", "marker": "P", "label": "LALIC"},
    "WeConvene":     {"color": "#7c3aed", "marker": "X", "label": "WeConvene"},
}


def read_csv(path):
    rows = []
    with open(path) as f:
        for d in csv.DictReader(f):
            for k in ["psnr", "bpp", "encode_throughput_MBps", "decode_throughput_MBps",
                       "encode_time_avg", "decode_time_avg", "mse", "rmse", "compression_ratio", "params"]:
                if k in d:
                    d[k] = float(d[k])
            rows.append(d)
    return rows


def classify(model_name, model_id):
    if model_name == "LIC-HPCM":
        return "LIC-HPCM-base" if "base" in model_id else "LIC-HPCM-large"
    return model_name


def group_rows(rows):
    groups = defaultdict(list)
    for r in rows:
        family = classify(r["model_name"], r["model_id"])
        groups[family].append(r)
    for v in groups.values():
        v.sort(key=lambda r: r["bpp"])
    return groups


def _apply_log_yticks(ax, vmin, vmax):
    ticks = []
    for exp in range(int(math.floor(math.log10(vmin))), int(math.ceil(math.log10(vmax))) + 1):
        for m in [1, 2, 5]:
            t = (10 ** exp) * m
            if vmin * 0.8 <= t <= vmax * 1.2:
                ticks.append(t)
    ax.yaxis.set_major_locator(ticker.FixedLocator(ticks))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.4g}"))


def plot_metric(groups, metric, y_label, title, filename, yscale="linear"):
    fig, ax = plt.subplots(figsize=(12, 7))

    for family, items in groups.items():
        style = MODEL_FAMILIES[family]
        x = [r["bpp"] for r in items]
        y = [r[metric] for r in items]
        if len(x) == 1:
            ax.scatter(x, y, s=72, color=style["color"], marker=style["marker"],
                       edgecolor="white", linewidth=1.0, label=style["label"], zorder=3)
        else:
            ax.plot(x, y, color=style["color"], marker=style["marker"],
                    linewidth=2.2, markersize=7, label=style["label"])

    ax.set_xlabel("BPP (bits per pixel)", fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_yscale(yscale)
    ax.grid(True, which="major", alpha=0.3, linewidth=0.6)
    ax.grid(True, which="minor", alpha=0.08, linewidth=0.3)

    if yscale == "log":
        all_y = [r[metric] for items in groups.values() for r in items if r[metric] > 0]
        if all_y:
            _apply_log_yticks(ax, min(all_y), max(all_y))

    ax.legend(loc="best", fontsize=10, framealpha=0.85)
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUTPUT_DIR / filename}")


def main():
    rows = read_csv(CSV_PATH)
    groups = group_rows(rows)
    groups_no_dcae = {k: v for k, v in groups.items() if k != "DCAE"}

    print("Generating Kodak plots (all models):")
    plot_metric(groups, "psnr", "PSNR (dB)",
                "Kodak — PSNR vs BPP", "psnr_vs_bpp.png")
    plot_metric(groups, "encode_throughput_MBps", "Encode Throughput (MB/s)",
                "Kodak — Encode Throughput vs BPP", "encode_throughput_vs_bpp.png",
                yscale="log")
    plot_metric(groups, "decode_throughput_MBps", "Decode Throughput (MB/s)",
                "Kodak — Decode Throughput vs BPP", "decode_throughput_vs_bpp.png",
                yscale="log")

    print("Generating Kodak plots (without DCAE):")
    plot_metric(groups_no_dcae, "psnr", "PSNR (dB)",
                "Kodak — PSNR vs BPP (w/o DCAE)", "psnr_vs_bpp_no_dcae.png")
    plot_metric(groups_no_dcae, "encode_throughput_MBps", "Encode Throughput (MB/s)",
                "Kodak — Encode Throughput vs BPP (w/o DCAE)", "encode_throughput_vs_bpp_no_dcae.png",
                yscale="log")
    plot_metric(groups_no_dcae, "decode_throughput_MBps", "Decode Throughput (MB/s)",
                "Kodak — Decode Throughput vs BPP (w/o DCAE)", "decode_throughput_vs_bpp_no_dcae.png",
                yscale="log")

    print("Done.")


if __name__ == "__main__":
    main()
