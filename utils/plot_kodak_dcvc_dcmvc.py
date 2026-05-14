"""
Aggregate DCVC-RT + DCMVC Kodak results and plot PSNR vs BPP, Throughput vs BPP,
and throughput bar chart. Style matches kodak_all_models/plots.

Usage: python utils/plot_kodak_dcvc_dcmvc.py
"""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = PROJECT_ROOT / "unified_results" / "kodak_dcvc_dcmvc" / "summary.json"
OUTPUT_DIR = PROJECT_ROOT / "unified_results" / "kodak_dcvc_dcmvc" / "plots"
CSV_PATH = OUTPUT_DIR / "kodak_dcvc_dcmvc_avg.csv"

MODEL_FAMILIES = {
    "DCVC-RT":  {"color": "#dc2626", "marker": "s", "label": "DCVC-RT"},
    "DCMVC":    {"color": "#2563eb", "marker": "o", "label": "DCMVC"},
}

BAR_COLORS = {"encode": "#4C72B0", "decode": "#DD8452"}


def load_summary(path):
    with open(path) as f:
        return json.load(f)


def compute_averages(rows):
    """Average per model_id across all 24 Kodak images."""
    groups = defaultdict(list)
    for r in rows:
        if "error" in r:
            continue
        groups[r["model_id"]].append(r)

    avg_rows = []
    for model_id, items in sorted(groups.items()):
        n = len(items)
        avg = {
            "model_name": items[0]["model_name"],
            "model_id": model_id,
            "params": items[0]["params"],
            "mse": np.mean([it["mse"] for it in items]),
            "rmse": np.mean([it["rmse"] for it in items]),
            "psnr": np.mean([it["psnr"] for it in items]),
            "bpp": np.mean([it["bpp"] for it in items]),
            "compression_ratio": np.mean([it["compression_ratio"] for it in items]),
            "encode_time_avg": np.mean([it["encode_time_avg"] for it in items]),
            "decode_time_avg": np.mean([it["decode_time_avg"] for it in items]),
            "encode_throughput_MBps": np.mean([it["encode_throughput_MBps"] for it in items]),
            "decode_throughput_MBps": np.mean([it["decode_throughput_MBps"] for it in items]),
            "n_samples": n,
        }
        avg_rows.append(avg)
    return avg_rows


def write_csv(rows, path):
    fields = ["model_name", "model_id", "params", "mse", "rmse", "psnr", "bpp",
              "compression_ratio", "encode_time_avg", "decode_time_avg",
              "encode_throughput_MBps", "decode_throughput_MBps"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    print(f"CSV: {path}")


def group_by_model(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["model_name"]].append(r)
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
        ax.plot(x, y, color=style["color"], marker=style["marker"],
                linewidth=2.2, markersize=8, label=style["label"])

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


def plot_throughput_bar(raw_rows):
    """Bar chart: average encode+decode throughput per model family (mean ± std across all q-levels × images)."""
    # Collect per-family raw throughput values (across all images and q-levels)
    family_enc = defaultdict(list)
    family_dec = defaultdict(list)
    for r in raw_rows:
        if "error" in r:
            continue
        family_enc[r["model_name"]].append(r["encode_throughput_MBps"])
        family_dec[r["model_name"]].append(r["decode_throughput_MBps"])

    names = sorted(family_enc.keys())
    enc_means = [np.mean(family_enc[n]) for n in names]
    dec_means = [np.mean(family_dec[n]) for n in names]
    enc_stds = [np.std(family_enc[n]) for n in names]
    dec_stds = [np.std(family_dec[n]) for n in names]

    x = np.arange(len(names))
    width = 0.30

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, enc_means, width, yerr=enc_stds, capsize=5,
                   label="Encode", color=BAR_COLORS["encode"], edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, dec_means, width, yerr=dec_stds, capsize=5,
                   label="Decode", color=BAR_COLORS["decode"], edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars1, enc_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar, val in zip(bars2, dec_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlabel("Model", fontsize=13)
    ax.set_ylabel("Throughput (MB/s)", fontsize=13)
    ax.set_title("Kodak — DCVC-RT / DCMVC Average Throughput", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.legend(fontsize=11, framealpha=0.85)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.6)
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "encode_decode_throughput_bar.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUTPUT_DIR / 'encode_decode_throughput_bar.png'}")


def main():
    raw = load_summary(SUMMARY_PATH)
    avg_rows = compute_averages(raw)
    write_csv(avg_rows, CSV_PATH)

    groups = group_by_model(avg_rows)
    print(f"Models: {list(groups.keys())}")
    for name, items in groups.items():
        for r in items:
            print(f"  {r['model_id']}: psnr={r['psnr']:.2f} bpp={r['bpp']:.4f} "
                  f"enc={r['encode_throughput_MBps']:.1f}MB/s dec={r['decode_throughput_MBps']:.1f}MB/s")

    print("\nGenerating plots:")
    plot_metric(groups, "psnr", "PSNR (dB)",
                "Kodak — DCVC-RT / DCMVC  PSNR vs BPP", "psnr_vs_bpp.png")
    plot_metric(groups, "encode_throughput_MBps", "Encode Throughput (MB/s)",
                "Kodak — DCVC-RT / DCMVC  Encode Throughput vs BPP",
                "encode_throughput_vs_bpp.png", yscale="log")
    plot_metric(groups, "decode_throughput_MBps", "Decode Throughput (MB/s)",
                "Kodak — DCVC-RT / DCMVC  Decode Throughput vs BPP",
                "decode_throughput_vs_bpp.png", yscale="log")
    plot_throughput_bar(raw)
    print("Done.")


if __name__ == "__main__":
    main()
