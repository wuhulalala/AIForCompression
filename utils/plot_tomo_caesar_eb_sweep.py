"""
Plot CAESAR tomo eb sweep results: PSNR vs BPP and encode/decode throughput bar chart.
Combines high-quality (eb=1e-4..1e-3) and eb sweep (2e-3..2e-2) results.

Usage: python utils/plot_tomo_caesar_eb_sweep.py
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SWEEP_PATH = PROJECT_ROOT / "unified_results" / "tomo_caesar_eb_sweep" / "summary.json"
OUTPUT_DIR = PROJECT_ROOT / "unified_results" / "tomo_caesar_eb_sweep" / "plots"
CSV_PATH = OUTPUT_DIR / "tomo_caesar_eb_sweep.csv"

CAESAR_STYLES = {
    "caesar_d": {"color": "#e74c3c", "marker": "o", "label": "CAESAR-D"},
    "caesar_v": {"color": "#3498db", "marker": "s", "label": "CAESAR-V"},
}

BAR_COLORS = {"encode": "#4C72B0", "decode": "#DD8452"}


def load_summary(path):
    with open(path) as f:
        return json.load(f)


def combine_results(*paths):
    all_rows = []
    for p in paths:
        if p.exists():
            all_rows.extend(load_summary(p))
        else:
            print(f"  SKIP missing: {p}")
    return [r for r in all_rows if "error" not in r]


def group_by_model(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["model_id"]].append(r)
    for v in groups.values():
        v.sort(key=lambda r: r["bpp"])
    return groups


def plot_psnr_vs_bpp(groups):
    """PSNR vs BPP for CAESAR-V and CAESAR-D."""
    fig, ax = plt.subplots(figsize=(12, 7))

    for model_id, items in sorted(groups.items()):
        if model_id not in CAESAR_STYLES:
            continue
        style = CAESAR_STYLES[model_id]
        x = [r["bpp"] for r in items]
        y = [r["psnr"] for r in items]
        ax.plot(x, y, color=style["color"], marker=style["marker"],
                linewidth=2.2, markersize=8, label=style["label"])

    ax.set_xlabel("BPP (bits per pixel)", fontsize=13)
    ax.set_ylabel("PSNR (dB)", fontsize=13)
    ax.set_title("Tomography — CAESAR PSNR vs BPP (eb sweep)", fontsize=15, fontweight="bold")
    ax.set_ylim(40, 65)
    ax.grid(True, which="major", alpha=0.3, linewidth=0.6)
    ax.grid(True, which="minor", alpha=0.08, linewidth=0.3)
    ax.legend(loc="best", fontsize=10, framealpha=0.85)
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "psnr_vs_bpp.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUTPUT_DIR / 'psnr_vs_bpp.png'}")


def plot_throughput_bar(groups):
    """Bar chart: average encode+decode throughput per model (mean ± std across eb values)."""
    names = []
    enc_means = []
    dec_means = []
    enc_stds = []
    dec_stds = []
    for model_id in sorted(groups.keys()):
        if model_id not in CAESAR_STYLES:
            continue
        style = CAESAR_STYLES[model_id]
        enc_arr = [r.get("encode_throughput", 0) / 1e6 for r in groups[model_id]]
        dec_arr = [r.get("decode_throughput", 0) / 1e6 for r in groups[model_id]]
        names.append(style["label"])
        enc_means.append(np.mean(enc_arr))
        dec_means.append(np.mean(dec_arr))
        enc_stds.append(np.std(enc_arr))
        dec_stds.append(np.std(dec_arr))

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
    ax.set_title("Tomography — CAESAR Average Encode / Decode Throughput", fontsize=15, fontweight="bold")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.legend(fontsize=11, framealpha=0.85)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "encode_decode_throughput_bar.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUTPUT_DIR / 'encode_decode_throughput_bar.png'}")


def write_csv(rows, path):
    fields = ["model_id", "eb", "params", "psnr", "bpp", "mse", "rmse", "compression_ratio",
              "encode_time_avg", "decode_time_avg", "encode_throughput_MBps", "decode_throughput_MBps"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["model_id"], r["bpp"])):
            r2 = dict(r)
            r2["encode_throughput_MBps"] = r.get("encode_throughput", 0) / 1e6
            r2["decode_throughput_MBps"] = r.get("decode_throughput", 0) / 1e6
            w.writerow(r2)
    print(f"CSV: {path}")


def main():
    rows = combine_results(SWEEP_PATH)
    groups = group_by_model(rows)

    print("CAESAR tomo eb sweep results:")
    for model_id, items in sorted(groups.items()):
        if model_id not in CAESAR_STYLES:
            continue
        for r in items:
            enc_mb = r.get("encode_throughput", 0) / 1e6
            dec_mb = r.get("decode_throughput", 0) / 1e6
            print(f"  {model_id} eb={r['eb']:.1e}: PSNR={r['psnr']:.2f} bpp={r['bpp']:.4f} "
                  f"CR={r['compression_ratio']:.1f}x enc={enc_mb:.1f}MB/s dec={dec_mb:.1f}MB/s")

    write_csv(rows, CSV_PATH)

    print("\nGenerating plots:")
    plot_psnr_vs_bpp(groups)
    plot_throughput_bar(groups)
    print("Done.")


if __name__ == "__main__":
    main()
