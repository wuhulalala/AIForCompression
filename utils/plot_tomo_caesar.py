"""
Plot CAESAR tomo results: PSNR vs BPP and encode/decode throughput bar chart.
Includes throughput comparison with DCVC-RT and DCMVC from tomo_video_models.

Usage: python utils/plot_tomo_caesar.py
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
CAESAR_PATH = PROJECT_ROOT / "unified_results" / "tomo_caesar_models" / "summary.json"
VIDEO_PATH = PROJECT_ROOT / "unified_results" / "tomo_video_models" / "summary.json"
OUTPUT_DIR = PROJECT_ROOT / "unified_results" / "tomo_caesar_models" / "plots"
CSV_PATH = OUTPUT_DIR / "tomo_caesar_avg.csv"

CAESAR_STYLES = {
    "caesar_d": {"color": "#e74c3c", "marker": "o", "label": "CAESAR-D"},
    "caesar_v": {"color": "#3498db", "marker": "s", "label": "CAESAR-V"},
}

VIDEO_STYLES = {
    "DCVC-RT": {"color": "#dc2626", "marker": "s", "label": "DCVC-RT"},
    "DCMVC":   {"color": "#2563eb", "marker": "o", "label": "DCMVC"},
}

BAR_COLORS = {"encode": "#4C72B0", "decode": "#DD8452"}


def load_summary(path):
    with open(path) as f:
        return json.load(f)


def group_by_model(rows):
    groups = defaultdict(list)
    for r in rows:
        if "error" in r:
            continue
        groups[r["model_id"]].append(r)
    for v in groups.values():
        v.sort(key=lambda r: r.get("bpp", r.get("eb", 0)))
    return groups


def plot_psnr_vs_bpp(groups):
    """PSNR vs BPP for CAESAR-D and CAESAR-V."""
    fig, ax = plt.subplots(figsize=(12, 7))

    for model_id, items in sorted(groups.items()):
        if model_id not in CAESAR_STYLES:
            continue
        style = CAESAR_STYLES[model_id]
        # Sort by eb (smaller eb = higher quality = lower bpp)
        x = [r["bpp"] for r in items]
        y = [r["psnr"] for r in items]
        ax.plot(x, y, color=style["color"], marker=style["marker"],
                linewidth=2.2, markersize=8, label=style["label"])
        # Annotate with eb values
        for xi, yi, r in zip(x, y, items):
            ax.annotate(f"eb={r['eb']:.0e}", (xi, yi),
                        textcoords="offset points", xytext=(8, -4),
                        fontsize=7.5, color=style["color"], alpha=0.85)

    ax.set_xlabel("BPP (bits per pixel)", fontsize=13)
    ax.set_ylabel("PSNR (dB)", fontsize=13)
    ax.set_title("Tomography — CAESAR PSNR vs BPP", fontsize=15, fontweight="bold")
    ax.grid(True, which="major", alpha=0.3, linewidth=0.6)
    ax.grid(True, which="minor", alpha=0.08, linewidth=0.3)
    ax.legend(loc="best", fontsize=10, framealpha=0.85)
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "psnr_vs_bpp.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUTPUT_DIR / 'psnr_vs_bpp.png'}")


def plot_throughput_bar(caesar_groups, video_groups):
    """Combined bar chart: average encode/decode throughput for CAESAR + video models."""
    # Build throughput data: {label: {"encode": val_MBps, "decode": val_MBps}}
    combined = {}

    for model_id, items in caesar_groups.items():
        if model_id not in CAESAR_STYLES:
            continue
        style = CAESAR_STYLES[model_id]
        enc_vals = [r.get("encode_throughput", 0) / 1e6 for r in items]
        dec_vals = [r.get("decode_throughput", 0) / 1e6 for r in items]
        combined[style["label"]] = {
            "encode": np.mean(enc_vals),
            "decode": np.mean(dec_vals),
        }

    for model_id, items in video_groups.items():
        if model_id not in VIDEO_STYLES:
            continue
        style = VIDEO_STYLES[model_id]
        enc_vals = [r.get("encode_throughput_MBps", 0) for r in items]
        dec_vals = [r.get("decode_throughput_MBps", 0) for r in items]
        combined[style["label"]] = {
            "encode": np.mean(enc_vals),
            "decode": np.mean(dec_vals),
        }

    # Sort by total throughput
    names = sorted(combined.keys(),
                   key=lambda n: combined[n]["encode"] + combined[n]["decode"])
    enc_vals = [combined[n]["encode"] for n in names]
    dec_vals = [combined[n]["decode"] for n in names]

    x = np.arange(len(names))
    width = 0.30

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, enc_vals, width, label="Encode",
                   color=BAR_COLORS["encode"], edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, dec_vals, width, label="Decode",
                   color=BAR_COLORS["decode"], edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars1, enc_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar, val in zip(bars2, dec_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xlabel("Model", fontsize=13)
    ax.set_ylabel("Throughput (MB/s)", fontsize=13)
    ax.set_title("Tomography — Average Encode / Decode Throughput", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11, rotation=15, ha="right")
    ax.legend(fontsize=11, framealpha=0.85)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "encode_decode_throughput_bar.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUTPUT_DIR / 'encode_decode_throughput_bar.png'}")


def main():
    caesar_raw = load_summary(CAESAR_PATH)
    caesar_groups = group_by_model(caesar_raw)

    print("CAESAR results:")
    for model_id, items in caesar_groups.items():
        for r in items:
            enc_mb = r.get("encode_throughput", 0) / 1e6
            dec_mb = r.get("decode_throughput", 0) / 1e6
            print(f"  {model_id} eb={r['eb']:.0e}: psnr={r['psnr']:.2f} bpp={r['bpp']:.4f} "
                  f"enc={enc_mb:.1f}MB/s dec={dec_mb:.1f}MB/s")

    video_raw = load_summary(VIDEO_PATH)
    video_groups = group_by_model(video_raw)

    print("\nGenerating plots:")
    plot_psnr_vs_bpp(caesar_groups)
    plot_throughput_bar(caesar_groups, video_groups)
    print("Done.")


if __name__ == "__main__":
    main()
