"""
Plot tomo image models results: PSNR vs BPP and encode/decode throughput bar chart.

Usage: python utils/plot_tomo_image.py
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = PROJECT_ROOT / "unified_results" / "tomo_image_models" / "summary.json"
OUTPUT_DIR = PROJECT_ROOT / "unified_results" / "tomo_image_models" / "plots"

# Model family styles (by model_name)
MODEL_STYLES = {
    "DCAE":            {"color": "#e74c3c", "marker": "o", "label": "DCAE"},
    "LIC-HPCM":        {"color": "#3498db", "marker": "s", "label": "LIC-HPCM"},
    "LIC_TCM":         {"color": "#2ecc71", "marker": "^", "label": "LIC-TCM"},
    "RwkvCompress":    {"color": "#9b59b6", "marker": "D", "label": "LALIC"},
    "WeConvene":       {"color": "#e67e22", "marker": "v", "label": "WeConvene"},
}

# Separate LIC-HPCM base vs large
MODEL_STYLES = {
    "DCAE":                  {"color": "#e74c3c", "marker": "o", "label": "DCAE"},
    "LIC-HPCM-base":         {"color": "#3498db", "marker": "s", "label": "LIC-HPCM-base"},
    "LIC-HPCM-large":        {"color": "#1abc9c", "marker": "s", "label": "LIC-HPCM-large"},
    "LIC_TCM":               {"color": "#2ecc71", "marker": "^", "label": "LIC-TCM"},
    "RwkvCompress":          {"color": "#9b59b6", "marker": "D", "label": "LALIC"},
    "WeConvene":             {"color": "#e67e22", "marker": "v", "label": "WeConvene"},
}

BAR_COLORS = {"encode": "#4C72B0", "decode": "#DD8452"}


def load_summary(path):
    with open(path) as f:
        return json.load(f)


def get_family_key(r):
    """Map model_id to a display family key."""
    name = r.get("model_name", "")
    mid = r.get("model_id", "")
    if name == "LIC-HPCM":
        if "large" in mid:
            return "LIC-HPCM-large"
        return "LIC-HPCM-base"
    if name == "LIC_TCM":
        if "large" in mid:
            return "LIC_TCM-large"
        return "LIC_TCM"
    return name


def group_by_family(rows):
    groups = defaultdict(list)
    for r in rows:
        if "error" in r:
            continue
        key = get_family_key(r)
        groups[key].append(r)
    # For each family, aggregate across checkpoints — average over samples for each checkpoint
    # Then sort by bpp
    result = {}
    for key, items in groups.items():
        # Group by model_id (checkpoint) and average
        by_ckpt = defaultdict(list)
        for r in items:
            by_ckpt[r["model_id"]].append(r)
        agg = []
        for ckpt, ckpt_items in by_ckpt.items():
            psnr = np.mean([r["psnr"] for r in ckpt_items])
            bpp = np.mean([r["bpp"] for r in ckpt_items])
            enc = np.mean([r.get("encode_throughput_MBps", r.get("encode_throughput", 0) / 1e6) for r in ckpt_items])
            dec = np.mean([r.get("decode_throughput_MBps", r.get("decode_throughput", 0) / 1e6) for r in ckpt_items])
            agg.append({"psnr": psnr, "bpp": bpp, "enc": enc, "dec": dec, "model_id": ckpt})
        agg.sort(key=lambda x: x["bpp"])
        result[key] = agg
    return result


def plot_psnr_vs_bpp(groups):
    """PSNR vs BPP for all image models."""
    fig, ax = plt.subplots(figsize=(12, 7))

    for key in sorted(groups.keys()):
        items = groups[key]
        style = MODEL_STYLES.get(key)
        if style is None:
            continue
        x = [r["bpp"] for r in items]
        y = [r["psnr"] for r in items]
        ax.plot(x, y, color=style["color"], marker=style["marker"],
                linewidth=2.2, markersize=8, label=style["label"])

    ax.set_xlabel("BPP (bits per pixel)", fontsize=13)
    ax.set_ylabel("PSNR (dB)", fontsize=13)
    ax.set_title("Tomography — Image Models PSNR vs BPP", fontsize=15, fontweight="bold")
    ax.set_ylim(45, 55)
    ax.grid(True, which="major", alpha=0.3, linewidth=0.6)
    ax.grid(True, which="minor", alpha=0.08, linewidth=0.3)
    ax.legend(loc="best", fontsize=10, framealpha=0.85)
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "psnr_vs_bpp.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUTPUT_DIR / 'psnr_vs_bpp.png'}")


def plot_throughput_bar(groups):
    """Bar chart: average encode/decode throughput for each model family with std error bars."""
    combined = {}

    for key, items in groups.items():
        style = MODEL_STYLES.get(key)
        if style is None:
            continue
        enc_vals = [r["enc"] for r in items]
        dec_vals = [r["dec"] for r in items]
        combined[style["label"]] = {
            "encode": np.mean(enc_vals),
            "decode": np.mean(dec_vals),
            "enc_std": np.std(enc_vals),
            "dec_std": np.std(dec_vals),
        }

    # Sort by total throughput
    names = sorted(combined.keys(),
                   key=lambda n: combined[n]["encode"] + combined[n]["decode"])
    enc_vals = [combined[n]["encode"] for n in names]
    dec_vals = [combined[n]["decode"] for n in names]
    enc_stds = [combined[n]["enc_std"] for n in names]
    dec_stds = [combined[n]["dec_std"] for n in names]

    x = np.arange(len(names))
    width = 0.30

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, enc_vals, width, yerr=enc_stds, capsize=5,
                   label="Encode",
                   color=BAR_COLORS["encode"], edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, dec_vals, width, yerr=dec_stds, capsize=5,
                   label="Decode",
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
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11, rotation=15, ha="right")
    ax.legend(fontsize=11, framealpha=0.85)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "encode_decode_throughput_bar.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUTPUT_DIR / 'encode_decode_throughput_bar.png'}")


def main():
    raw = load_summary(SUMMARY_PATH)
    groups = group_by_family(raw)

    print("Image model results (per-checkpoint averages):")
    for key in sorted(groups.keys()):
        style = MODEL_STYLES.get(key, {})
        for r in groups[key]:
            print(f"  {style.get('label', key)} {r['model_id']}: psnr={r['psnr']:.2f} bpp={r['bpp']:.4f} "
                  f"enc={r['enc']:.1f}MB/s dec={r['dec']:.1f}MB/s")

    print("\nGenerating plots:")
    plot_psnr_vs_bpp(groups)
    plot_throughput_bar(groups)
    print("Done.")


if __name__ == "__main__":
    main()
