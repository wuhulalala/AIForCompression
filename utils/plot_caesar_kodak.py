"""
Plot CAESAR-D and CAESAR-V Kodak results: PSNR vs BPP, and combined throughput bar chart
with all models from kodak_all_models.
Usage: python utils/plot_caesar_kodak.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "unified_results" / "kodak_caesar_plots"

CAESAR_PATHS = {
    "CAESAR-D": PROJECT_ROOT / "unified_results" / "kodak_caesar_d_bpp" / "summary.json",
    "CAESAR-V": PROJECT_ROOT / "unified_results" / "kodak_caesar_v" / "summary.json",
}

CAESAR_STYLES = {
    "CAESAR-D": {"color": "#e74c3c", "marker": "o", "label": "CAESAR-D"},
    "CAESAR-V": {"color": "#3498db", "marker": "s", "label": "CAESAR-V"},
}

BAR_COLORS = {"encode": "#4C72B0", "decode": "#DD8452"}


def load_caesar_results(path):
    with open(path) as f:
        return [r for r in json.load(f) if "error" not in r]


def plot_psnr_vs_bpp(all_data):
    """PSNR vs BPP line plot, following kodak_all_models/plots style."""
    fig, ax = plt.subplots(figsize=(12, 7))

    for model_name, rows in all_data.items():
        style = CAESAR_STYLES[model_name]
        rows_sorted = sorted(rows, key=lambda r: r["bpp"])
        x = [r["bpp"] for r in rows_sorted]
        y = [r["psnr"] for r in rows_sorted]
        ax.plot(x, y, color=style["color"], marker=style["marker"],
                linewidth=2.2, markersize=7, label=style["label"])

    ax.set_xlabel("BPP (bits per pixel)", fontsize=13)
    ax.set_ylabel("PSNR (dB)", fontsize=13)
    ax.set_title("Kodak — CAESAR PSNR vs BPP", fontsize=15, fontweight="bold")
    ax.grid(True, which="major", alpha=0.3, linewidth=0.6)
    ax.grid(True, which="minor", alpha=0.08, linewidth=0.3)
    ax.legend(loc="best", fontsize=10, framealpha=0.85)
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "psnr_vs_bpp.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUTPUT_DIR / 'psnr_vs_bpp.png'}")


def plot_throughput_bar(caesar_data):
    """Bar chart: average encode/decode throughput for CAESAR models only."""
    combined = {}
    for name, rows in caesar_data.items():
        combined[name] = {
            "encode": [r["encode_throughput"] / 1e6 for r in rows],
            "decode": [r["decode_throughput"] / 1e6 for r in rows],
        }

    model_names = sorted(combined.keys(),
                         key=lambda n: np.mean(combined[n]["encode"]) + np.mean(combined[n]["decode"]))

    enc_means = [np.mean(combined[n]["encode"]) for n in model_names]
    dec_means = [np.mean(combined[n]["decode"]) for n in model_names]
    enc_stds = [np.std(combined[n]["encode"]) for n in model_names]
    dec_stds = [np.std(combined[n]["decode"]) for n in model_names]

    x = np.arange(len(model_names))
    width = 0.28

    fig, ax = plt.subplots(figsize=(5.5, 5))
    bars1 = ax.bar(x - width / 2, enc_means, width, yerr=enc_stds, capsize=5,
                   label="Encode", color=BAR_COLORS["encode"],
                   edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, dec_means, width, yerr=dec_stds, capsize=5,
                   label="Decode", color=BAR_COLORS["decode"],
                   edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars1, enc_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    for bar, val in zip(bars2, dec_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xlabel("Model", fontsize=13)
    ax.set_ylabel("Throughput (MB/s)", fontsize=13)
    ax.set_title("Kodak — CAESAR Average Encode / Decode Throughput", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=12)
    ax.set_xlim(-0.45, 1.45)
    ax.legend(fontsize=11, framealpha=0.85)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "encode_decode_throughput_bar.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUTPUT_DIR / 'encode_decode_throughput_bar.png'}")


def main():
    caesar_data = {}
    for model_name, path in CAESAR_PATHS.items():
        rows = load_caesar_results(path)
        print(f"Loaded {len(rows)} CAESAR results for {model_name}")
        caesar_data[model_name] = rows

    print("\nGenerating CAESAR Kodak plots:")
    plot_psnr_vs_bpp(caesar_data)
    plot_throughput_bar(caesar_data)
    print("Done.")


if __name__ == "__main__":
    main()
