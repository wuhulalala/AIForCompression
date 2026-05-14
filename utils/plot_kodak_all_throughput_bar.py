"""
Plot average encode/decode throughput bar chart for kodak_all_models image models.
Usage: python utils/plot_kodak_all_throughput_bar.py
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "unified_results" / "kodak_all_models" / "kodak_avg_summary.csv"
OUTPUT_DIR = PROJECT_ROOT / "unified_results" / "kodak_all_models" / "plots"

BAR_COLORS = {"encode": "#4C72B0", "decode": "#DD8452"}


def read_csv(path):
    rows = []
    with open(path) as f:
        for d in csv.DictReader(f):
            for k in ["encode_throughput_MBps", "decode_throughput_MBps"]:
                d[k] = float(d[k])
            rows.append(d)
    return rows


def classify(model_name, model_id):
    if model_name == "LIC-HPCM":
        return "LIC-HPCM-base" if "base" in model_id else "LIC-HPCM-large"
    if model_name == "RwkvCompress":
        return "LALIC"
    return model_name


def main():
    rows = read_csv(CSV_PATH)
    groups = defaultdict(list)
    for r in rows:
        groups[classify(r["model_name"], r["model_id"])].append(r)

    names = sorted(groups.keys(),
                   key=lambda n: np.mean([r["encode_throughput_MBps"] for r in groups[n]])
                                 + np.mean([r["decode_throughput_MBps"] for r in groups[n]]))

    enc_means = [np.mean([r["encode_throughput_MBps"] for r in groups[n]]) for n in names]
    dec_means = [np.mean([r["decode_throughput_MBps"] for r in groups[n]]) for n in names]
    enc_stds = [np.std([r["encode_throughput_MBps"] for r in groups[n]]) for n in names]
    dec_stds = [np.std([r["decode_throughput_MBps"] for r in groups[n]]) for n in names]

    x = np.arange(len(names))
    width = 0.30

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, enc_means, width, yerr=enc_stds, capsize=4,
                   label="Encode", color=BAR_COLORS["encode"],
                   edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, dec_means, width, yerr=dec_stds, capsize=4,
                   label="Decode", color=BAR_COLORS["decode"],
                   edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars1, enc_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar, val in zip(bars2, dec_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Model", fontsize=13)
    ax.set_ylabel("Throughput (MB/s)", fontsize=13)
    ax.set_title("Kodak — Average Encode / Decode Throughput", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.legend(fontsize=11, framealpha=0.85)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.6)
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "encode_decode_throughput_bar.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'encode_decode_throughput_bar.png'}")


if __name__ == "__main__":
    main()
