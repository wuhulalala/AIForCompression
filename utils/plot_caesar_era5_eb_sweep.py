"""
Plot ERA5 CAESAR eb sweep results: PSNR vs BPP and encode/decode throughput bar.
Usage: python utils/plot_caesar_era5_eb_sweep.py
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "logs" / "results"
OUTPUT_DIR = PROJECT_ROOT / "logs" / "results" / "caesar_eb_sweep_plots"

CAESAR_STYLES = {
    "CAESAR-D": {"color": "#e74c3c", "marker": "o", "label": "CAESAR-D"},
    "CAESAR-V": {"color": "#3498db", "marker": "s", "label": "CAESAR-V"},
}
BAR_COLORS = {"encode": "#4C72B0", "decode": "#DD8452"}


def parse_eb(dirname):
    m = re.search(r"CAESAR_eb_(\d+p?\d*)e?m(\d+)$", dirname)
    if not m:
        return None
    mantissa = m.group(1).replace("p", ".")
    exponent = m.group(2)
    return float(f"{mantissa}e-{exponent}")


def collect_results():
    groups = defaultdict(list)
    for d in sorted(RESULTS_DIR.glob("CAESAR_eb_*")):
        eb = parse_eb(d.name)
        if eb is None:
            continue
        summary = d / "summary.json"
        if not summary.exists():
            continue
        for r in json.load(open(summary)):
            if "error" in r:
                continue
            r["eb"] = eb
            r["bpv"] = 32.0 / r["compression_ratio"] if r.get("compression_ratio", 0) > 0 else None
            groups[r["model_id"]].append(r)
    for v in groups.values():
        v.sort(key=lambda r: r["eb"])
    return groups


def plot_psnr_vs_bpv(groups):
    fig, ax = plt.subplots(figsize=(12, 7))
    for model_id, items in sorted(groups.items()):
        label = "CAESAR-D" if model_id == "caesar_d" else "CAESAR-V"
        style = CAESAR_STYLES[label]
        x = [r["bpv"] for r in items]
        y = [r["psnr"] for r in items]
        ax.plot(x, y, color=style["color"], marker=style["marker"],
                linewidth=2.2, markersize=8, label=style["label"])

    ax.set_xlabel("BPV (bits per value)", fontsize=13)
    ax.set_ylabel("PSNR (dB)", fontsize=13)
    ax.set_title("ERA5 — CAESAR PSNR vs BPV (eb sweep)", fontsize=15, fontweight="bold")
    ax.set_ylim(54, 92)
    ax.grid(True, which="major", alpha=0.3, linewidth=0.6)
    ax.grid(True, which="minor", alpha=0.08, linewidth=0.3)
    ax.legend(loc="best", fontsize=10, framealpha=0.85)
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "psnr_vs_bpv.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUTPUT_DIR / 'psnr_vs_bpv.png'}")


def plot_throughput_bar(groups):
    combined = {}
    for model_id, items in groups.items():
        label = "CAESAR-D" if model_id == "caesar_d" else "CAESAR-V"
        style = CAESAR_STYLES[label]
        enc = [r["encode_throughput"] / 1e6 for r in items]
        dec = [r["decode_throughput"] / 1e6 for r in items]
        combined[style["label"]] = {
            "encode": np.mean(enc), "decode": np.mean(dec),
            "enc_std": np.std(enc), "dec_std": np.std(dec),
        }

    names = sorted(combined.keys(),
                   key=lambda n: combined[n]["encode"] + combined[n]["decode"])
    enc_vals = [combined[n]["encode"] for n in names]
    dec_vals = [combined[n]["decode"] for n in names]
    enc_stds = [combined[n]["enc_std"] for n in names]
    dec_stds = [combined[n]["dec_std"] for n in names]

    x = np.arange(len(names))
    width = 0.28

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.bar(x - width / 2, enc_vals, width, yerr=enc_stds, capsize=5,
           label="Encode", color=BAR_COLORS["encode"], edgecolor="white", linewidth=0.5)
    ax.bar(x + width / 2, dec_vals, width, yerr=dec_stds, capsize=5,
           label="Decode", color=BAR_COLORS["decode"], edgecolor="white", linewidth=0.5)

    for bar, val in zip(ax.patches[::2], enc_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    for bar, val in zip(ax.patches[1::2], dec_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xlabel("Model", fontsize=13)
    ax.set_ylabel("Throughput (MB/s)", fontsize=13)
    ax.set_title("ERA5 — CAESAR Average Encode / Decode Throughput", fontsize=13, fontweight="bold")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_xlim(-0.45, 1.45)
    ax.legend(fontsize=11, framealpha=0.85)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "encode_decode_throughput_bar.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUTPUT_DIR / 'encode_decode_throughput_bar.png'}")


def main():
    groups = collect_results()
    print("CAESAR ERA5 eb sweep:")
    for model_id, items in sorted(groups.items()):
        label = "CAESAR-D" if model_id == "caesar_d" else "CAESAR-V"
        for r in items:
            print(f"  {label} eb={r['eb']:.1e}: PSNR={r['psnr']:.2f} CR={r['compression_ratio']:.1f} BPV={r['bpv']:.4f}")

    print("\nGenerating plots:")
    plot_psnr_vs_bpv(groups)
    plot_throughput_bar(groups)
    print("Done.")


if __name__ == "__main__":
    main()
