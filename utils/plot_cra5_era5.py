"""
CRA5 standalone plot for ERA5.
Usage: python utils/plot_cra5_era5.py
"""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path("/data/run01/scxj523/zsh/project/AIForCompression")
CRA5_PATH = PROJECT / "models" / "CRA5" / "summary.json"
OUTPUT = PROJECT / "models" / "CRA5" / "plots"

BAR_COLORS = {"encode": "#4C72B0", "decode": "#DD8452"}


def main():
    data = json.load(open(CRA5_PATH))
    if not isinstance(data, list):
        data = [data]

    # PSNR vs BPV scatter (single point with annotation)
    print(f"CRA5 ERA5: {len(data)} point(s)")
    psnr = float(data[0]["psnr"])
    cr = float(data[0]["compression_ratio"])
    bpv = 32.0 / cr  # BPV for float32
    enc = (np.prod([int(d) for d in data[0]["data_shape"]]) * 4) / data[0]["avg_encode_time"] / 1e6
    dec = (np.prod([int(d) for d in data[0]["data_shape"]]) * 4) / data[0]["avg_decode_time"] / 1e6

    # PSNR vs BPV
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter([bpv], [psnr], s=200, color="#e67e22", marker="*",
               edgecolors="#e67e22", linewidth=2, zorder=10, label="CRA5")
    ax.set_xlabel("BPV (bits per value)", fontsize=13)
    ax.set_ylabel("PSNR (dB)", fontsize=13)
    ax.set_title("ERA5 — CRA5 PSNR vs BPV", fontsize=15, fontweight="bold")
    ax.set_xlim(0, bpv * 2)
    ax.set_ylim(psnr - 5, psnr + 5)
    ax.legend(fontsize=11)
    ax.annotate(f"PSNR={psnr:.1f} dB\nBPV={bpv:.4f}\nCR={cr:.1f}",
                (bpv, psnr), textcoords="offset points", xytext=(15, 15),
                fontsize=10, bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / "psnr_vs_bpv.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUTPUT / 'psnr_vs_bpv.png'}")

    # Throughput bar
    fig, ax = plt.subplots(figsize=(4, 5))
    ax.bar([0], [enc], 0.3, color=BAR_COLORS["encode"], label="Encode", edgecolor="white")
    ax.bar([1], [dec], 0.3, color=BAR_COLORS["decode"], label="Decode", edgecolor="white")
    ax.text(0, enc, f"{enc:.0f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.text(1, dec, f"{dec:.0f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Encode", "Decode"])
    ax.set_ylabel("Throughput (MB/s)", fontsize=13)
    ax.set_title("ERA5 — CRA5 Throughput", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT / "encode_decode_throughput_bar.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUTPUT / 'encode_decode_throughput_bar.png'}")
    print(f"PSNR={psnr:.1f}, BPV={bpv:.4f}, Enc={enc:.0f}MB/s, Dec={dec:.0f}MB/s")


if __name__ == "__main__":
    main()
