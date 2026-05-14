"""
Plot tomo video model results: PSNR vs BPP and encode/decode throughput bar chart.
Style matches kodak_dcvc_dcmvc/plots.

Usage: python utils/plot_tomo_video.py
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
SUMMARY_PATH = PROJECT_ROOT / "unified_results" / "tomo_video_models" / "summary.json"
OUTPUT_DIR = PROJECT_ROOT / "unified_results" / "tomo_video_models" / "plots"
CSV_PATH = OUTPUT_DIR / "tomo_video_avg.csv"

MODEL_FAMILIES = {
    "DCMVC":   {"color": "#2563eb", "marker": "o", "label": "DCMVC"},
}

BAR_COLORS = {"encode": "#4C72B0", "decode": "#DD8452"}


def load_summary(path):
    with open(path) as f:
        return json.load(f)


def compute_averages(rows):
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


def plot_psnr_vs_bpp(groups):
    """PSNR vs BPP."""
    fig, ax = plt.subplots(figsize=(12, 7))

    for family, items in groups.items():
        style = MODEL_FAMILIES.get(family)
        if style is None:
            continue
        x = [r["bpp"] for r in items]
        y = [r["psnr"] for r in items]
        ax.plot(x, y, color=style["color"], marker=style["marker"],
                linewidth=2.2, markersize=8, label=style["label"])

    ax.set_xlabel("BPP (bits per pixel)", fontsize=13)
    ax.set_ylabel("PSNR (dB)", fontsize=13)
    ax.set_title("Tomography — PSNR vs BPP (3-frame stacked)", fontsize=15, fontweight="bold")
    ax.set_ylim(48, 55)
    ax.grid(True, which="major", alpha=0.3, linewidth=0.6)
    ax.grid(True, which="minor", alpha=0.08, linewidth=0.3)
    ax.legend(loc="best", fontsize=10, framealpha=0.85)
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "psnr_vs_bpp.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUTPUT_DIR / 'psnr_vs_bpp.png'}")


def plot_throughput_bar(rows):
    """Bar chart: average encode+decode throughput per model family."""
    rows = [r for r in rows if r["model_name"] != "DCVC-RT"]
    # Group by model family
    groups = defaultdict(list)
    for r in rows:
        groups[r["model_name"]].append(r)
    # Average across quality levels per family
    combined = {}
    for name, items in groups.items():
        style = MODEL_FAMILIES.get(name)
        if style is None:
            continue
        enc_vals = [r["encode_throughput_MBps"] for r in items]
        dec_vals = [r["decode_throughput_MBps"] for r in items]
        combined[style["label"]] = {
            "encode": np.mean(enc_vals),
            "decode": np.mean(dec_vals),
            "enc_std": np.std(enc_vals),
            "dec_std": np.std(dec_vals),
        }

    names = sorted(combined.keys(),
                   key=lambda n: combined[n]["encode"] + combined[n]["decode"])
    enc_vals = [combined[n]["encode"] for n in names]
    dec_vals = [combined[n]["decode"] for n in names]
    enc_stds = [combined[n]["enc_std"] for n in names]
    dec_stds = [combined[n]["dec_std"] for n in names]

    x = np.arange(len(names))
    width = 0.22

    fig, ax = plt.subplots(figsize=(5, 5))
    bars1 = ax.bar(x - width / 2, enc_vals, width, yerr=enc_stds, capsize=5,
                   label="Encode",
                   color=BAR_COLORS["encode"], edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, dec_vals, width, yerr=dec_stds, capsize=5,
                   label="Decode",
                   color=BAR_COLORS["decode"], edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars1, enc_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    for bar, val in zip(bars2, dec_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xlabel("Model", fontsize=13)
    ax.set_ylabel("Throughput (MB/s)", fontsize=13)
    ax.set_title("Tomography — Average Encode / Decode Throughput (3-frame stacked)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_xlim(-0.45, 0.45)
    ax.legend(fontsize=11, framealpha=0.85)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.6)
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "encode_decode_throughput_bar.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUTPUT_DIR / 'encode_decode_throughput_bar.png'}")


def main():
    raw = load_summary(SUMMARY_PATH)
    raw = [r for r in raw if r.get("model_name") != "DCVC-RT"]
    avg_rows = compute_averages(raw)
    write_csv(avg_rows, CSV_PATH)

    groups = group_by_model(avg_rows)
    print(f"Models: {list(groups.keys())}")
    for name, items in groups.items():
        for r in items:
            print(f"  {r['model_id']}: psnr={r['psnr']:.2f} bpp={r['bpp']:.4f} "
                  f"enc={r['encode_throughput_MBps']:.1f}MB/s dec={r['decode_throughput_MBps']:.1f}MB/s")

    print("\nGenerating plots:")
    plot_psnr_vs_bpp(groups)
    plot_throughput_bar(avg_rows)
    print("Done.")


if __name__ == "__main__":
    main()
