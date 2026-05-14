"""
Dataset overview: PSNR vs BPP and throughput bar chart for ALL models.
Usage: python utils/plot_kodak_overview.py --dataset kodak|tomo
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path("/data/run01/scxj523/zsh/project/AIForCompression")

DATASET_CONFIG = {
    "kodak": {
        "title": "Kodak",
        "sources": [
            ("unified_results/kodak_all_models/summary.json",       "image"),
            ("unified_results/kodak_dcvc_dcmvc/summary.json",       "video"),
            ("unified_results/kodak_caesar_v/summary.json",         "caesar"),
            ("unified_results/kodak_caesar_d/summary.json",         "caesar"),
            ("unified_results/kodak_caesar_d_bpp/summary.json",     "caesar"),
            ("unified_results/cra5_kodak_stacked/summary.json",     "cra5"),
        ],
        "output": "unified_results/kodak_overview",
        "psnr_ylim": (15, 43),
    },
    "tomo": {
        "title": "Tomography",
        "sources": [
            ("unified_results/tomo_image_models/summary.json",      "image"),
            ("unified_results/tomo_video_models/summary.json",      "video"),
            ("unified_results/tomo_caesar_eb_sweep/summary.json",   "caesar"),
            ("unified_results/cra5_tomo_stacked/summary.json",      "cra5"),
        ],
        "output": "unified_results/tomo_overview",
        "psnr_ylim": (46, 62),
        "bpp_log": True,
    },
}


# Category colors (4 categories → 4 colors)
CAT_COLORS = {
    "image":  "#2563eb",  # blue
    "video":  "#dc2626",  # red
    "caesar": "#16a34a",  # green
    "cra5":   "#e67e22",  # orange
}

# Model markers within each category
MODEL_MARKERS = {
    "DCAE":           "o",
    "WeConvene":      "v",
    "LIC-HPCM-base":  "s",
    "LIC-HPCM-large": "D",
    "LIC-TCM":        "^",
    "LALIC":           "P",
    "DCVC-RT":        "s",
    "DCMVC":          "D",
    "CAESAR-V":       "^",
    "CAESAR-D":       "v",
    "CRA5":           "*",
}

BAR_COLORS = {"encode": "#4C72B0", "decode": "#DD8452"}


def load_all(sources):
    all_rows = []
    for rel_path, family in sources:
        path = PROJECT / rel_path
        if not path.exists():
            print(f"SKIP missing: {path}")
            continue
        with open(path) as f:
            data = json.load(f)
        for r in data:
            if "error" in r:
                continue
            r["_family"] = family
            all_rows.append(r)
    return all_rows


def model_label(r):
    name = r.get("model_name", "") or r.get("model", "")
    mid = r.get("model_id", "") or r.get("model", "")
    if name == "LIC-HPCM":
        return "LIC-HPCM-large" if "large" in mid else "LIC-HPCM-base"
    if name == "LIC_TCM":
        if "large" in mid:
            return None
        return "LIC-TCM"
    if name == "RwkvCompress":
        return "LALIC"
    if name == "CAESAR":
        return "CAESAR-V" if "caesar_v" in mid else "CAESAR-D"
    if name == "CRA5-VAEformer":
        return "CRA5"
    return name


def get_metrics(r):
    psnr = r.get("psnr")
    bpp = r.get("bpp")
    cr = r.get("compression_ratio")
    bpp = float(bpp) if bpp else None
    enc = (r.get("encode_throughput_MBps") or r.get("encode_throughput", 0) / 1e6 or 0)
    dec = (r.get("decode_throughput_MBps") or r.get("decode_throughput", 0) / 1e6 or 0)
    label = model_label(r)
    # CRA5: BPP is computed with 268-channel convention, fix to actual BPP via CR
    if label == "CRA5" and cr and float(cr) > 0:
        bpp = 32.0 / float(cr)
    # CRA5: compute throughput from avg_encode_time / avg_decode_time
    if label == "CRA5" and enc == 0:
        orig_bytes = r.get("original_bytes", 0)
        enc_time = r.get("avg_encode_time") or r.get("encode_time_avg", 1)
        enc = (orig_bytes / enc_time / 1e6) if enc_time > 0 else 0
    if label == "CRA5" and dec == 0:
        orig_bytes = r.get("original_bytes", 0)
        dec_time = r.get("avg_decode_time") or r.get("decode_time_avg", 1)
        dec = (orig_bytes / dec_time / 1e6) if dec_time > 0 else 0
    if psnr is None or bpp is None or label is None:
        return None
    return {"psnr": float(psnr), "bpp": float(bpp), "cr": float(cr or 0),
            "enc": float(enc), "dec": float(dec), "label": label,
            "family": r.get("_family", "?")}


def group_into_curves(rows):
    """Group rows into (label, family, curve_key) and average per-key.
    For image/video: curve_key = model_id (checkpoint)
    For CAESAR:      curve_key = eb (error bound)
    For CRA5:        curve_key = model_id (single point)
    """
    raw = defaultdict(lambda: defaultdict(list))
    for r in rows:
        m = get_metrics(r)
        if m is None:
            continue
        label = m["label"]

        # Determine curve key: checkpoint id for image/video, eb for CAESAR
        if label in ("CAESAR-V", "CAESAR-D"):
            key = r.get("eb", "?")
        else:
            key = r.get("model_id", "?")

        raw[(label, m["family"])][key].append(m)

    # Average per curve_key, then sort by bpp
    result = {}
    for (label, family), key_map in raw.items():
        points = []
        for key, items in key_map.items():
            points.append({
                "psnr": np.mean([x["psnr"] for x in items]),
                "bpp":  np.mean([x["bpp"] for x in items]),
                "cr":   np.mean([x["cr"] for x in items]),
                "enc":  np.mean([x["enc"] for x in items]),
                "dec":  np.mean([x["dec"] for x in items]),
            })
        points.sort(key=lambda p: p["bpp"])
        result[(label, family)] = points
    return result


def plot_psnr_vs_bpp(groups, output_dir, title, psnr_ylim=None, bpp_log=False, psnr_log=False, suffix=""):
    fig, ax = plt.subplots(figsize=(16, 9))
    if bpp_log:
        ax.set_xscale("log")
    if psnr_log:
        ax.set_yscale("log")

    for (label, family) in sorted(groups.keys()):
        pts = groups[(label, family)]
        color = CAT_COLORS.get(family, "#999999")
        marker = MODEL_MARKERS.get(label, "o")
        x = [p["bpp"] for p in pts]
        y = [p["psnr"] for p in pts]

        if len(pts) == 1:
            ax.scatter(x, y, s=160, color=color, marker=marker,
                       edgecolors="white", linewidth=1.0, label=label, zorder=10)
        else:
            ax.plot(x, y, color=color, marker=marker, linestyle="-",
                    linewidth=2.0, markersize=7, label=label)

    ax.set_xlabel("BPP (bits per pixel)", fontsize=14)
    ax.set_ylabel("PSNR (dB)", fontsize=14)
    ax.set_title(f"{title} — All Models PSNR vs BPP", fontsize=17, fontweight="bold")
    if psnr_ylim:
        ax.set_ylim(*psnr_ylim)

    # Annotate outliers below ylim range
    if psnr_ylim:
        outlier_notes = []
        for (label, family), pts in groups.items():
            if not pts:
                continue
            avg_psnr = np.mean([p["psnr"] for p in pts])
            if avg_psnr < psnr_ylim[0]:
                outlier_notes.append((label, avg_psnr, min(p["bpp"] for p in pts), max(p["bpp"] for p in pts)))
        if outlier_notes:
            ax.text(0.02, 0.08,
                    "Below axis:\n" + "\n".join(f"  {n}: PSNR={v:.1f}dB, BPP={b0:.3f}~{b1:.3f}" for n, v, b0, b1 in outlier_notes),
                    transform=ax.transAxes, fontsize=8, color="#999",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9))

    ax.grid(True, which="major", alpha=0.25, linewidth=0.6)
    ax.grid(True, which="minor", alpha=0.06, linewidth=0.3)

    handles, labels = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    cat_handles = [
        Line2D([0],[0], marker='s', color='w', markerfacecolor=CAT_COLORS["image"],  markersize=10, label='Image'),
        Line2D([0],[0], marker='s', color='w', markerfacecolor=CAT_COLORS["video"],  markersize=10, label='Video'),
        Line2D([0],[0], marker='s', color='w', markerfacecolor=CAT_COLORS["caesar"], markersize=10, label='CAESAR'),
        Line2D([0],[0], marker='s', color='w', markerfacecolor=CAT_COLORS["cra5"],   markersize=10, label='CRA5'),
    ]
    leg1 = ax.legend(handles=cat_handles, loc='upper left', fontsize=9, title='Category',
                     title_fontsize=10, framealpha=0.9)
    ax.add_artist(leg1)
    if handles:
        ax.legend(handles=handles, labels=labels, loc='lower right', fontsize=8, ncol=2, framealpha=0.85)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = f"psnr_vs_bpp_by_category{suffix}.png"
    fig.savefig(output_dir / fname, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {output_dir / fname}")


def plot_throughput_bar(groups, output_dir, title):
    combined = {}
    for (label, family), pts in groups.items():
        enc = [p["enc"] for p in pts if p["enc"] > 0]
        dec = [p["dec"] for p in pts if p["dec"] > 0]
        if not enc or not dec:
            continue
        combined[label] = {"encode": np.mean(enc), "decode": np.mean(dec), "family": family}

    names = sorted(combined.keys(), key=lambda n: combined[n]["encode"] + combined[n]["decode"])
    enc_vals = [combined[n]["encode"] for n in names]
    dec_vals = [combined[n]["decode"] for n in names]

    x = np.arange(len(names))
    width = 0.32

    fig, ax = plt.subplots(figsize=(max(10, len(names)*1.1), 7))
    # Color tick labels by category
    tick_colors = [CAT_COLORS.get(combined[n]["family"], "#666") for n in names]

    b1 = ax.bar(x - width/2, enc_vals, width, label="Encode", color=BAR_COLORS["encode"], edgecolor="white")
    b2 = ax.bar(x + width/2, dec_vals, width, label="Decode", color=BAR_COLORS["decode"], edgecolor="white")

    for bar, val in zip(b1, enc_vals):
        t = f"{val:.1f}" if val < 100 else f"{val:.0f}"
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), t,
                ha="center", va="bottom", fontsize=7, fontweight="bold")
    for bar, val in zip(b2, dec_vals):
        t = f"{val:.1f}" if val < 100 else f"{val:.0f}"
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), t,
                ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_ylabel("Throughput (MB/s)", fontsize=14)
    ax.set_title(f"{title} — All Models Encode / Decode Throughput", fontsize=17, fontweight="bold")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11, rotation=20, ha="right")
    for tick, color in zip(ax.get_xticklabels(), tick_colors):
        tick.set_color(color)
    ax.legend(fontsize=12, framealpha=0.85)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "encode_decode_throughput_by_category.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {output_dir / 'encode_decode_throughput_by_category.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="kodak", choices=["kodak", "tomo"])
    args = parser.parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    title = cfg["title"]
    output = PROJECT / cfg["output"]

    rows = load_all(cfg["sources"])
    print(f"Dataset: {args.dataset}, loaded {len(rows)} rows")
    groups = group_into_curves(rows)
    for (label, family), pts in sorted(groups.items()):
        print(f"  [{family}] {label}: {len(pts)} pts, PSNR={pts[0]['psnr']:.1f}~{pts[-1]['psnr']:.1f}, BPP={pts[0]['bpp']:.4f}~{pts[-1]['bpp']:.4f}")

    print("\nGenerating plots:")
    plot_psnr_vs_bpp(groups, output, title, cfg.get("psnr_ylim"), cfg.get("bpp_log", False), cfg.get("psnr_log", False))
    # Extra linear-linear version (no ylim, all models visible)
    if cfg.get("bpp_log") or cfg.get("psnr_log") or cfg.get("psnr_ylim"):
        plot_psnr_vs_bpp(groups, output, title, None, False, False, suffix="_linear")
    plot_throughput_bar(groups, output, title)
    print("Done.")


if __name__ == "__main__":
    main()
