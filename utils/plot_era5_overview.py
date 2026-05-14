"""
ERA5 overview: PSNR vs BPP and throughput bar chart for ALL models (CAESAR, image, video, CRA5).
Same colors and style as plot_kodak_overview.py.
Usage: python utils/plot_era5_overview.py
"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path("/data/run01/scxj523/zsh/project/AIForCompression")
RESULTS = PROJECT / "logs" / "results"
OUTPUT = PROJECT / "logs" / "results" / "era5_overview"

SOURCES = [
    # Image models
    (RESULTS / "DCAE/summary.json",                "image"),
    (RESULTS / "LIC-HPCM-base/summary.json",       "image"),
    (RESULTS / "LIC-HPCM-large/summary.json",      "image"),
    (RESULTS / "LIC_TCM/summary.json",             "image"),
    (RESULTS / "RwkvCompress/summary.json",        "image"),
    (RESULTS / "WeConvene/summary.json",           "image"),
    # Video models
    (RESULTS / "DCMVC/summary.json",               "video"),
    (RESULTS / "DCVC-RT/summary.json",             "video"),
    # CRA5
    (PROJECT / "models/CRA5/summary.json",         "cra5"),
]

CAT_COLORS = {
    "image":  "#2563eb",
    "video":  "#dc2626",
    "caesar": "#16a34a",
    "cra5":   "#e67e22",
}

MODEL_MARKERS = {
    "DCAE":           "o",
    "WeConvene":      "v",
    "LIC-HPCM-base":  "s",
    "LIC-HPCM-large": "D",
    "LIC-TCM":        "^",
    "LALIC":          "P",
    "DCVC-RT":        "s",
    "DCMVC":          "D",
    "CAESAR-V":       "^",
    "CAESAR-D":       "v",
    "CRA5":           "*",
}

BAR_COLORS = {"encode": "#4C72B0", "decode": "#DD8452"}


def load_image_video_cra5_sources():
    rows = []
    for path, family in SOURCES:
        if not path.exists():
            continue
        data = json.load(open(path))
        if not isinstance(data, list):
            data = [data]
        for r in data:
            if not isinstance(r, dict) or "error" in r:
                continue
            r["_family"] = family
            rows.append(r)
    return rows


def load_caesar_sources():
    rows = []
    for d in sorted(RESULTS.glob("CAESAR*")):
        s = d / "summary.json"
        if not s.exists() or d.name.startswith("caesar_eb_sweep"):
            continue
        eb = parse_eb(d.name)
        for r in json.load(open(s)):
            if not isinstance(r, dict) or "error" in r:
                continue
            r["_family"] = "caesar"
            if eb is not None and r.get("eb") is None:
                r["eb"] = eb
            rows.append(r)
    return rows


def parse_eb(dirname):
    import re
    m = re.search(r"CAESAR_eb_(\d+p?\d*)e?m(\d+)$", dirname)
    if not m:
        return None
    mantissa = m.group(1).replace("p", ".")
    exponent = m.group(2)
    return float(f"{mantissa}e-{exponent}")


def model_label(r):
    name = r.get("model_name", "") or ""
    mid = r.get("model_id", "") or ""
    model = r.get("model", "") or ""

    if name == "LIC-HPCM":
        return "LIC-HPCM-large" if "large" in mid else "LIC-HPCM-base"
    if name == "LIC_TCM":
        if "large" in mid:
            return None
        return "LIC-TCM"
    if name == "RwkvCompress" or mid.startswith("RwkvCompress"):
        return "LALIC"
    if name == "CAESAR" or model == "CAESAR" or mid in ("caesar_v", "caesar_d"):
        return "CAESAR-V" if "caesar_v" in mid else "CAESAR-D"
    if mid.startswith("DCAE"):
        return "DCAE"
    if mid.startswith("DCMVC"):
        return "DCMVC"
    if mid.startswith("DCVC"):
        return "DCVC-RT"
    if mid.startswith("WeConvene"):
        return "WeConvene"
    if mid.startswith("LIC-HPCM-base"):
        return "LIC-HPCM-base"
    if mid.startswith("LIC-HPCM-large"):
        return "LIC-HPCM-large"
    if mid.startswith("LICTCM"):
        return "LIC-TCM"
    if model == "CRA5-VAEformer" or name == "CRA5-VAEformer":
        return "CRA5"
    return name or None


def get_metrics(r):
    # Handle compress sub-dict (some image models)
    c = r.get("compress", {})
    psnr = c.get("psnr") or r.get("psnr")
    bpp = c.get("bpp") or r.get("bpp")
    cr = c.get("compression_ratio") or r.get("compression_ratio")

    # Throughput: detect unit — >10000 is bytes/s, <=10000 is already MB/s
    enc_tp = r.get("encode_throughput", 0) or 0
    dec_tp = r.get("decode_throughput", 0) or 0
    if enc_tp > 10000:
        enc = enc_tp / 1e6
    elif enc_tp > 0:
        enc = enc_tp
    else:
        orig_bytes = (c.get("original_bytes") or r.get("original_bytes") or 0)
        if orig_bytes == 0 and "data_shape" in r:
            ds = r["data_shape"]
            orig_bytes = np.prod([int(d) for d in ds]) * 4
        enc_time = c.get("encode_time") or r.get("encode_time_avg") or r.get("avg_encode_time") or 1
        enc = (orig_bytes / enc_time / 1e6) if enc_time > 0 else 0
    if dec_tp > 10000:
        dec = dec_tp / 1e6
    elif dec_tp > 0:
        dec = dec_tp
    else:
        orig_bytes = (c.get("original_bytes") or r.get("original_bytes") or 0)
        if orig_bytes == 0 and "data_shape" in r:
            ds = r["data_shape"]
            orig_bytes = np.prod([int(d) for d in ds]) * 4
        dec_time = c.get("decode_time") or r.get("decode_time_avg") or r.get("avg_decode_time") or 1
        dec = (orig_bytes / dec_time / 1e6) if dec_time > 0 else 0

    label = model_label(r)
    if psnr is None or bpp is None or label is None:
        return None
    # Convert BPP to BPV: BPV = 32 / compression_ratio (universal, works for any channel count)
    cr = c.get("compression_ratio") or r.get("compression_ratio")
    if cr and float(cr) > 0:
        bpp = 32.0 / float(cr)
    else:
        bpp = float(bpp)
    return {
        "psnr": float(psnr), "bpp": float(bpp),
        "cr": float(cr or 0), "enc": float(enc), "dec": float(dec),
        "label": label, "family": r.get("_family", "?")
    }


def group_into_curves(rows):
    raw = defaultdict(lambda: defaultdict(list))
    for r in rows:
        m = get_metrics(r)
        if m is None:
            continue
        label = m["label"]
        if label in ("CAESAR-V", "CAESAR-D"):
            key = r.get("eb", "?")
        else:
            key = r.get("model_id", r.get("model", "?"))
        raw[(label, m["family"])][key].append(m)

    result = {}
    for (label, family), key_map in raw.items():
        pts = []
        for _, items in key_map.items():
            pts.append({
                "psnr": np.mean([x["psnr"] for x in items]),
                "bpp":  np.mean([x["bpp"] for x in items]),
                "cr":   np.mean([x["cr"] for x in items]),
                "enc":  np.mean([x["enc"] for x in items]),
                "dec":  np.mean([x["dec"] for x in items]),
            })
        pts.sort(key=lambda p: p["bpp"])
        result[(label, family)] = pts
    return result


def plot_psnr_vs_bpp(groups, suffix=""):
    fig, ax = plt.subplots(figsize=(16, 9))
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

    ax.set_xlabel("BPV (bits per value)", fontsize=14)
    ax.set_ylabel("PSNR (dB)", fontsize=14)
    title_text = "ERA5 — All Models PSNR vs BPV" + (" (full range)" if suffix else "")
    ax.set_title(title_text, fontsize=17, fontweight="bold")
    if not suffix:
        ax.set_ylim(42, 92)
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
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fname = f"psnr_vs_bpp_by_category{suffix}.png"
    fig.savefig(OUTPUT / fname, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUTPUT / fname}")


def plot_throughput_bar(groups):
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
    ax.set_title("ERA5 — All Models Encode / Decode Throughput", fontsize=17, fontweight="bold")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11, rotation=20, ha="right")
    for tick, color in zip(ax.get_xticklabels(), tick_colors):
        tick.set_color(color)
    ax.legend(fontsize=12, framealpha=0.85)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT / "encode_decode_throughput_by_category.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUTPUT / 'encode_decode_throughput_by_category.png'}")


def main():
    rows = load_image_video_cra5_sources() + load_caesar_sources()
    print(f"Loaded {len(rows)} rows")
    groups = group_into_curves(rows)
    for (label, family), pts in sorted(groups.items()):
        print(f"  [{family}] {label}: {len(pts)} pts, PSNR={pts[0]['psnr']:.1f}~{pts[-1]['psnr']:.1f}, BPP={pts[0]['bpp']:.4f}~{pts[-1]['bpp']:.4f}")

    print("\nGenerating plots:")
    plot_psnr_vs_bpp(groups)
    plot_psnr_vs_bpp(groups, suffix="_linear")
    plot_throughput_bar(groups)
    print("Done.")


if __name__ == "__main__":
    main()
