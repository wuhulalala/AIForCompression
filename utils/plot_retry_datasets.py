"""Plot retry results for NYX and Shanghai Xray with cleaner style."""
import json, os, math
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT = Path("/data/run01/scxj523/zsh/project/AIForCompression")
CAT_COLORS = {"image": "#2563eb", "video": "#dc2626"}
BAR_COLORS = {"encode": "#4C72B0", "decode": "#DD8452"}
MODEL_MARKERS = {"DCAE": "o", "LIC-HPCM-base": "s", "LIC-HPCM-large": "D", "DCMVC": "s"}

DATASETS = {
    "Shanghai_Xray": {
        "title": "Shanghai X-ray SAXS (retest)",
        "sources": [("unified_results/shanghai_xray_retry/summary.json", "image")],
        "output": "unified_results/shanghai_xray_retry_overview",
    },
    "NYX": {
        "title": "NYX baryon density (retest)",
        "sources": [("unified_results/nyx_retry/summary.json", "image")],
        "output": "unified_results/nyx_retry_overview",
    },
}

def load_all(sources):
    all_rows = []
    for rel_path, family in sources:
        path = PROJECT / rel_path
        if not path.exists(): continue
        data = json.load(open(path))
        for r in data:
            if "error" in r: continue
            if "DCVC_RT_Intra" in r.get("model_id","") and (r.get("bpp",0) or 0) > 3: continue
            if family == "image" and r.get("model_name","") in ("DCMVC","DCVC-RT"):
                r["_family"] = "video"
            else:
                r["_family"] = family
            all_rows.append(r)
    return all_rows

def model_label(r):
    name = r.get("model_name","") or r.get("model","")
    mid = r.get("model_id","") or r.get("model","")
    if name == "LIC-HPCM": return "LIC-HPCM-large" if "large" in mid else "LIC-HPCM-base"
    if name == "DCMVC": return "DCMVC"
    return name

def group_into_curves(rows):
    raw = defaultdict(lambda: defaultdict(list))
    for r in rows:
        label = model_label(r)
        if label is None: continue
        key = r.get("model_id","?")
        psnr = r.get("psnr"); bpp = r.get("bpp"); mse = r.get("mse")
        if mse is not None and mse < 1e-12: continue  # skip perfect-recon tiles
        if psnr is None or bpp is None or not np.isfinite(float(psnr)): continue
        enc = r.get("encode_throughput_MBps") or 0
        dec = r.get("decode_throughput_MBps") or 0
        if enc == 0:
            ob = r.get("original_bytes",0)
            et = r.get("encode_time_avg",0)
            enc = (ob/et/1e6) if et > 0 else 0
        if dec == 0:
            ob = r.get("original_bytes",0)
            dt = r.get("decode_time_avg",0)
            dec = (ob/dt/1e6) if dt > 0 else 0
        raw[(label, r["_family"])][key].append({
            "psnr": float(psnr), "bpp": float(bpp),
            "cr": float(r.get("compression_ratio",0) or 1),
            "enc": float(enc), "dec": float(dec),
        })
    result = {}
    for (label, family), key_map in raw.items():
        pts = []
        for key, items in key_map.items():
            pts.append({k: np.mean([x[k] for x in items]) for k in ["psnr","bpp","cr","enc","dec"]})
        pts.sort(key=lambda p: p["bpp"])
        if pts: result[(label,family)] = pts
    return result

def plot_psnr_vs_bpp(groups, output_dir, title):
    fig, ax = plt.subplots(figsize=(14, 8))
    for (label, family) in sorted(groups.keys()):
        pts = groups[(label, family)]
        color = CAT_COLORS.get(family, "#999")
        marker = MODEL_MARKERS.get(label, "o")
        x, y = [p["bpp"] for p in pts], [p["psnr"] for p in pts]
        ax.plot(x, y, color=color, marker=marker, linestyle="-", linewidth=2.5, markersize=10, label=label, zorder=5)
        for px, py in zip(x, y):
            ax.annotate(f"{py:.1f}", (px, py), textcoords="offset points", xytext=(0,10), fontsize=7, ha='center')

    ax.set_xlabel("BPP (bits per pixel)", fontsize=14)
    ax.set_ylabel("PSNR (dB)", fontsize=14)
    ax.set_title(f"{title}  —  PSNR vs BPP", fontsize=17, fontweight="bold")
    ax.grid(True, alpha=0.25)
    cat_handles = [
        Line2D([0],[0], marker='s', color='w', markerfacecolor=CAT_COLORS["image"], markersize=10, label='Image Models'),
        Line2D([0],[0], marker='s', color='w', markerfacecolor=CAT_COLORS["video"], markersize=10, label='Video Models'),
    ]
    leg1 = ax.legend(handles=cat_handles, loc='upper left', fontsize=9, title='Category', title_fontsize=10)
    ax.add_artist(leg1)
    handles, labels = ax.get_legend_handles_labels()
    if handles: ax.legend(handles=handles, labels=labels, loc='lower right', fontsize=8)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "psnr_vs_bpp.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {output_dir / 'psnr_vs_bpp.png'}")

def plot_throughput_bar(groups, output_dir, title):
    combined = {}
    for (label, family), pts in groups.items():
        enc = [p["enc"] for p in pts if p["enc"] > 0]
        dec = [p["dec"] for p in pts if p["dec"] > 0]
        if not enc or not dec: continue
        combined[label] = {"encode": np.mean(enc), "decode": np.mean(dec), "family": family}
    if not combined: return
    names = sorted(combined.keys(), key=lambda n: combined[n]["encode"] + combined[n]["decode"])
    enc_vals = [combined[n]["encode"] for n in names]
    dec_vals = [combined[n]["decode"] for n in names]
    x = np.arange(len(names)); width = 0.32
    fig, ax = plt.subplots(figsize=(max(9, len(names)*1.1), 7))
    tick_colors = [CAT_COLORS.get(combined[n]["family"], "#666") for n in names]
    b1 = ax.bar(x - width/2, enc_vals, width, label="Encode", color=BAR_COLORS["encode"], edgecolor="white")
    b2 = ax.bar(x + width/2, dec_vals, width, label="Decode", color=BAR_COLORS["decode"], edgecolor="white")
    for bar, val in zip(b1, enc_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f"{val:.1f}" if val<100 else f"{val:.0f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    for bar, val in zip(b2, dec_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f"{val:.1f}" if val<100 else f"{val:.0f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.set_ylabel("Throughput (MB/s)", fontsize=14)
    ax.set_title(f"{title}  —  Encode / Decode Throughput", fontsize=17, fontweight="bold")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11, rotation=20, ha="right")
    for tick, color in zip(ax.get_xticklabels(), tick_colors):
        tick.set_color(color)
    ax.legend(fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "encode_decode_throughput.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {output_dir / 'encode_decode_throughput.png'}")

def main():
    for ds_name, cfg in DATASETS.items():
        rows = load_all(cfg["sources"])
        if not rows:
            print(f"{ds_name}: SKIP (no data)")
            continue
        print(f"\n{ds_name}: {len(rows)} rows")
        groups = group_into_curves(rows)
        for (label, family), pts in sorted(groups.items()):
            print(f"  [{family}] {label}: {len(pts)}pts, BPP={pts[0]['bpp']:.4f}~{pts[-1]['bpp']:.4f}, PSNR={pts[0]['psnr']:.1f}~{pts[-1]['psnr']:.1f}dB")
        out = PROJECT / cfg["output"]
        plot_psnr_vs_bpp(groups, out, cfg["title"])
        plot_throughput_bar(groups, out, cfg["title"])
    print("\nDone.")

if __name__ == "__main__":
    main()
