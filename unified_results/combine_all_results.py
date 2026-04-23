"""
Combine compressAI pretrained model results (MSE only) with new model results
and generate unified summary + plots.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path('/data/run01/scxj523/zsh/project/AIForCompression/unified_results')
OUT_DIR.mkdir(exist_ok=True)

# ── 1. Load compressAI results (MSE only) ──
with open('/data/run01/scxj523/zsh/project/AIForCompression/CRA5/results/all_compressai/summary.json') as f:
    compressai_data = json.load(f)

mse_entries = [d for d in compressai_data if d['metric'] == 'mse']

# Filter out anomalous entries (cheng2020-anchor q4 has negative PSNR)
mse_entries = [d for d in mse_entries if d['forward']['psnr'] > 0]

# ── 2. Load new model results from results.md ──
new_models = [
    # WeConvene
    {"model": "WeConvene", "quality": "0.05",   "psnr": 81.52, "cr": 489.2,  "bpp": 17.5303, "time": 15.5, "mse": 1703.3,  "params": "107.1M"},
    {"model": "WeConvene", "quality": "0.025",  "psnr": 80.11, "cr": 681.9,  "bpp": 12.5758, "time": 11.7, "mse": 2356.1,  "params": "107.1M"},
    {"model": "WeConvene", "quality": "0.013",  "psnr": 78.98, "cr": 949.1,  "bpp": 9.0356,  "time": 11.7, "mse": 3057.0,  "params": "107.1M"},
    {"model": "WeConvene", "quality": "0.0067", "psnr": 76.83, "cr": 1274.0, "bpp": 6.7314,  "time": 11.7, "mse": 5012.6,  "params": "107.1M"},
    {"model": "WeConvene", "quality": "0.0035", "psnr": 75.82, "cr": 1866.4, "bpp": 4.5951,  "time": 11.7, "mse": 6321.8,  "params": "107.1M"},
    {"model": "WeConvene", "quality": "0.0025", "psnr": 74.95, "cr": 2149.1, "bpp": 3.9905,  "time": 11.7, "mse": 7724.3,  "params": "107.1M"},
    # DCAE
    {"model": "DCAE", "quality": "0.05",   "psnr": 81.39, "cr": 516.4,  "bpp": 16.6058, "time": 15.9, "mse": 1754.8,  "params": "119.4M"},
    {"model": "DCAE", "quality": "0.025",  "psnr": 80.59, "cr": 714.9,  "bpp": 11.9962, "time": 10.1, "mse": 2109.4,  "params": "119.4M"},
    {"model": "DCAE", "quality": "0.013",  "psnr": 79.61, "cr": 983.3,  "bpp": 8.7214,  "time": 10.1, "mse": 2642.4,  "params": "119.4M"},
    {"model": "DCAE", "quality": "0.0067", "psnr": 78.14, "cr": 1382.5, "bpp": 6.2031,  "time": 10.1, "mse": 3709.0,  "params": "119.4M"},
    {"model": "DCAE", "quality": "0.0035", "psnr": 76.08, "cr": 2092.2, "bpp": 4.0991,  "time": 10.1, "mse": 5957.4,  "params": "119.4M"},
    {"model": "DCAE", "quality": "0.0018", "psnr": 75.13, "cr": 2923.8, "bpp": 2.9332,  "time": 10.1, "mse": 7411.4,  "params": "119.4M"},
    # LIC-TCM
    {"model": "LIC-TCM", "quality": "0.05",       "psnr": 80.63, "cr": 466.1,  "bpp": 18.4000, "time": 14.7, "mse": 2088.2,  "params": "45.2M"},
    {"model": "LIC-TCM", "quality": "0.025",      "psnr": 78.55, "cr": 663.1,  "bpp": 12.9336, "time": 9.2,  "mse": 3370.0,  "params": "45.2M"},
    {"model": "LIC-TCM", "quality": "0.013",      "psnr": 77.19, "cr": 919.2,  "bpp": 9.3300,  "time": 9.2,  "mse": 4616.2,  "params": "45.2M"},
    {"model": "LIC-TCM", "quality": "0.0067",     "psnr": 76.85, "cr": 1313.9, "bpp": 6.5269,  "time": 9.2,  "mse": 4986.2,  "params": "45.2M"},
    {"model": "LIC-TCM", "quality": "0.0035",     "psnr": 75.24, "cr": 1842.8, "bpp": 4.6537,  "time": 9.3,  "mse": 7231.5,  "params": "45.2M"},
    {"model": "LIC-TCM", "quality": "0.0025",     "psnr": 74.51, "cr": 2210.1, "bpp": 3.8804,  "time": 9.2,  "mse": 8550.7,  "params": "45.2M"},
    {"model": "LIC-TCM", "quality": "0.05_large",  "psnr": 81.33, "cr": 479.4,  "bpp": 17.8892, "time": 16.2, "mse": 1778.2,  "params": "76.6M"},
    # CRA5 (268-channel native model, single forward pass)
    {"model": "CRA5", "quality": "native", "psnr": 83.82, "cr": 468.34, "bpp": 18.31, "time": 1.33, "mse": 1003.34, "params": "404.7M"},
]

# CompressAI standard lambda mapping for MSE
COMPRESSAI_LAMBDA = {
    1: 0.0018, 2: 0.0035, 3: 0.0067, 4: 0.0130,
    5: 0.0250, 6: 0.0483, 7: 0.0932, 8: 0.1800,
}

# ── 3. Build unified table rows ──
rows = []
for d in mse_entries:
    q = d['quality']
    lmbda = COMPRESSAI_LAMBDA.get(q, q)
    rows.append({
        'model': d['arch'],
        'quality': f"q{q} (λ={lmbda})",
        'quality_raw': q,
        'lambda': lmbda,
        'psnr': d['forward']['psnr'],
        'cr': d['forward']['compression_ratio'],
        'bpp': d['forward']['bpp'],
        'time': d['forward']['time'],
        'mse': d['forward']['mse'],
        'params': d.get('params', 0),
        'group_rmse': d.get('group_rmse', {}),
    })

for d in new_models:
    rows.append({
        'model': d['model'],
        'quality': f"λ={d['quality']}",
        'quality_raw': d['quality'],
        'lambda': 0.0 if d['quality'] == 'native' else (float(d['quality'].replace('_large', '')) if '_large' not in d['quality'] else 0.05),
        'psnr': d['psnr'],
        'cr': d['cr'],
        'bpp': d['bpp'],
        'time': d['time'],
        'mse': d['mse'],
        'params': d['params'],
        'group_rmse': {},
    })

# ── 4. Write unified results markdown ──
def fmt_params(p):
    if isinstance(p, str):
        return p
    if p >= 1e6:
        return f"{p/1e6:.1f}M"
    return str(p)

md_lines = [
    "# Unified ERA5 Compression Results (All Models, MSE Only)",
    "",
    "- Data: ERA5 2024-06-01T00:00:00 (268 channels, 721×1440)",
    "",
    "## Summary Table",
    "",
    "| Model | Quality (λ) | PSNR (dB) | 压缩比 | BPP | 推理时间 (s) | MSE | Params |",
    "|-------|-------------|-----------|--------|-----|-------------|-----|--------|",
]

# Sort: by model name, then by lambda descending (higher lambda = higher quality)
rows_sorted = sorted(rows, key=lambda r: (r['model'], -r['lambda']))
for r in rows_sorted:
    md_lines.append(
        f"| {r['model']} | {r['quality']} | {r['psnr']:.2f} | {r['cr']:.1f}x | {r['bpp']:.4f} | {r['time']:.1f} | {r['mse']:.1f} | {fmt_params(r['params'])} |"
    )

md_text = "\n".join(md_lines) + "\n"
with open(OUT_DIR / 'results_all.md', 'w') as f:
    f.write(md_text)
print(f"Wrote {OUT_DIR / 'results_all.md'}")

# ── 5. Plotting ──
plt.rcParams.update({
    'font.size': 12,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
})

# Get all unique models
all_models = sorted(set(r['model'] for r in rows))

# Color palette
cmap = plt.cm.get_cmap('tab10', len(all_models))
model_colors = {m: cmap(i) for i, m in enumerate(all_models)}

# ── Plot 1: PSNR Bar Chart (grouped by BPP range) ──
fig, ax = plt.subplots(figsize=(18, 8))

# For bar chart, pick representative BPP ranges to compare
# Use BPP bins to group models
bpp_bins = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 45), (45, 60), (60, 80)]
bin_labels = ['<5', '5-10', '10-15', '15-20', '20-30', '30-45', '45-60', '60-80']

bar_data = {}  # model -> {bin_label: best_psnr}
for r in rows:
    m = r['model']
    bpp = r['bpp']
    for (lo, hi), label in zip(bpp_bins, bin_labels):
        if lo <= bpp < hi:
            if m not in bar_data:
                bar_data[m] = {}
            if label not in bar_data[m] or r['psnr'] > bar_data[m][label]:
                bar_data[m][label] = r['psnr']
            break

x = np.arange(len(bin_labels))
n_models = len(bar_data)
width = 0.8 / max(n_models, 1)

for i, (model, data) in enumerate(sorted(bar_data.items())):
    vals = [data.get(bl, 0) for bl in bin_labels]
    mask = [v > 0 for v in vals]
    positions = x[mask] + (i - n_models/2 + 0.5) * width
    heights = [v for v, m in zip(vals, mask) if m]
    bars = ax.bar(positions, heights, width * 0.9, label=model, color=model_colors[model])
    for bar, h in zip(bars, heights):
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                    f'{h:.1f}', ha='center', va='bottom', fontsize=6, rotation=45)

ax.set_xlabel('BPP Range', fontsize=14)
ax.set_ylabel('PSNR (dB)', fontsize=14)
ax.set_title('PSNR Comparison across Models and BPP Ranges (MSE Only)', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(bin_labels)
ax.legend(fontsize=9, ncol=3, loc='upper right')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(bottom=55)
plt.savefig(OUT_DIR / 'psnr_bar_all.png', dpi=150)
plt.close()
print(f"Saved psnr_bar_all.png")

# ── Plot 2: Compression Ratio Bar Chart ──
fig, ax = plt.subplots(figsize=(18, 8))

cr_data = {}
for r in rows:
    m = r['model']
    bpp = r['bpp']
    for (lo, hi), label in zip(bpp_bins, bin_labels):
        if lo <= bpp < hi:
            if m not in cr_data:
                cr_data[m] = {}
            if label not in cr_data[m] or r['cr'] > cr_data[m][label]:
                cr_data[m][label] = r['cr']
            break

for i, (model, data) in enumerate(sorted(cr_data.items())):
    vals = [data.get(bl, 0) for bl in bin_labels]
    mask = [v > 0 for v in vals]
    positions = x[mask] + (i - n_models/2 + 0.5) * width
    heights = [v for v, m in zip(vals, mask) if m]
    bars = ax.bar(positions, heights, width * 0.9, label=model, color=model_colors[model])
    for bar, h in zip(bars, heights):
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                    f'{h:.0f}', ha='center', va='bottom', fontsize=6, rotation=45)

ax.set_xlabel('BPP Range', fontsize=14)
ax.set_ylabel('Compression Ratio', fontsize=14)
ax.set_title('Compression Ratio Comparison (MSE Only)', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(bin_labels)
ax.legend(fontsize=9, ncol=3, loc='upper right')
ax.grid(True, alpha=0.3, axis='y')
plt.savefig(OUT_DIR / 'compression_ratio_bar_all.png', dpi=150)
plt.close()
print(f"Saved compression_ratio_bar_all.png")

# ── Plot 3: Inference Time Histogram ──
fig, ax = plt.subplots(figsize=(14, 7))

# Group by model, show average inference time
model_times = {}
for r in rows:
    m = r['model']
    if m not in model_times:
        model_times[m] = []
    model_times[m].append(r['time'])

models_sorted = sorted(model_times.keys())
avg_times = [np.mean(model_times[m]) for m in models_sorted]
min_times = [np.min(model_times[m]) for m in models_sorted]
max_times = [np.max(model_times[m]) for m in models_sorted]

x_pos = np.arange(len(models_sorted))
bars = ax.bar(x_pos, avg_times, 0.6, color=[model_colors[m] for m in models_sorted])

# Error bars showing min-max range
yerr_low = [avg - mn for avg, mn in zip(avg_times, min_times)]
yerr_high = [mx - avg for avg, mx in zip(avg_times, max_times)]
ax.errorbar(x_pos, avg_times, yerr=[yerr_low, yerr_high], fmt='none', ecolor='black', capsize=5)

for bar, h in zip(bars, avg_times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'{h:.1f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('Model', fontsize=14)
ax.set_ylabel('Inference Time (s)', fontsize=14)
ax.set_title('Average Inference Time per Model (MSE Only)', fontsize=16)
ax.set_xticks(x_pos)
ax.set_xticklabels(models_sorted, rotation=30, ha='right', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.savefig(OUT_DIR / 'inference_time_bar_all.png', dpi=150)
plt.close()
print(f"Saved inference_time_bar_all.png")

# ── Plot 4: Rate-Distortion Curve (BPP vs PSNR) ──
fig, ax = plt.subplots(figsize=(14, 8))

for model in all_models:
    model_rows = sorted([r for r in rows if r['model'] == model], key=lambda r: r['bpp'])
    if not model_rows:
        continue
    bpps = [r['bpp'] for r in model_rows]
    psnrs = [r['psnr'] for r in model_rows]
    ax.plot(bpps, psnrs, '-o', color=model_colors[model], label=model, markersize=5, linewidth=2)

ax.set_xlabel('BPP (bits per pixel)', fontsize=14)
ax.set_ylabel('PSNR (dB)', fontsize=14)
ax.set_title('Rate-Distortion Curve on ERA5 (MSE Only)', fontsize=16)
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)
plt.savefig(OUT_DIR / 'rate_distortion_all.png', dpi=150)
plt.close()
print(f"Saved rate_distortion_all.png")

print("\nDone! All outputs in:", OUT_DIR)
