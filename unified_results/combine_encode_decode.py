"""
Combine all model results with encode/decode times separated.
Same format as combine_all_results.py but with encode_time and decode_time columns.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path('/data/run01/scxj523/zsh/project/AIForCompression/unified_results')
OUT_DIR.mkdir(exist_ok=True)

# ── 1. Load compressAI results ──
with open('/data/run01/scxj523/zsh/project/AIForCompression/CRA5/results/all_compressai/summary.json') as f:
    compressai_data = json.load(f)

mse_entries = [d for d in compressai_data if d['metric'] == 'mse' and 'compress' in d and 'error' not in d.get('compress', {})]

# ── 2. Load CRA5 native model ──
with open('/data/run01/scxj523/zsh/project/AIForCompression/CRA5/summary.json') as f:
    cra5 = json.load(f)

# ── 3. Load DCAE, LIC-TCM, WeConvene ──
with open('/data/run01/scxj523/zsh/project/AIForCompression/DCAE/results_era5/summary.json') as f:
    dcae_data = json.load(f)
with open('/data/run01/scxj523/zsh/project/AIForCompression/LIC_TCM/results_era5/summary.json') as f:
    lictcm_data = json.load(f)
with open('/data/run01/scxj523/zsh/project/AIForCompression/WeConvene/results_era5/summary.json') as f:
    weconvene_data = json.load(f)

COMPRESSAI_LAMBDA = {
    1: 0.0018, 2: 0.0035, 3: 0.0067, 4: 0.0130,
    5: 0.0250, 6: 0.0483, 7: 0.0932, 8: 0.1800,
}

# ── 4. Build unified rows ──
rows = []

# CRA5 native
rows.append({
    'model': 'CRA5',
    'quality': 'λ=native',
    'quality_raw': 'native',
    'lambda': 0.0,
    'psnr': cra5['psnr'],
    'cr': cra5['compression_ratio'],
    'bpp': cra5['bpp'],
    'encode_time': cra5['avg_encode_time'],
    'decode_time': cra5['avg_decode_time'],
    'mse': cra5['mse'],
    'params': '404.7M',
})

# CompressAI models
for d in mse_entries:
    q = d['quality']
    lmbda = COMPRESSAI_LAMBDA.get(q, q)
    c = d['compress']
    rows.append({
        'model': d['arch'],
        'quality': f"q{q} (λ={lmbda})",
        'quality_raw': q,
        'lambda': lmbda,
        'psnr': c['psnr'],
        'cr': c['compression_ratio'],
        'bpp': c['bpp'],
        'encode_time': c['encode_time'],
        'decode_time': c['decode_time'],
        'mse': c['mse'],
        'params': d.get('params', 0),
    })

# DCAE
for d in dcae_data:
    if 'error' in d:
        continue
    c = d['compress']
    rows.append({
        'model': 'DCAE',
        'quality': f"λ={d['lmbda']}",
        'quality_raw': d['lmbda'],
        'lambda': float(d['lmbda']),
        'psnr': c['psnr'],
        'cr': c['compression_ratio'],
        'bpp': c['bpp'],
        'encode_time': c['encode_time'],
        'decode_time': c['decode_time'],
        'mse': c['mse'],
        'params': d.get('params', 0),
    })

# LIC-TCM
for d in lictcm_data:
    if 'error' in d:
        continue
    c = d['compress']
    lmbda_str = d['lmbda']
    lmbda_val = float(lmbda_str.replace('_large', '')) if '_large' not in lmbda_str else 0.05
    rows.append({
        'model': 'LIC-TCM',
        'quality': f"λ={lmbda_str}",
        'quality_raw': lmbda_str,
        'lambda': lmbda_val,
        'psnr': c['psnr'],
        'cr': c['compression_ratio'],
        'bpp': c['bpp'],
        'encode_time': c['encode_time'],
        'decode_time': c['decode_time'],
        'mse': c['mse'],
        'params': d.get('params', 0),
    })

# WeConvene
for d in weconvene_data:
    if 'error' in d:
        continue
    c = d['compress']
    rows.append({
        'model': 'WeConvene',
        'quality': f"λ={d['lmbda']}",
        'quality_raw': d['lmbda'],
        'lambda': float(d['lmbda']),
        'psnr': c['psnr'],
        'cr': c['compression_ratio'],
        'bpp': c['bpp'],
        'encode_time': c['encode_time'],
        'decode_time': c['decode_time'],
        'mse': c['mse'],
        'params': d.get('params', 0),
    })

# ── 5. Write markdown ──
def fmt_params(p):
    if isinstance(p, str):
        return p
    if p >= 1e6:
        return f"{p/1e6:.1f}M"
    return str(p)

md_lines = [
    "# Unified ERA5 Compression Results — Encode/Decode (All Models, MSE Only)",
    "",
    "- Data: ERA5 2024-06-01T00:00:00 (268 channels, 721×1440)",
    "",
    "## Summary Table",
    "",
    "| Model | Quality (λ) | PSNR (dB) | 压缩比 | BPP | 编码时间 (s) | 解码时间 (s) | 总时间 (s) | MSE | Params |",
    "|-------|-------------|-----------|--------|-----|-------------|-------------|-----------|-----|--------|",
]

rows_sorted = sorted(rows, key=lambda r: (r['model'], -r['lambda']))
for r in rows_sorted:
    total = r['encode_time'] + r['decode_time']
    md_lines.append(
        f"| {r['model']} | {r['quality']} | {r['psnr']:.2f} | {r['cr']:.1f}x | {r['bpp']:.4f} "
        f"| {r['encode_time']:.1f} | {r['decode_time']:.1f} | {total:.1f} | {r['mse']:.1f} | {fmt_params(r['params'])} |"
    )

md_text = "\n".join(md_lines) + "\n"
with open(OUT_DIR / 'results_encode_decode.md', 'w') as f:
    f.write(md_text)
print(f"Wrote {OUT_DIR / 'results_encode_decode.md'}")

# ── 6. Plotting ──
plt.rcParams.update({
    'font.size': 12,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
})

all_models = sorted(set(r['model'] for r in rows))
cmap = plt.cm.get_cmap('tab10', len(all_models))
model_colors = {m: cmap(i) for i, m in enumerate(all_models)}

bpp_bins = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 45), (45, 60), (60, 80)]
bin_labels = ['<5', '5-10', '10-15', '15-20', '20-30', '30-45', '45-60', '60-80']

# ── Plot 1: PSNR Bar Chart ──
fig, ax = plt.subplots(figsize=(18, 8))
bar_data = {}
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
plt.savefig(OUT_DIR / 'psnr_bar_enc_dec.png', dpi=150)
plt.close()
print("Saved psnr_bar_enc_dec.png")

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
plt.savefig(OUT_DIR / 'compression_ratio_bar_enc_dec.png', dpi=150)
plt.close()
print("Saved compression_ratio_bar_enc_dec.png")

# ── Plot 3: Encode/Decode Time Bar Chart (stacked) ──
fig, ax = plt.subplots(figsize=(14, 7))

model_enc = {}
model_dec = {}
for r in rows:
    m = r['model']
    if m not in model_enc:
        model_enc[m] = []
        model_dec[m] = []
    model_enc[m].append(r['encode_time'])
    model_dec[m].append(r['decode_time'])

models_sorted = sorted(model_enc.keys())
avg_enc = [np.mean(model_enc[m]) for m in models_sorted]
avg_dec = [np.mean(model_dec[m]) for m in models_sorted]

x_pos = np.arange(len(models_sorted))
bars_enc = ax.bar(x_pos, avg_enc, 0.6, label='Encode', color='#4C72B0')
bars_dec = ax.bar(x_pos, avg_dec, 0.6, bottom=avg_enc, label='Decode', color='#DD8452')

for i, (e, d) in enumerate(zip(avg_enc, avg_dec)):
    total = e + d
    ax.text(x_pos[i], total + 0.3, f'{total:.1f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.text(x_pos[i], e / 2, f'{e:.1f}', ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    ax.text(x_pos[i], e + d / 2, f'{d:.1f}', ha='center', va='center', fontsize=8, color='white', fontweight='bold')

ax.set_xlabel('Model', fontsize=14)
ax.set_ylabel('Time (s)', fontsize=14)
ax.set_title('Average Encode/Decode Time per Model (MSE Only)', fontsize=16)
ax.set_xticks(x_pos)
ax.set_xticklabels(models_sorted, rotation=30, ha='right', fontsize=10)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
plt.savefig(OUT_DIR / 'encode_decode_time_bar.png', dpi=150)
plt.close()
print("Saved encode_decode_time_bar.png")

# ── Plot 4: Rate-Distortion Curve ──
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
plt.savefig(OUT_DIR / 'rate_distortion_enc_dec.png', dpi=150)
plt.close()
print("Saved rate_distortion_enc_dec.png")

# ── Plot 5: Model Params Bar Chart ──
fig, ax = plt.subplots(figsize=(14, 7))

model_params = {}
for r in rows:
    m = r['model']
    p = r['params']
    if isinstance(p, str):
        p = float(p.replace('M', '')) * 1e6
    model_params[m] = p

models_sorted_p = sorted(model_params.keys())
params_vals = [model_params[m] / 1e6 for m in models_sorted_p]

x_pos_p = np.arange(len(models_sorted_p))
bars = ax.bar(x_pos_p, params_vals, 0.6, color=[model_colors[m] for m in models_sorted_p])

for bar, h in zip(bars, params_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{h:.1f}M', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('Model', fontsize=14)
ax.set_ylabel('Parameters (M)', fontsize=14)
ax.set_title('Model Parameters Comparison', fontsize=16)
ax.set_xticks(x_pos_p)
ax.set_xticklabels(models_sorted_p, rotation=30, ha='right', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.savefig(OUT_DIR / 'model_params_bar_enc_dec.png', dpi=150)
plt.close()
print("Saved model_params_bar_enc_dec.png")

print("\nDone! All outputs in:", OUT_DIR)
