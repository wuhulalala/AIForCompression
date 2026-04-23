"""
Generate unified results table and cross-model visualizations for all compression models.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
VIS_DIR = PROJECT_ROOT / 'unified_results'
VIS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    'WeConvene': PROJECT_ROOT / 'WeConvene' / 'results_era5',
    'DCAE':      PROJECT_ROOT / 'DCAE' / 'results_era5',
    'LIC-TCM':   PROJECT_ROOT / 'LIC_TCM' / 'results_era5',
}

COLORS = {'WeConvene': '#1f77b4', 'DCAE': '#ff7f0e', 'LIC-TCM': '#2ca02c'}
MARKERS = {'WeConvene': 'o', 'DCAE': 's', 'LIC-TCM': '^'}

TIMESTAMP = '2024-06-01T00:00:00'

VAR_INFO = {
    'z_500': ('Geopotential@500hPa', 'm²/s²'),
    't_850': ('Temperature@850hPa', 'K'),
    't2m':   ('2m Temperature', 'K'),
    'u10':   ('10m U-wind', 'm/s'),
    'sp':    ('Surface Pressure', 'Pa'),
}

# ─── Load all results ───
all_results = {}
all_samples = {}
for model_name, results_dir in MODELS.items():
    with open(results_dir / 'summary.json') as f:
        summary = json.load(f)
    valid = [r for r in summary if 'error' not in r]
    valid.sort(key=lambda r: float(r['lmbda']) if r['lmbda'].replace('.','',1).isdigit() else 0, reverse=True)
    all_results[model_name] = valid

    samples_dir = results_dir / 'samples'
    orig_file = samples_dir / f'original_{TIMESTAMP}.npz'
    if orig_file.exists():
        all_samples[model_name] = {'original': dict(np.load(orig_file))}
        for r in valid:
            sf = samples_dir / f"{r['model_id']}_{TIMESTAMP}.npz"
            if sf.exists():
                all_samples[model_name][r['model_id']] = dict(np.load(sf))

# Use one original (they should all be the same)
orig = all_samples[list(all_samples.keys())[0]]['original']

# ─── 1. Unified Summary Table (markdown) ───
md_lines = []
md_lines.append('# Unified ERA5 Compression Results')
md_lines.append('')
md_lines.append('- Data: ERA5 2024-06-01T00:00:00 (268 channels, 721×1440)')
md_lines.append('')
md_lines.append('## Summary Table')
md_lines.append('')
md_lines.append('| Model | Quality (λ) | PSNR (dB) | 压缩比 | BPP | 推理时间 (s) | MSE | Params |')
md_lines.append('|-------|-------------|-----------|--------|-----|-------------|-----|--------|')

for model_name, results in all_results.items():
    for r in results:
        f = r['forward']
        params = f"{r['params']/1e6:.1f}M"
        md_lines.append(
            f"| {model_name} | {r['lmbda']} | {f['psnr']:.2f} | {f['compression_ratio']:.1f}x | "
            f"{f['bpp']:.4f} | {f['time']:.1f} | {f['mse']:.1f} | {params} |"
        )

# Per-variable RMSE tables
p_vars = ['z', 'q', 'u', 'v', 't', 'r', 'w']
s_vars = ['v10', 'u10', 'v100', 'u100', 't2m', 'tcc', 'sp', 'tp', 'msl']

md_lines.append('')
md_lines.append('## Per-Variable RMSE')
md_lines.append('')
md_lines.append('### Pressure-Level Variables (mean RMSE across 37 levels)')
md_lines.append('')
md_lines.append('| Model | λ | ' + ' | '.join(p_vars) + ' |')
md_lines.append('|---|---|' + '|'.join(['---']*len(p_vars)) + '|')
for model_name, results in all_results.items():
    for r in results:
        g = r.get('group_rmse', {})
        vals = []
        for v in p_vars:
            if v in g and 'mean_rmse' in g[v]:
                vals.append(f"{g[v]['mean_rmse']:.4f}")
            else:
                vals.append('N/A')
        md_lines.append(f"| {model_name} | {r['lmbda']} | " + ' | '.join(vals) + ' |')

md_lines.append('')
md_lines.append('### Single-Level Variables (RMSE)')
md_lines.append('')
md_lines.append('| Model | λ | ' + ' | '.join(s_vars) + ' |')
md_lines.append('|---|---|' + '|'.join(['---']*len(s_vars)) + '|')
for model_name, results in all_results.items():
    for r in results:
        g = r.get('group_rmse', {})
        vals = []
        for v in s_vars:
            if v in g and 'rmse' in g[v]:
                vals.append(f"{g[v]['rmse']:.4f}")
            else:
                vals.append('N/A')
        md_lines.append(f"| {model_name} | {r['lmbda']} | " + ' | '.join(vals) + ' |')

md_text = '\n'.join(md_lines) + '\n'
with open(VIS_DIR / 'results.md', 'w') as f:
    f.write(md_text)
print("Saved results.md")

# ─── 2. RD Curve: all models on one plot ───
fig, ax = plt.subplots(figsize=(10, 7))
for model_name, results in all_results.items():
    bpps = [r['forward']['bpp'] for r in results]
    psnrs = [r['forward']['psnr'] for r in results]
    ax.plot(bpps, psnrs, marker=MARKERS[model_name], color=COLORS[model_name],
            linewidth=2, markersize=8, label=model_name)
    for i, r in enumerate(results):
        ax.annotate(f"λ={r['lmbda']}", (bpps[i], psnrs[i]),
                    textcoords="offset points", xytext=(6, 6), fontsize=7, alpha=0.7)
ax.set_xlabel('BPP (bits per pixel)', fontsize=13)
ax.set_ylabel('PSNR (dB)', fontsize=13)
ax.set_title('Rate-Distortion Comparison on ERA5 (268 channels)', fontsize=15, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(VIS_DIR / 'rd_curve_all.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved rd_curve_all.png")

# ─── 3. Compression Ratio comparison (bar chart) ───
fig, ax = plt.subplots(figsize=(14, 6))
# Group by approximate lambda
lambda_groups = {}
for model_name, results in all_results.items():
    for r in results:
        lmbda = r['lmbda']
        key = lmbda
        if key not in lambda_groups:
            lambda_groups[key] = {}
        lambda_groups[key][model_name] = r

# Sort lambdas numerically
sorted_lambdas = sorted(lambda_groups.keys(),
                        key=lambda x: float(x) if x.replace('.','',1).isdigit() else 0,
                        reverse=True)
x = np.arange(len(sorted_lambdas))
model_names = list(MODELS.keys())
width = 0.25

for j, model_name in enumerate(model_names):
    crs = []
    for lmbda in sorted_lambdas:
        if model_name in lambda_groups.get(lmbda, {}):
            crs.append(lambda_groups[lmbda][model_name]['forward']['compression_ratio'])
        else:
            crs.append(0)
    ax.bar(x + j * width, crs, width, label=model_name, color=COLORS[model_name], alpha=0.8)

ax.set_xlabel('Quality (λ)', fontsize=12)
ax.set_ylabel('Compression Ratio', fontsize=12)
ax.set_title('Compression Ratio Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels([f'λ={l}' for l in sorted_lambdas], rotation=20, ha='right')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(VIS_DIR / 'compression_ratio_all.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved compression_ratio_all.png")

# ─── 4. Per-variable RMSE comparison for matched lambdas ───
common_lambdas = ['0.05', '0.025', '0.013', '0.0067', '0.0035', '0.0025']

for var_key, (var_name, var_unit) in VAR_INFO.items():
    fig, ax = plt.subplots(figsize=(10, 5))
    for model_name in model_names:
        rmses = []
        valid_lambdas = []
        for lmbda in common_lambdas:
            for r in all_results.get(model_name, []):
                if r['lmbda'] == lmbda:
                    o = orig[var_key]
                    mid = r['model_id']
                    if mid in all_samples.get(model_name, {}):
                        rec = all_samples[model_name][mid][var_key]
                        rmses.append(np.sqrt(np.mean((o - rec) ** 2)))
                        valid_lambdas.append(lmbda)
                    break
        if rmses:
            ax.plot(range(len(valid_lambdas)), rmses, marker=MARKERS[model_name],
                    color=COLORS[model_name], linewidth=2, markersize=7, label=model_name)
            ax.set_xticks(range(len(valid_lambdas)))
            ax.set_xticklabels([f'λ={l}' for l in valid_lambdas], rotation=15, ha='right')

    ax.set_xlabel('Quality (λ)', fontsize=11)
    ax.set_ylabel(f'RMSE ({var_unit})', fontsize=11)
    ax.set_title(f'{var_name} RMSE Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(VIS_DIR / f'rmse_compare_{var_key}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved rmse_compare_{var_key}.png")

# ─── 5. Visual comparison: best quality (lambda=0.05) across models ───
target_lambda = '0.05'
fig = plt.figure(figsize=(6 * (len(model_names) + 1), 5 * len(VAR_INFO)))
gs = fig.add_gridspec(len(VAR_INFO), len(model_names) + 2, hspace=0.35, wspace=0.3)

for row, (var_key, (var_name, var_unit)) in enumerate(VAR_INFO.items()):
    orig_data = orig[var_key]
    vmin, vmax = np.percentile(orig_data, [1, 99])

    # Original
    ax = fig.add_subplot(gs[row, 0])
    im = ax.imshow(orig_data, cmap='RdYlBu_r', vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_title('Original', fontsize=10, fontweight='bold')
    if row == 0:
        ax.set_title('Original', fontsize=11, fontweight='bold')
    ax.set_ylabel(var_name, fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.tick_params(labelsize=7)

    for col, model_name in enumerate(model_names):
        # Find the lambda=0.05 result
        mid = None
        for r in all_results[model_name]:
            if r['lmbda'] == target_lambda:
                mid = r['model_id']
                psnr = r['forward']['psnr']
                bpp = r['forward']['bpp']
                break
        ax = fig.add_subplot(gs[row, col + 1])
        if mid and mid in all_samples.get(model_name, {}):
            rec = all_samples[model_name][mid][var_key]
            im = ax.imshow(rec, cmap='RdYlBu_r', vmin=vmin, vmax=vmax, aspect='auto')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            rmse = np.sqrt(np.mean((orig_data - rec) ** 2))
            ax.set_title(f'{model_name}\nRMSE={rmse:.2f}', fontsize=9)
        else:
            ax.set_title(f'{model_name}\nN/A', fontsize=9)
            ax.axis('off')
        ax.tick_params(labelsize=7)

    # Error map for all models combined (worst model's error)
    ax = fig.add_subplot(gs[row, len(model_names) + 1])
    errors = {}
    for model_name in model_names:
        for r in all_results[model_name]:
            if r['lmbda'] == target_lambda:
                mid = r['model_id']
                if mid in all_samples.get(model_name, {}):
                    rec = all_samples[model_name][mid][var_key]
                    errors[model_name] = np.abs(orig_data - rec)
                break
    if errors:
        # Show the max error across models
        max_err_name = max(errors, key=lambda m: errors[m].mean())
        err = errors[max_err_name]
        im = ax.imshow(err, cmap='hot', vmin=0, vmax=np.percentile(err, 99), aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f'|Error| ({max_err_name})', fontsize=9)
    ax.tick_params(labelsize=7)

fig.suptitle(f'Visual Comparison at λ={target_lambda} (Best Quality)', fontsize=16, fontweight='bold', y=1.0)
fig.savefig(VIS_DIR / 'visual_compare_best.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved visual_compare_best.png")

# ─── 6. Inference time comparison ───
fig, ax = plt.subplots(figsize=(10, 5))
for model_name in model_names:
    times = []
    lambdas = []
    for r in all_results[model_name]:
        lmbda = r['lmbda']
        if lmbda in common_lambdas:
            times.append(r['forward']['time'])
            lambdas.append(lmbda)
    if times:
        ax.plot(range(len(lambdas)), times, marker=MARKERS[model_name],
                color=COLORS[model_name], linewidth=2, markersize=7, label=model_name)
        ax.set_xticks(range(len(lambdas)))
        ax.set_xticklabels([f'λ={l}' for l in lambdas], rotation=15, ha='right')
ax.set_xlabel('Quality (λ)', fontsize=12)
ax.set_ylabel('Inference Time (s)', fontsize=12)
ax.set_title('Inference Time Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(VIS_DIR / 'inference_time_all.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved inference_time_all.png")

# ─── 7. Error distribution comparison at lambda=0.05 ───
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for i, (var_key, (var_name, var_unit)) in enumerate(VAR_INFO.items()):
    ax = axes[i]
    o = orig[var_key].flatten()
    for model_name in model_names:
        for r in all_results[model_name]:
            if r['lmbda'] == target_lambda:
                mid = r['model_id']
                if mid in all_samples.get(model_name, {}):
                    rec = all_samples[model_name][mid][var_key].flatten()
                    err = o - rec
                    ax.hist(err, bins=100, alpha=0.4, density=True, label=model_name,
                            color=COLORS[model_name])
                break
    ax.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel(f'Error ({var_unit})', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(f'{var_name}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
axes[-1].axis('off')
fig.suptitle(f'Error Distribution Comparison at λ={target_lambda}', fontsize=15, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(VIS_DIR / 'error_dist_compare.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved error_dist_compare.png")

print(f"\nAll unified results saved to: {VIS_DIR}")
print(md_text)
