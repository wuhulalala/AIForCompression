"""
Visualize WeConvene ERA5 reconstruction results.
Generates comprehensive comparison plots:
1. Original vs Reconstructed side-by-side for each variable & quality level
2. Error maps (absolute difference)
3. Value distribution histograms (original vs reconstructed overlay)
4. MSE per variable across quality levels
5. Summary RD curve (PSNR vs BPP)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / 'results_era5'
SAMPLES_DIR = RESULTS_DIR / 'samples'
VIS_DIR = RESULTS_DIR / 'visualizations'
VIS_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = '2024-06-01T00:00:00'

# Variable display names and units
VAR_INFO = {
    'z_500': ('Geopotential @ 500hPa', 'm²/s²'),
    't_850': ('Temperature @ 850hPa', 'K'),
    't2m':   ('2m Temperature', 'K'),
    'u10':   ('10m U-wind', 'm/s'),
    'sp':    ('Surface Pressure', 'Pa'),
}

# Load summary
with open(RESULTS_DIR / 'summary.json') as f:
    summary = json.load(f)

# Sort by lambda (high to low quality)
valid_results = [r for r in summary if 'error' not in r]
valid_results.sort(key=lambda r: float(r['lmbda']), reverse=True)

# Load original
orig = dict(np.load(SAMPLES_DIR / f'original_{TIMESTAMP}.npz'))

# Load all reconstructions
recons = {}
for r in valid_results:
    mid = r['model_id']
    fpath = SAMPLES_DIR / f"{mid}_{TIMESTAMP}.npz"
    if fpath.exists():
        recons[mid] = dict(np.load(fpath))

model_ids = [r['model_id'] for r in valid_results if r['model_id'] in recons]
n_models = len(model_ids)

print(f"Loaded {n_models} models, variables: {list(VAR_INFO.keys())}")

# ─── 1. Per-variable: Original vs Reconstructed vs Error (one big figure per variable) ───
for var_key, (var_name, var_unit) in VAR_INFO.items():
    fig, axes = plt.subplots(3, n_models + 1, figsize=(5 * (n_models + 1), 14))
    fig.suptitle(f'{var_name} ({var_key})', fontsize=18, fontweight='bold', y=0.98)

    orig_data = orig[var_key]
    vmin, vmax = np.percentile(orig_data, [1, 99])

    # Original column
    im = axes[0, 0].imshow(orig_data, cmap='RdYlBu_r', vmin=vmin, vmax=vmax, aspect='auto')
    axes[0, 0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Spatial Map', fontsize=11)
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # Histogram for original
    axes[1, 0].hist(orig_data.flatten(), bins=100, alpha=0.7, color='steelblue', density=True)
    axes[1, 0].set_title('Value Distribution', fontsize=11)
    axes[1, 0].set_ylabel('Density', fontsize=11)
    axes[1, 0].set_xlabel(f'{var_unit}', fontsize=10)

    # Empty error slot for original
    axes[2, 0].axis('off')
    axes[2, 0].text(0.5, 0.5, 'N/A\n(Reference)', ha='center', va='center',
                    fontsize=14, transform=axes[2, 0].transAxes)

    for j, mid in enumerate(model_ids):
        col = j + 1
        recon_data = recons[mid][var_key]
        lmbda = [r['lmbda'] for r in valid_results if r['model_id'] == mid][0]
        psnr = [r['forward']['psnr'] for r in valid_results if r['model_id'] == mid][0]
        bpp = [r['forward']['bpp'] for r in valid_results if r['model_id'] == mid][0]

        # Reconstructed map
        im = axes[0, col].imshow(recon_data, cmap='RdYlBu_r', vmin=vmin, vmax=vmax, aspect='auto')
        axes[0, col].set_title(f'λ={lmbda}\nBPP={bpp:.2f}', fontsize=11)
        plt.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)

        # Histogram overlay
        axes[1, col].hist(orig_data.flatten(), bins=100, alpha=0.5, color='steelblue',
                          density=True, label='Original')
        axes[1, col].hist(recon_data.flatten(), bins=100, alpha=0.5, color='coral',
                          density=True, label='Recon')
        axes[1, col].legend(fontsize=8)
        axes[1, col].set_xlabel(f'{var_unit}', fontsize=10)

        # Error map
        error = np.abs(orig_data - recon_data)
        err_vmax = np.percentile(error, 99)
        im_err = axes[2, col].imshow(error, cmap='hot', vmin=0, vmax=err_vmax, aspect='auto')
        rmse = np.sqrt(np.mean((orig_data - recon_data) ** 2))
        mae = np.mean(error)
        axes[2, col].set_title(f'|Error| RMSE={rmse:.2f}\nMAE={mae:.2f}', fontsize=10)
        plt.colorbar(im_err, ax=axes[2, col], fraction=0.046, pad=0.04)
        if j == 0:
            axes[2, 0].set_ylabel('Absolute Error', fontsize=11)

    for ax_row in axes:
        for ax in ax_row:
            ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    outpath = VIS_DIR / f'compare_{var_key}.png'
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {outpath}")


# ─── 2. RMSE per variable across quality levels (bar chart) ───
fig, ax = plt.subplots(figsize=(12, 6))
var_keys = list(VAR_INFO.keys())
x = np.arange(len(var_keys))
width = 0.8 / n_models

for j, mid in enumerate(model_ids):
    rmses = []
    for vk in var_keys:
        diff = orig[vk] - recons[mid][vk]
        rmses.append(np.sqrt(np.mean(diff ** 2)))
    lmbda = [r['lmbda'] for r in valid_results if r['model_id'] == mid][0]
    ax.bar(x + j * width, rmses, width, label=f'λ={lmbda}')

ax.set_xlabel('Variable', fontsize=12)
ax.set_ylabel('RMSE', fontsize=12)
ax.set_title('RMSE per Variable across Quality Levels', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * n_models / 2 - width / 2)
ax.set_xticklabels([VAR_INFO[k][0] for k in var_keys], rotation=15, ha='right')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(VIS_DIR / 'rmse_per_variable.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved rmse_per_variable.png")


# ─── 3. RD Curve: PSNR vs BPP ───
fig, ax = plt.subplots(figsize=(8, 6))
bpps = [r['forward']['bpp'] for r in valid_results]
psnrs = [r['forward']['psnr'] for r in valid_results]
lambdas = [r['lmbda'] for r in valid_results]

ax.plot(bpps, psnrs, 'o-', color='steelblue', linewidth=2, markersize=8)
for i, lmbda in enumerate(lambdas):
    ax.annotate(f'λ={lmbda}', (bpps[i], psnrs[i]), textcoords="offset points",
                xytext=(8, 8), fontsize=9)

ax.set_xlabel('BPP (bits per pixel)', fontsize=12)
ax.set_ylabel('PSNR (dB)', fontsize=12)
ax.set_title('WeConvene Rate-Distortion Curve on ERA5', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(VIS_DIR / 'rd_curve.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved rd_curve.png")


# ─── 4. Compression ratio vs PSNR ───
fig, ax1 = plt.subplots(figsize=(8, 6))
crs = [r['forward']['compression_ratio'] for r in valid_results]
ax1.bar(range(n_models), crs, color='steelblue', alpha=0.7)
ax1.set_ylabel('Compression Ratio', fontsize=12, color='steelblue')
ax1.set_xticks(range(n_models))
ax1.set_xticklabels([f'λ={r["lmbda"]}' for r in valid_results], rotation=15, ha='right')

ax2 = ax1.twinx()
ax2.plot(range(n_models), psnrs, 'o-', color='coral', linewidth=2, markersize=8)
ax2.set_ylabel('PSNR (dB)', fontsize=12, color='coral')

ax1.set_title('Compression Ratio & PSNR across Quality Levels', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(VIS_DIR / 'cr_psnr.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved cr_psnr.png")


# ─── 5. Scatter plot: Original vs Reconstructed values ───
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()

for i, (var_key, (var_name, var_unit)) in enumerate(VAR_INFO.items()):
    ax = axes[i]
    o = orig[var_key].flatten()
    # Use best and worst quality
    best_mid = model_ids[0]   # highest lambda
    worst_mid = model_ids[-1]  # lowest lambda
    r_best = recons[best_mid][var_key].flatten()
    r_worst = recons[worst_mid][var_key].flatten()

    # Subsample for scatter
    idx = np.random.RandomState(42).choice(len(o), min(5000, len(o)), replace=False)

    ax.scatter(o[idx], r_worst[idx], alpha=0.2, s=3, c='coral',
               label=f'λ={valid_results[-1]["lmbda"]}')
    ax.scatter(o[idx], r_best[idx], alpha=0.2, s=3, c='steelblue',
               label=f'λ={valid_results[0]["lmbda"]}')
    lims = [min(o.min(), r_best.min()), max(o.max(), r_best.max())]
    ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel(f'Original ({var_unit})', fontsize=10)
    ax.set_ylabel(f'Reconstructed ({var_unit})', fontsize=10)
    ax.set_title(var_name, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, markerscale=5)
    ax.set_aspect('equal', adjustable='box')

axes[-1].axis('off')
fig.suptitle('Original vs Reconstructed Value Scatter', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(VIS_DIR / 'scatter_orig_vs_recon.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved scatter_orig_vs_recon.png")


# ─── 6. Error distribution (histogram of errors) per quality ───
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()

for i, (var_key, (var_name, var_unit)) in enumerate(VAR_INFO.items()):
    ax = axes[i]
    o = orig[var_key].flatten()
    for j, mid in enumerate(model_ids):
        lmbda = [r['lmbda'] for r in valid_results if r['model_id'] == mid][0]
        err = (o - recons[mid][var_key].flatten())
        ax.hist(err, bins=100, alpha=0.4, density=True, label=f'λ={lmbda}')
    ax.set_xlabel(f'Error ({var_unit})', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(f'{var_name} Error Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7)
    ax.axvline(0, color='k', linestyle='--', alpha=0.5)

axes[-1].axis('off')
fig.suptitle('Reconstruction Error Distribution', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(VIS_DIR / 'error_distribution.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved error_distribution.png")


# ─── 7. Summary table ───
print("\n" + "=" * 100)
print(f"{'Model':<35} {'λ':>8} {'PSNR(dB)':>10} {'BPP':>10} {'CR':>10} {'MSE':>14} {'Time(s)':>10}")
print("-" * 100)
for r in valid_results:
    fwd = r['forward']
    print(f"{r['model_id']:<35} {r['lmbda']:>8} {fwd['psnr']:>10.2f} {fwd['bpp']:>10.4f} "
          f"{fwd['compression_ratio']:>10.2f} {fwd['mse']:>14.4f} {fwd['time']:>10.1f}")
print("=" * 100)

print(f"\nAll visualizations saved to: {VIS_DIR}")
