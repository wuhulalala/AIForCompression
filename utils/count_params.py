"""Count parameters from checkpoint files and plot bar chart."""
import os, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CKPT_BASE = '/data/run01/scxj523/zsh/project/AIForCompression/checkpoints'

def count_params_from_ckpt(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if 'state_dict' in ckpt:
        sd = ckpt['state_dict']
    elif 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
    else:
        sd = ckpt
    total = sum(v.numel() for v in sd.values())
    return total

# Collect: model_family -> list of param counts (one per quality)
results = {}

# CompressAI models: checkpoints/{arch}/{metric}/*.pth.tar
compressai_archs = ['bmshj2018-factorized', 'bmshj2018-hyperprior',
                    'mbt2018-mean', 'mbt2018', 'cheng2020-anchor', 'cheng2020-attn']
for arch in compressai_archs:
    arch_dir = os.path.join(CKPT_BASE, arch, 'mse')  # use mse metric
    if not os.path.isdir(arch_dir):
        continue
    counts = []
    for f in sorted(os.listdir(arch_dir)):
        if f.endswith('.pth.tar'):
            n = count_params_from_ckpt(os.path.join(arch_dir, f))
            counts.append(n)
    if counts:
        results[arch] = counts

# CRA5
cra5_path = os.path.join(CKPT_BASE, 'vaeformer-pretrained/mse/cra5_268v_300k.pth')
if os.path.exists(cra5_path):
    results['CRA5'] = [count_params_from_ckpt(cra5_path)]

# DCAE
dcae_dir = os.path.join(CKPT_BASE, 'dcae')
if os.path.isdir(dcae_dir):
    counts = []
    for f in sorted(os.listdir(dcae_dir)):
        if f.endswith('.pth.tar'):
            counts.append(count_params_from_ckpt(os.path.join(dcae_dir, f)))
    if counts:
        results['DCAE'] = counts

# LIC-TCM
lictcm_dir = os.path.join(CKPT_BASE, 'lictcm')
if os.path.isdir(lictcm_dir):
    counts = []
    for f in sorted(os.listdir(lictcm_dir)):
        if f.endswith('.pth.tar'):
            counts.append(count_params_from_ckpt(os.path.join(lictcm_dir, f)))
    if counts:
        results['LIC-TCM'] = counts

# WeConvene
wc_dir = os.path.join(CKPT_BASE, 'weconvene')
if os.path.isdir(wc_dir):
    counts = []
    for f in sorted(os.listdir(wc_dir)):
        if f.endswith('.pth.tar'):
            counts.append(count_params_from_ckpt(os.path.join(wc_dir, f)))
    if counts:
        results['WeConvene'] = counts

# Print
print(f"{'Model':<30} {'Avg Params':>15} {'Min':>15} {'Max':>15}")
print("=" * 78)
for name, counts in results.items():
    avg = np.mean(counts)
    print(f"{name:<30} {avg/1e6:>12.2f}M {min(counts)/1e6:>12.2f}M {max(counts)/1e6:>12.2f}M")

# Plot
model_names = list(results.keys())
param_avg = np.array([np.mean(v) / 1e6 for v in results.values()])
param_min = np.array([min(v) / 1e6 for v in results.values()])
param_max = np.array([max(v) / 1e6 for v in results.values()])

idx = np.argsort(param_avg)
model_names = [model_names[i] for i in idx]
param_avg = param_avg[idx]
param_min = param_min[idx]
param_max = param_max[idx]
yerr_low = param_avg - param_min
yerr_high = param_max - param_avg

colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))

fig, ax = plt.subplots(figsize=(14, 7))
bars = ax.bar(range(len(model_names)), param_avg, color=colors,
              yerr=[yerr_low, yerr_high], capsize=5, edgecolor='black', linewidth=0.5)

for i, (bar, val) in enumerate(zip(bars, param_avg)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + yerr_high[i] + 0.5,
            f'{val:.1f}M', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_xticks(range(len(model_names)))
ax.set_xticklabels(model_names, rotation=30, ha='right', fontsize=12)
ax.set_ylabel('Parameters (M)', fontsize=14)
ax.set_title('Model Size Comparison (Parameter Count)', fontsize=16, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()

out_dir = '/data/run01/scxj523/zsh/project/AIForCompression/unified_results'
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'model_params_bar.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out_path}")
