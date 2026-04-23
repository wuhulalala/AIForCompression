"""Generate encode/decode time bar chart."""
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

enc = defaultdict(list)
dec = defaultdict(list)

# DCAE
with open('DCAE/results_era5/summary.json') as f:
    for r in json.load(f):
        enc['DCAE'].append(r['compress']['encode_time'])
        dec['DCAE'].append(r['compress']['decode_time'])

# LIC-TCM
with open('LIC_TCM/results_era5/summary.json') as f:
    for r in json.load(f):
        enc['LIC-TCM'].append(r['compress']['encode_time'])
        dec['LIC-TCM'].append(r['compress']['decode_time'])

# WeConvene
with open('WeConvene/results_era5/summary.json') as f:
    for r in json.load(f):
        enc['WeConvene'].append(r['compress']['encode_time'])
        dec['WeConvene'].append(r['compress']['decode_time'])

# CRA5
with open('/data/run01/scxj523/zsh/project/cra5/summary.json') as f:
    r = json.load(f)
    enc['CRA5'] = r['encode_times']
    dec['CRA5'] = r['decode_times']

# CompressAI
with open('CRA5/results/all_compressai/summary.json') as f:
    for r in json.load(f):
        arch = r['arch']
        enc[arch].append(r['compress']['encode_time'])
        dec[arch].append(r['compress']['decode_time'])

# Sort by total time
names = sorted(enc.keys(), key=lambda x: np.mean(enc[x]) + np.mean(dec[x]))
enc_means = [np.mean(enc[n]) for n in names]
dec_means = [np.mean(dec[n]) for n in names]
enc_stds = [np.std(enc[n]) for n in names]
dec_stds = [np.std(dec[n]) for n in names]

x = np.arange(len(names))
width = 0.35

fig, ax = plt.subplots(figsize=(15, 7))
bars1 = ax.bar(x - width/2, enc_means, width, yerr=enc_stds, capsize=3,
               label='Encode', color='#4C72B0', edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x + width/2, dec_means, width, yerr=dec_stds, capsize=3,
               label='Decode', color='#DD8452', edgecolor='white', linewidth=0.5)

# Add value labels
for bar, m in zip(bars1, enc_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'{m:.1f}s', ha='center', va='bottom', fontsize=8, fontweight='bold')
for bar, m in zip(bars2, dec_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'{m:.1f}s', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Time (s)', fontsize=12)
ax.set_title('Encode / Decode Time per Model (Compress Mode, MSE Only)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=30, ha='right')
ax.set_yscale('log')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('unified_results/encode_decode_time_bar.png', dpi=150, bbox_inches='tight')
print('Saved: unified_results/encode_decode_time_bar.png')
