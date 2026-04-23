"""Generate encode/decode time bar chart for selected channels results.
Non-CRA5 models divide by 2 (computed on 6 channels, need per-3-channel time).
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

enc = defaultdict(list)
dec = defaultdict(list)

# Load selected channels results
with open('results_selected_channels/summary.json') as f:
    data = json.load(f)

for r in data:
    if 'error' in r:
        continue
    model_name = r['model_name']
    if model_name == 'CompressAI':
        mid = r['model_id']
        for sep in ['_mse_', '_ms-ssim_']:
            if sep in mid:
                arch = mid.split(sep)[0]
                break
        else:
            arch = mid
        enc[arch].append(r['encode_time_avg'])
        dec[arch].append(r['decode_time_avg'])
    else:
        enc[model_name].append(r['encode_time_avg'])
        dec[model_name].append(r['decode_time_avg'])

# Divide non-CRA5 by 2 (6 channels -> 3 channels equivalent)
for name in enc:
    enc[name] = [t / 2 for t in enc[name]]
    dec[name] = [t / 2 for t in dec[name]]

# Add CRA5 (no division)
with open('/data/run01/scxj523/zsh/project/cra5/summary.json') as f:
    cra5 = json.load(f)
enc['CRA5'] = cra5['encode_times']
dec['CRA5'] = cra5['decode_times']

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
            f'{m:.2f}s', ha='center', va='bottom', fontsize=8, fontweight='bold')
for bar, m in zip(bars2, dec_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'{m:.2f}s', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Time (s)', fontsize=12)
ax.set_title('Encode / Decode Time per Model (Selected Channels, 3ch equivalent)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=30, ha='right')
ax.set_yscale('log')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('results_selected_channels/encode_decode_time_bar.png', dpi=150, bbox_inches='tight')
print('Saved: results_selected_channels/encode_decode_time_bar.png')
