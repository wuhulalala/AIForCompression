"""Generate MSE / RMSE error bar chart for selected channels results."""
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

mse_dict = defaultdict(list)

# Load selected channels results
with open('results_selected_channels/summary.json') as f:
    data = json.load(f)

for r in data:
    if 'error' in r:
        continue
    # Only use mse metric results
    if r.get('metric', 'mse') != 'mse':
        continue
    model_name = r['model_name']
    if model_name == 'CompressAI':
        mid = r['model_id']
        if '_mse_' in mid:
            arch = mid.split('_mse_')[0]
        else:
            continue
        mse_dict[arch].append(r['mse'])
    elif model_name == 'CRA5':
        continue  # CRA5 from separate file
    else:
        mse_dict[model_name].append(r['mse'])

# Add CRA5
with open('/data/run01/scxj523/zsh/project/cra5/summary.json') as f:
    cra5 = json.load(f)
mse_dict['CRA5'].append(cra5['mse'])

# Sort by average MSE
names = sorted(mse_dict.keys(), key=lambda x: np.mean(mse_dict[x]))
mse_means = [np.mean(mse_dict[n]) for n in names]
mse_stds = [np.std(mse_dict[n]) for n in names]
rmse_means = [np.sqrt(np.mean(mse_dict[n])) for n in names]
rmse_stds = [np.std([np.sqrt(v) for v in mse_dict[n]]) for n in names]

x = np.arange(len(names))
width = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# MSE
bars1 = ax1.bar(x, mse_means, width * 2, yerr=mse_stds, capsize=3,
                color='#4C72B0', edgecolor='white', linewidth=0.5)
for bar, m in zip(bars1, mse_means):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f'{m:.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax1.set_xlabel('Model', fontsize=12)
ax1.set_ylabel('MSE', fontsize=12)
ax1.set_title('Average MSE per Model', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(names, rotation=30, ha='right')
ax1.set_ylim(bottom=0)

# RMSE
bars2 = ax2.bar(x, rmse_means, width * 2, yerr=rmse_stds, capsize=3,
                color='#DD8452', edgecolor='white', linewidth=0.5)
for bar, m in zip(bars2, rmse_means):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f'{m:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax2.set_xlabel('Model', fontsize=12)
ax2.set_ylabel('RMSE', fontsize=12)
ax2.set_title('Average RMSE per Model', fontsize=14)
ax2.set_xticks(x)
ax2.set_xticklabels(names, rotation=30, ha='right')
ax2.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig('results_selected_channels/mse_rmse_bar.png', dpi=150, bbox_inches='tight')
print('Saved: results_selected_channels/mse_rmse_bar.png')
