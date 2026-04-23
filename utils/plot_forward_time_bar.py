"""Generate inference time bar chart using forward time data."""
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# CompressAI data from summary_forward.json
with open('CRA5/results/all_compressai/summary_forward.json') as f:
    content = f.read().strip()
data = json.loads('[' + content)

compressai_times = defaultdict(list)
for r in data:
    if r.get('metric', '') != 'mse':
        continue
    t = r.get('forward', {}).get('time', None)
    if t is not None:
        compressai_times[r['arch']].append(t)

# Remove warmup (first value per arch if it's an outlier > 2x median)
for arch in compressai_times:
    vals = compressai_times[arch]
    if len(vals) > 2:
        median = sorted(vals)[len(vals)//2]
        compressai_times[arch] = [v for v in vals if v < 2 * median]

# All models: name -> (mean, std)
models = {}

# From the previous chart (DCAE, LIC-TCM, WeConvene)
models['CRA5'] = (0.33, 0.0)
models['DCAE'] = (11.1, 4.5)
models['LIC-TCM'] = (11.0, 5.0)
models['WeConvene'] = (12.3, 3.0)

# CompressAI models
for arch, vals in compressai_times.items():
    models[arch] = (np.mean(vals), np.std(vals))

# Sort by mean time
names = sorted(models.keys(), key=lambda x: models[x][0])
means = [models[n][0] for n in names]
stds = [models[n][1] for n in names]

# Plot
fig, ax = plt.subplots(figsize=(14, 6))
colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
bars = ax.bar(names, means, yerr=stds, capsize=4, color=colors, edgecolor='white', linewidth=0.5)

for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
            f'{m:.1f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Inference Time (s)', fontsize=12)
ax.set_title('Average Inference Time per Model (Forward, MSE Only)', fontsize=14)
ax.set_ylim(bottom=0)
ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig('unified_results/inference_time_bar_forward.png', dpi=150, bbox_inches='tight')
print('Saved: unified_results/inference_time_bar_forward.png')
