import os
import sys
import traceback

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
_log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'debug.log')

import json
import numpy as np
import torch
import time
import xarray as xr

from cra5.models.compressai.zoo import vaeformer_pretrained

# --- Config ---
DATA_ROOT = '/data/run01/scxj523/zsh/project/AIForCompression/Data/ERA5/2024'
TIME_STAMP = '2024-06-01T00:00:00'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

VNAMES = dict(
    pressure=['z', 'q', 'u', 'v', 't', 'r', 'w'],
    single=['v10', 'u10', 'v100', 'u100', 't2m', 'tcc', 'sp', 'tp', 'msl'],
)
PRESSURE_LEVELS = [
    1000., 975., 950., 925., 900., 875., 850., 825., 800.,
    775., 750., 700., 650., 600., 550., 500., 450., 400.,
    350., 300., 250., 225., 200., 175., 150., 125., 100.,
    70., 50., 30., 20., 10., 7., 5., 3., 2., 1.,
]
TOTAL_LEVELS = list(PRESSURE_LEVELS)

# --- Mean/Std ---
api_dir = os.path.join(os.path.dirname(__file__), 'cra5', 'api')
with open(os.path.join(api_dir, 'mean_std.json'), 'r') as f:
    mean_std = json.load(f)
with open(os.path.join(api_dir, 'mean_std_single.json'), 'r') as f:
    mean_std_single = json.load(f)

level_mapping = [TOTAL_LEVELS.index(val) for val in PRESSURE_LEVELS if val in TOTAL_LEVELS]
mean_list, std_list = [], []
for vname in VNAMES['pressure']:
    mean_list += [mean_std['mean'][vname][idx] for idx in level_mapping]
    std_list += [mean_std['std'][vname][idx] for idx in level_mapping]
for vname in VNAMES['single']:
    mean_list.append(mean_std_single['mean'][vname])
    std_list.append(mean_std_single['std'][vname])
mean_np = np.array(mean_list, dtype=np.float32)
std_np = np.array(std_list, dtype=np.float32)

mean_t = torch.from_numpy(mean_np[:, None, None]).to(DEVICE)
std_t = torch.from_numpy(std_np[:, None, None]).to(DEVICE)

# --- Read NC data ---
pressure_file = os.path.join(DATA_ROOT, f'{TIME_STAMP}_pressure.nc')
single_file = os.path.join(DATA_ROOT, f'{TIME_STAMP}_single.nc')

one_step = []
pressure_data = xr.open_dataset(pressure_file, engine='netcdf4')
single_data = xr.open_dataset(single_file, engine='netcdf4')
pha_levels = list(pressure_data.pressure_level.data)
lm = [pha_levels.index(val) for val in PRESSURE_LEVELS if val in pha_levels]
for vname in VNAMES['pressure']:
    D = pressure_data[vname].data
    for level in lm:
        one_step.append(D[0][level][None])
for vname in VNAMES['single']:
    D = single_data[vname].data
    if vname == 'tp':
        D = D * 1000
    one_step.append(D)
pressure_data.close()
single_data.close()
original_data = np.concatenate(one_step, 0).astype(np.float32)

num_channels = original_data.shape[0]
print(f"Data shape: {original_data.shape}, channels: {num_channels}")

# --- Load model ---
print(f'Device: {DEVICE}')
net = vaeformer_pretrained(quality=268, pretrained=True).eval().to(DEVICE)

# --- Normalize & forward ---
data = torch.from_numpy(original_data).to(DEVICE)
x = ((data - mean_t) / std_t).unsqueeze(0)

# --- Warmup (compress + decompress) ---
with torch.no_grad():
    net.update()
    compressed = net.compress(x)
    _ = net.decompress(compressed['strings'], compressed['z_shape'])
    if x.is_cuda:
        torch.cuda.synchronize()

# --- Run 5 rounds: encode and decode separately ---
NUM_ROUNDS = 5
encode_times = []
decode_times = []
for i in range(NUM_ROUNDS):
    with torch.no_grad():
        t0 = time.time()
        compressed = net.compress(x)
        if x.is_cuda:
            torch.cuda.synchronize()
        t1 = time.time()
        decompressed = net.decompress(compressed['strings'], compressed['z_shape'])
        if x.is_cuda:
            torch.cuda.synchronize()
        t2 = time.time()
        encode_times.append(t1 - t0)
        decode_times.append(t2 - t1)
    print(f"Round {i+1}: encode={encode_times[-1]:.4f}s, decode={decode_times[-1]:.4f}s")

avg_encode = np.mean(encode_times)
avg_decode = np.mean(decode_times)
print(f"\nAvg encode time ({NUM_ROUNDS} rounds): {avg_encode:.4f}s")
print(f"Avg decode time ({NUM_ROUNDS} rounds): {avg_decode:.4f}s")
print(f"Avg total time ({NUM_ROUNDS} rounds): {avg_encode + avg_decode:.4f}s")

# --- De-normalize & metrics (use last round output) ---
x_hat = decompressed['x_hat']
reconstructed = (x_hat.squeeze(0) * std_t + mean_t).cpu().numpy()

data_range = float(original_data.max() - original_data.min())
if data_range < 1e-6:
    data_range = 1.0
mse = float(np.mean((original_data - reconstructed) ** 2))
psnr = 10 * np.log10(data_range ** 2 / mse) if mse > 1e-10 else float('inf')

H, W = original_data.shape[1], original_data.shape[2]
# BPP from actual bitstream
bitstream_bytes = sum(len(s[0]) for s in compressed['strings'])
bpp = bitstream_bytes * 8.0 / (H * W)
compression_ratio = (num_channels * H * W * 4) / bitstream_bytes if bitstream_bytes > 0 else float('inf')

print(f"MSE: {mse:.8f}")
print(f"PSNR: {psnr:.2f} dB")
print(f"BPP: {bpp:.6f}")
print(f"Compression ratio: {compression_ratio:.2f}:1")
print(f"Bitstream size: {bitstream_bytes} bytes")

# --- Save to summary.json ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
summary = {
    'model': 'CRA5-VAEformer',
    'quality': 268,
    'timestamp': TIME_STAMP,
    'data_shape': list(original_data.shape),
    'device': DEVICE,
    'num_rounds': NUM_ROUNDS,
    'encode_times': [round(t, 4) for t in encode_times],
    'decode_times': [round(t, 4) for t in decode_times],
    'avg_encode_time': round(float(avg_encode), 4),
    'avg_decode_time': round(float(avg_decode), 4),
    'avg_total_time': round(float(avg_encode + avg_decode), 4),
    'mse': round(mse, 8),
    'psnr': round(float(psnr), 2),
    'bpp': round(bpp, 6),
    'compression_ratio': round(compression_ratio, 2),
    'bitstream_bytes': bitstream_bytes,
}
summary_path = os.path.join(SCRIPT_DIR, 'summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"Results saved to {summary_path}")
