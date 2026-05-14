#!/bin/bash
#SBATCH --job-name=dcae_era5_dbg
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --export=NONE
#SBATCH --output=/data/run01/scxj523/zsh/project/AIForCompression/logs/dcae_era5_debug_%j.log
#SBATCH --error=/data/run01/scxj523/zsh/project/AIForCompression/logs/dcae_era5_debug_%j.log

set -eo pipefail
eval "$(/data/home/scxj523/run/miniconda3/bin/conda shell.bash hook)"
conda activate /data/run01/scxj523/zsh/envs/zsh

python3 << 'PYEOF'
import sys, json, numpy as np, torch
from pathlib import Path

root = Path('/data/run01/scxj523/zsh/project/AIForCompression')
sys.path.insert(0, str(root / 'models' / 'DCAE'))

import test_era5 as t

device = 'cuda'
ckpt = root / 'checkpoints' / 'dcae' / 'mse_0.0018.pth.tar'
data_root = Path('/data/run01/scxj523/zsh/project/Data/ERA5/2024')

pairs = t.find_nc_pairs(str(data_root))
pressure_f, single_f, ts = pairs[0]
with open(root / 'utils' / 'mean_std_outputs' / 'ERA5_2024' / 'mean_std.json', 'r') as f:
    mean_std = json.load(f)
with open(root / 'utils' / 'mean_std_outputs' / 'ERA5_2024' / 'mean_std_single.json', 'r') as f:
    mean_std_single = json.load(f)

level_mapping = [t.TOTAL_LEVELS.index(val) for val in t.PRESSURE_LEVELS if val in t.TOTAL_LEVELS]
mean_list, std_list = [], []
for vname in t.VNAMES['pressure']:
    mean_list += [mean_std['mean'][vname][idx] for idx in level_mapping]
    std_list += [mean_std['std'][vname][idx] for idx in level_mapping]
for vname in t.VNAMES['single']:
    mean_list.append(mean_std_single['mean'][vname])
    std_list.append(mean_std_single['std'][vname])
mean = np.array(mean_list, dtype=np.float32)
std = np.array(std_list, dtype=np.float32)
raw = t.read_nc(pressure_f, single_f)
norm = (raw - mean[:, None, None]) / std[:, None, None]
padded, orig_H, orig_W = t.pad_to_divisor(norm, divisor=128)
x = torch.from_numpy(padded[None]).to(device)

print(f"timestamp={ts}")
print(f"raw_shape={raw.shape}, padded_shape={padded.shape}, orig_H={orig_H}, orig_W={orig_W}")
print(f"device={device}, gpu={torch.cuda.get_device_name(0)}")

net = t.load_dcae(str(ckpt), device)

fwd = t.test_forward_grouped(net, x, orig_H, orig_W)
x_hat_fwd = fwd['x_hat'].squeeze(0).cpu().numpy()
x_hat_fwd_denorm = t.denormalize(x_hat_fwd, mean, std)
psnr_fwd, mse_fwd = t.calculate_psnr(raw[:, :orig_H, :orig_W], x_hat_fwd_denorm)
print(f"forward: psnr={psnr_fwd:.4f} mse={mse_fwd:.4f} bpp={fwd['bpp']:.4f}")

cd = t.test_compress_decompress_grouped(net, x, orig_H, orig_W)
x_hat_cd = cd['x_hat'].squeeze(0).cpu().numpy()
x_hat_cd_denorm = t.denormalize(x_hat_cd, mean, std)
psnr_cd, mse_cd = t.calculate_psnr(raw[:, :orig_H, :orig_W], x_hat_cd_denorm)
print(f"compress: psnr={psnr_cd:.4f} mse={mse_cd:.4f} bpp={cd['bpp']:.4f}")

diff = np.max(np.abs(x_hat_fwd - x_hat_cd))
print(f"max_abs_diff_norm={diff:.6f}")
PYEOF
