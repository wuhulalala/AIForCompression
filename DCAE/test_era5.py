"""
Test DCAE model on ERA5 dataset.
Data processing follows run_all_compressai.py: split 268 channels into groups of 3,
apply minmax normalization per group, then feed to the 3-channel DCAE model.

Usage:
  python test_era5.py --data_root /path/to/ERA5/2024 --gpu 0
  python test_era5.py --data_root /path/to/ERA5/2024 --gpu 0 --checkpoint /path/to/checkpoint.pth.tar --compress
"""

import os
import sys
import json
import time
import math
import argparse
import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr

from models import DCAE

# ─── ERA5 variable definitions ───
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
IN_CHANNELS = len(VNAMES['pressure']) * len(PRESSURE_LEVELS) + len(VNAMES['single'])  # 268
GROUP_SIZE = 3


# ─── Mean/Std for z-score normalization ───
def get_mean_std():
    cra5_api_dir = os.path.join(
        os.path.dirname(__file__), '..', 'CRA5', 'cra5', 'api'
    )
    with open(os.path.join(cra5_api_dir, 'mean_std.json'), 'r') as f:
        mean_std = json.load(f)
    with open(os.path.join(cra5_api_dir, 'mean_std_single.json'), 'r') as f:
        mean_std_single = json.load(f)

    level_mapping = [TOTAL_LEVELS.index(val) for val in PRESSURE_LEVELS if val in TOTAL_LEVELS]
    mean_list, std_list = [], []
    for vname in VNAMES['pressure']:
        mean_list += [mean_std['mean'][vname][idx] for idx in level_mapping]
        std_list += [mean_std['std'][vname][idx] for idx in level_mapping]
    for vname in VNAMES['single']:
        mean_list.append(mean_std_single['mean'][vname])
        std_list.append(mean_std_single['std'][vname])
    return np.array(mean_list, dtype=np.float32), np.array(std_list, dtype=np.float32)


def get_channel_names():
    names = []
    for vname in VNAMES['pressure']:
        for level in PRESSURE_LEVELS:
            names.append(f"{vname}_{int(level)}")
    for vname in VNAMES['single']:
        names.append(vname)
    return names


# ─── Read ERA5 NC files ───
def read_nc(pressure_file, single_file):
    one_step = []
    pressure_data = xr.open_dataset(pressure_file, engine='netcdf4')
    single_data = xr.open_dataset(single_file, engine='netcdf4')
    pha_levels = list(pressure_data.pressure_level.data)
    level_mapping = [pha_levels.index(val) for val in PRESSURE_LEVELS if val in pha_levels]
    for vname in VNAMES['pressure']:
        D = pressure_data[vname].data
        for level in level_mapping:
            one_step.append(D[0][level][None])
    for vname in VNAMES['single']:
        D = single_data[vname].data
        if vname == 'tp':
            D = D * 1000
        one_step.append(D)
    pressure_data.close()
    single_data.close()
    return np.concatenate(one_step, 0).astype(np.float32)


def find_nc_pairs(data_root):
    pairs = []
    for root, dirs, files in os.walk(data_root):
        for f in sorted(files):
            if f.endswith('_pressure.nc'):
                ts = f.replace('_pressure.nc', '')
                single_f = os.path.join(root, f'{ts}_single.nc')
                pressure_f = os.path.join(root, f)
                if os.path.exists(single_f):
                    pairs.append((pressure_f, single_f, ts))
    pairs.sort(key=lambda x: x[2])
    return pairs


# ─── Padding / Cropping ───
def pad_to_divisor(data, divisor=128):
    """Pad to be divisible by divisor. DCAE uses p=128 for padding."""
    C, H, W = data.shape
    pad_h = (divisor - H % divisor) % divisor
    pad_w = (divisor - W % divisor) % divisor
    if pad_h > 0 or pad_w > 0:
        data = np.pad(data, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
    return data, H, W


# ─── Channel grouping (same as run_all_compressai.py) ───
def split_channel_groups(x, group_size=GROUP_SIZE):
    """Split (B, C, H, W) into groups of `group_size` channels.
    If the last group has fewer channels, pad by repeating the last channel."""
    B, C, H, W = x.shape
    groups = []
    for i in range(0, C, group_size):
        chunk = x[:, i:i + group_size]
        actual_c = chunk.shape[1]
        if actual_c < group_size:
            pad = chunk[:, -1:].expand(B, group_size - actual_c, H, W)
            chunk = torch.cat([chunk, pad], dim=1)
        groups.append((chunk, actual_c))
    return groups


def minmax_normalize(chunk):
    """Per-channel minmax normalization to [0, 1]."""
    B, C, H, W = chunk.shape
    cmin = chunk.reshape(B, C, -1).min(dim=-1, keepdim=True).values.unsqueeze(-1)
    cmax = chunk.reshape(B, C, -1).max(dim=-1, keepdim=True).values.unsqueeze(-1)
    scale = cmax - cmin
    scale = scale.clamp(min=1e-8)
    normed = (chunk - cmin) / scale
    return normed, cmin, scale


def minmax_denormalize(chunk, cmin, scale):
    return chunk * scale + cmin


# ─── Metrics ───
def calculate_psnr(original, reconstructed, data_range=None):
    orig64 = original.astype(np.float64)
    recon64 = reconstructed.astype(np.float64)
    mse = float(np.mean((orig64 - recon64) ** 2))
    if mse < 1e-10:
        return float('inf'), mse
    if data_range is None:
        data_range = float(orig64.max() - orig64.min())
        if data_range < 1e-6:
            data_range = 1.0
    return float(10 * np.log10(data_range ** 2 / mse)), mse


def denormalize(data_np, mean, std):
    return data_np * std[:, None, None] + mean[:, None, None]


def compute_bitstream_size(strings):
    total = 0
    for s_list in strings:
        for s in s_list:
            total += len(s)
    return total


def compute_group_rmse(x_orig_norm, x_hat_norm, mean, std):
    """Compute per-variable RMSE in physical space."""
    x_denorm = x_orig_norm * std[:, None, None] + mean[:, None, None]
    x_hat_denorm = x_hat_norm * std[:, None, None] + mean[:, None, None]
    group_metrics = {}
    idx = 0
    for vname in VNAMES['pressure']:
        rmse_list = []
        for _ in PRESSURE_LEVELS:
            rmse_list.append(float(np.sqrt(np.mean((x_denorm[idx] - x_hat_denorm[idx]) ** 2))))
            idx += 1
        group_metrics[vname] = {
            'mean_rmse': float(np.mean(rmse_list)),
            'max_rmse': float(np.max(rmse_list)),
        }
    for vname in VNAMES['single']:
        group_metrics[vname] = {
            'rmse': float(np.sqrt(np.mean((x_denorm[idx] - x_hat_denorm[idx]) ** 2)))
        }
        idx += 1
    return group_metrics


# ─── Forward pass (no actual compression, uses likelihoods for bpp estimation) ───
@torch.no_grad()
def test_forward_grouped(model, x, orig_H, orig_W):
    groups = split_channel_groups(x)
    x_hat_parts = []
    total_bpp_bits = 0.0
    total_time = 0.0

    for chunk, actual_c in groups:
        normed, cmin, scale = minmax_normalize(chunk)
        t0 = time.time()
        out = model(normed)
        if x.is_cuda:
            torch.cuda.synchronize()
        total_time += time.time() - t0

        out['x_hat'].clamp_(0, 1)
        x_hat_chunk = minmax_denormalize(out['x_hat'], cmin, scale)
        x_hat_parts.append(x_hat_chunk[:, :actual_c, :orig_H, :orig_W])

        for key, likelihoods in out['likelihoods'].items():
            lk = likelihoods[:, :, :orig_H, :orig_W] if likelihoods.shape[-2] >= orig_H else likelihoods
            total_bpp_bits += -torch.log2(lk.clamp(min=1e-10)).sum().item()

    x_hat = torch.cat(x_hat_parts, dim=1)
    bpp = total_bpp_bits / (orig_H * orig_W)
    return {
        'x_hat': x_hat,
        'bpp': bpp,
        'compression_ratio': (IN_CHANNELS * 32.0) / bpp if bpp > 0 else float('inf'),
        'time': total_time,
    }


# ─── Real compress/decompress pass ───
@torch.no_grad()
def test_compress_decompress_grouped(model, x, orig_H, orig_W):
    groups = split_channel_groups(x)
    x_hat_parts = []
    total_bitstream_bytes = 0
    total_encode_time = 0.0
    total_decode_time = 0.0

    for chunk, actual_c in groups:
        normed, cmin, scale = minmax_normalize(chunk)
        t0 = time.time()
        compressed = model.compress(normed)
        if x.is_cuda:
            torch.cuda.synchronize()
        t1 = time.time()
        decompressed = model.decompress(compressed['strings'], compressed['shape'])
        if x.is_cuda:
            torch.cuda.synchronize()
        t2 = time.time()

        total_encode_time += t1 - t0
        total_decode_time += t2 - t1
        total_bitstream_bytes += compute_bitstream_size(compressed['strings'])

        x_hat_chunk = minmax_denormalize(decompressed['x_hat'], cmin, scale)
        x_hat_parts.append(x_hat_chunk[:, :actual_c, :orig_H, :orig_W])

    x_hat = torch.cat(x_hat_parts, dim=1)
    original_bytes = IN_CHANNELS * orig_H * orig_W * 4
    bpp = total_bitstream_bytes * 8.0 / (orig_H * orig_W)
    return {
        'x_hat': x_hat,
        'bpp': bpp,
        'bitstream_bytes': total_bitstream_bytes,
        'original_bytes': original_bytes,
        'compression_ratio': original_bytes / total_bitstream_bytes if total_bitstream_bytes > 0 else float('inf'),
        'encode_time': total_encode_time,
        'decode_time': total_decode_time,
    }


# ─── Main ───
def parse_args():
    parser = argparse.ArgumentParser(description='Test DCAE on ERA5 dataset')
    parser.add_argument('--data_root', type=str,
                        default='/home/bingxing2/home/scx9kvs/zsh/backup/data/ERA5/2024')
    parser.add_argument('--output_dir', type=str,
                        default='/home/bingxing2/home/scx9kvs/zsh/backup/AIForCompression/DCAE/results_era5')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to DCAE checkpoint (.pth.tar)')
    parser.add_argument('--compress', action='store_true',
                        help='Also run actual compress/decompress (slower)')
    parser.add_argument('--max_samples', type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load mean/std for z-score normalization
    mean, std = get_mean_std()
    channel_names = get_channel_names()
    print(f"Total ERA5 channels: {IN_CHANNELS} ({len(VNAMES['pressure'])} pressure vars x "
          f"{len(PRESSURE_LEVELS)} levels + {len(VNAMES['single'])} single vars)")
    print(f"Group size: {GROUP_SIZE}, total groups: {math.ceil(IN_CHANNELS / GROUP_SIZE)}")

    # Find ERA5 file pairs
    file_pairs = find_nc_pairs(args.data_root)
    if not file_pairs:
        print(f"No ERA5 file pairs found in {args.data_root}")
        return
    if args.max_samples > 0:
        file_pairs = file_pairs[:args.max_samples]
    print(f"Using {len(file_pairs)} ERA5 timestamps")

    # Read and preprocess data
    all_data = []
    for pressure_f, single_f, ts in file_pairs:
        print(f"Reading {ts}...")
        raw = read_nc(pressure_f, single_f)
        norm = (raw - mean[:, None, None]) / std[:, None, None]
        padded, orig_H, orig_W = pad_to_divisor(norm, divisor=128)
        all_data.append({
            'timestamp': ts,
            'raw': raw,
            'norm': norm,
            'padded': padded,
            'orig_H': orig_H,
            'orig_W': orig_W,
        })
        print(f"  Raw shape: {raw.shape}, Padded shape: {padded.shape}, "
              f"orig H={orig_H}, W={orig_W}")

    # Load DCAE model
    print("\nLoading DCAE model...")
    net = DCAE()
    net = net.to(device)
    net.eval()

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = {}
        for k, v in checkpoint["state_dict"].items():
            state_dict[k.replace("module.", "")] = v
        net.load_state_dict(state_dict)
    else:
        print("WARNING: No checkpoint provided, using random weights. Results will be meaningless.")

    net.update()

    total_params = sum(p.numel() for p in net.parameters())
    print(f"Model parameters: {total_params:,}")

    # Run evaluation
    results = []
    for data in all_data:
        ts = data['timestamp']
        print(f"\n{'='*60}")
        print(f"Testing timestamp: {ts}")
        print(f"{'='*60}")

        x = torch.from_numpy(data['padded'][None]).to(device)
        orig_H, orig_W = data['orig_H'], data['orig_W']
        raw_crop = data['raw'][:, :orig_H, :orig_W]

        # Forward pass
        print("Running forward pass (grouped, 3-channel)...")
        fwd = test_forward_grouped(net, x, orig_H, orig_W)
        x_hat_np = fwd['x_hat'].squeeze(0).cpu().numpy()
        x_hat_denorm = denormalize(x_hat_np, mean, std)
        psnr, mse = calculate_psnr(raw_crop, x_hat_denorm)
        group_rmse = compute_group_rmse(
            data['norm'][:, :orig_H, :orig_W], x_hat_np, mean, std
        )

        result = {
            'model': 'DCAE',
            'timestamp': ts,
            'params': total_params,
            'checkpoint': args.checkpoint or 'none',
            'forward': {
                'mse': mse,
                'psnr': psnr,
                'bpp': fwd['bpp'],
                'compression_ratio': fwd['compression_ratio'],
                'time': fwd['time'],
            },
            'group_rmse': group_rmse,
        }

        print(f"\n--- Forward Results ---")
        print(f"  PSNR:              {psnr:.2f} dB")
        print(f"  BPP:               {fwd['bpp']:.4f}")
        print(f"  Compression Ratio: {fwd['compression_ratio']:.2f}x")
        print(f"  Time:              {fwd['time']:.2f}s")

        print(f"\n--- Per-Variable RMSE (physical space) ---")
        for vname in VNAMES['pressure']:
            m = group_rmse[vname]
            print(f"  {vname:>3s}: mean_rmse={m['mean_rmse']:.4f}, max_rmse={m['max_rmse']:.4f}")
        for vname in VNAMES['single']:
            m = group_rmse[vname]
            print(f"  {vname:>4s}: rmse={m['rmse']:.4f}")

        # Actual compress/decompress
        if args.compress:
            print("\nRunning actual compress/decompress (grouped)...")
            comp = test_compress_decompress_grouped(net, x, orig_H, orig_W)
            comp_hat_np = comp['x_hat'].squeeze(0).cpu().numpy()
            comp_hat_denorm = denormalize(comp_hat_np, mean, std)
            comp_psnr, comp_mse = calculate_psnr(raw_crop, comp_hat_denorm)
            result['compress'] = {
                'mse': comp_mse,
                'psnr': comp_psnr,
                'bpp': comp['bpp'],
                'bitstream_bytes': comp['bitstream_bytes'],
                'original_bytes': comp['original_bytes'],
                'compression_ratio': comp['compression_ratio'],
                'encode_time': comp['encode_time'],
                'decode_time': comp['decode_time'],
            }
            print(f"\n--- Compress/Decompress Results ---")
            print(f"  PSNR:              {comp_psnr:.2f} dB")
            print(f"  BPP:               {comp['bpp']:.4f}")
            print(f"  Bitstream:         {comp['bitstream_bytes']:,} bytes")
            print(f"  Original:          {comp['original_bytes']:,} bytes")
            print(f"  Compression Ratio: {comp['compression_ratio']:.2f}x")
            print(f"  Encode Time:       {comp['encode_time']:.2f}s")
            print(f"  Decode Time:       {comp['decode_time']:.2f}s")

        results.append(result)

    # Save results
    summary_file = output_dir / 'dcae_era5_results.json'
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {summary_file}")


if __name__ == '__main__':
    main()
