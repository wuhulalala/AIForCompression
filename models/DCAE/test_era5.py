"""
Test DCAE model on ERA5 dataset.
Data processing follows run_all_compressai.py: split 268 channels into groups of 3,
apply minmax normalization per group, then feed to the 3-channel DCAE model.

Supports iterating over all checkpoints in a directory (like run_all_compressai.py).

Usage:
  # Single checkpoint
  python test_era5.py --checkpoint /path/to/checkpoint.pth.tar

  # All checkpoints in directory
  python test_era5.py --ckpt_dir ./checkpoints

  # With actual compress/decompress
  python test_era5.py --ckpt_dir ./checkpoints --compress
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

# Checkpoint naming convention: {metric}_{lambda}.pth.tar
# e.g. mse_0.05.pth.tar, msssim_60.5.pth.tar
CHECKPOINT_CONFIGS = [
    {'metric': 'mse', 'lmbda': '0.05'},
    {'metric': 'mse', 'lmbda': '0.025'},
    {'metric': 'mse', 'lmbda': '0.013'},
    {'metric': 'mse', 'lmbda': '0.0067'},
    {'metric': 'mse', 'lmbda': '0.0035'},
    {'metric': 'mse', 'lmbda': '0.0018'},
    {'metric': 'msssim', 'lmbda': '60.5'},
    {'metric': 'msssim', 'lmbda': '31.73'},
    {'metric': 'msssim', 'lmbda': '16.64'},
    {'metric': 'msssim', 'lmbda': '8.73'},
    {'metric': 'msssim', 'lmbda': '4.58'},
    {'metric': 'msssim', 'lmbda': '2.40'},
]


# ─── Mean/Std for z-score normalization ───
def get_mean_std():
    norm_dir = os.path.join(
        os.path.dirname(__file__), '..', '..', 'normalization'
    )
    with open(os.path.join(norm_dir, 'mean_std.json'), 'r') as f:
        mean_std = json.load(f)
    with open(os.path.join(norm_dir, 'mean_std_single.json'), 'r') as f:
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


def get_sample_channel_indices():
    p_count = len(PRESSURE_LEVELS)
    base_single = len(VNAMES['pressure']) * p_count
    return {
        'z_500': VNAMES['pressure'].index('z') * p_count + PRESSURE_LEVELS.index(500.),
        't_850': VNAMES['pressure'].index('t') * p_count + PRESSURE_LEVELS.index(850.),
        't2m': base_single + VNAMES['single'].index('t2m'),
        'u10': base_single + VNAMES['single'].index('u10'),
        'sp': base_single + VNAMES['single'].index('sp'),
    }


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
    C, H, W = data.shape
    pad_h = (divisor - H % divisor) % divisor
    pad_w = (divisor - W % divisor) % divisor
    if pad_h > 0 or pad_w > 0:
        data = np.pad(data, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
    return data, H, W


# ─── Channel grouping (same as run_all_compressai.py) ───
def split_channel_groups(x, group_size=GROUP_SIZE):
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


# ─── Forward pass (uses likelihoods for bpp estimation) ───
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


def find_checkpoints(ckpt_dir):
    """Scan ckpt_dir for .pth.tar files matching known configs, return list of dicts."""
    ckpt_dir = Path(ckpt_dir)
    configs = []
    for cfg in CHECKPOINT_CONFIGS:
        fname = f"{cfg['metric']}_{cfg['lmbda']}.pth.tar"
        fpath = ckpt_dir / fname
        if fpath.exists():
            configs.append({
                'metric': cfg['metric'],
                'lmbda': cfg['lmbda'],
                'checkpoint': str(fpath),
                'model_id': f"DCAE_{cfg['metric']}_lmbda{cfg['lmbda']}",
            })
    # Also pick up any unknown .pth.tar files
    known_names = {f"{c['metric']}_{c['lmbda']}.pth.tar" for c in CHECKPOINT_CONFIGS}
    for f in sorted(ckpt_dir.glob('*.pth.tar')):
        if f.name not in known_names:
            stem = f.stem.replace('.pth', '')
            configs.append({
                'metric': 'unknown',
                'lmbda': stem,
                'checkpoint': str(f),
                'model_id': f"DCAE_{stem}",
            })
    return configs


def load_dcae(checkpoint_path, device):
    net = DCAE()
    net = net.to(device)
    net.eval()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = {}
    for k, v in checkpoint["state_dict"].items():
        state_dict[k.replace("module.", "")] = v
    net.load_state_dict(state_dict)
    net.update()
    return net


# ─── Main ───
def parse_args():
    parser = argparse.ArgumentParser(description='Test DCAE on ERA5 dataset')
    parser.add_argument('--data_root', type=str,
                        default='/data/run01/scxj523/zsh/data/ERA5/2024')
    parser.add_argument('--output_dir', type=str,
                        default='/data/run01/scxj523/zsh/project/AIForCompression/DCAE/results_era5')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to a single DCAE checkpoint (.pth.tar)')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Directory containing multiple checkpoints to iterate over')
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
    samples_dir = output_dir / 'samples'
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Load mean/std for z-score normalization
    mean, std = get_mean_std()
    channel_names = get_channel_names()
    sample_indices = get_sample_channel_indices()
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

    # Save original samples
    for data in all_data:
        raw_crop = data['raw'][:, :data['orig_H'], :data['orig_W']]
        orig_samples = {name: raw_crop[idx] for name, idx in sample_indices.items()}
        np.savez(samples_dir / f"original_{data['timestamp']}.npz", **orig_samples)

    # Build checkpoint list
    ckpt_list = []
    if args.ckpt_dir:
        ckpt_list = find_checkpoints(args.ckpt_dir)
        if not ckpt_list:
            print(f"No .pth.tar checkpoints found in {args.ckpt_dir}")
            return
    elif args.checkpoint:
        stem = Path(args.checkpoint).stem.replace('.pth', '')
        ckpt_list = [{
            'metric': 'unknown',
            'lmbda': stem,
            'checkpoint': args.checkpoint,
            'model_id': f"DCAE_{stem}",
        }]
    else:
        print("ERROR: Must specify --checkpoint or --ckpt_dir")
        return

    print(f"Total models to test: {len(ckpt_list)}")

    # Load existing summary for resume support
    summary_file = output_dir / 'summary.json'
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        tested = set(r['model_id'] for r in summary if 'error' not in r)
    else:
        summary = []
        tested = set()

    start_time = time.time()
    completed = 0

    for i, ckpt_cfg in enumerate(ckpt_list):
        model_id = ckpt_cfg['model_id']
        if model_id in tested:
            print(f"[{i+1}/{len(ckpt_list)}] Skip (already tested): {model_id}")
            completed += 1
            continue

        print(f"\n[{i+1}/{len(ckpt_list)}] Testing: {model_id}")
        print(f"  Checkpoint: {ckpt_cfg['checkpoint']}")

        net = None
        try:
            net = load_dcae(ckpt_cfg['checkpoint'], device)
            total_params = sum(p.numel() for p in net.parameters())

            for data in all_data:
                ts = data['timestamp']
                x = torch.from_numpy(data['padded'][None]).to(device)
                orig_H, orig_W = data['orig_H'], data['orig_W']
                raw_crop = data['raw'][:, :orig_H, :orig_W]

                # Compress/Decompress pass
                fwd = test_compress_decompress_grouped(net, x, orig_H, orig_W)
                x_hat_np = fwd['x_hat'].squeeze(0).cpu().numpy()
                x_hat_denorm = denormalize(x_hat_np, mean, std)
                psnr, mse = calculate_psnr(raw_crop, x_hat_denorm)
                group_rmse = compute_group_rmse(
                    data['norm'][:, :orig_H, :orig_W], x_hat_np, mean, std
                )

                result = {
                    'model_id': model_id,
                    'arch': 'DCAE',
                    'metric': ckpt_cfg['metric'],
                    'lmbda': ckpt_cfg['lmbda'],
                    'params': total_params,
                    'timestamp': ts,
                    'compress': {
                        'mse': mse,
                        'psnr': psnr,
                        'bpp': fwd['bpp'],
                        'bitstream_bytes': fwd['bitstream_bytes'],
                        'original_bytes': fwd['original_bytes'],
                        'compression_ratio': fwd['compression_ratio'],
                        'encode_time': fwd['encode_time'],
                        'decode_time': fwd['decode_time'],
                        'encode_throughput': fwd['original_bytes'] / fwd['encode_time'] / 1e6 if fwd['encode_time'] > 0 else None,
                        'decode_throughput': fwd['original_bytes'] / fwd['decode_time'] / 1e6 if fwd['decode_time'] > 0 else None,
                    },
                    'group_rmse': group_rmse,
                }

                # Save reconstructed samples
                recon_samples = {name: x_hat_denorm[idx] for name, idx in sample_indices.items()}
                np.savez(samples_dir / f"{model_id}_{ts}.npz", **recon_samples)

                summary.append(result)
                print(f"  {ts}: PSNR={psnr:.2f}dB, BPP={fwd['bpp']:.4f}, "
                      f"CR={fwd['compression_ratio']:.2f}x, "
                      f"Enc={fwd['encode_time']:.1f}s ({fwd['original_bytes']/fwd['encode_time']/1e6:.1f}MB/s), Dec={fwd['decode_time']:.1f}s ({fwd['original_bytes']/fwd['decode_time']/1e6:.1f}MB/s)")

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            summary.append({
                'model_id': model_id,
                'arch': 'DCAE',
                'metric': ckpt_cfg['metric'],
                'lmbda': ckpt_cfg['lmbda'],
                'error': str(e),
            })

        finally:
            del net
            torch.cuda.empty_cache()
            gc.collect()

        completed += 1
        elapsed = time.time() - start_time
        avg_per = elapsed / completed
        remaining = avg_per * (len(ckpt_list) - i - 1)
        print(f"  Elapsed: {elapsed/60:.1f}min, ETA: {remaining/60:.1f}min")

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

    print(f"\nDone. Total time: {(time.time()-start_time)/60:.1f}min")
    print(f"Results: {summary_file}")
    print(f"Samples: {samples_dir}")


if __name__ == '__main__':
    main()
