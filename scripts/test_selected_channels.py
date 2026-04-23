"""
Unified test script for all models on ERA5 dataset.
Only tests 6 target channels: z500, t850, v10, u10, t2m, msl.
Grouped models get only these 6 channels (2 groups of 3).
CRA5 still uses full 268 channels, then extracts 6 for metrics.
Runs 5 rounds of compress/decompress to average encode/decode time.
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
import xarray as xr

# ─── ERA5 definitions ───
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
IN_CHANNELS = len(VNAMES['pressure']) * len(PRESSURE_LEVELS) + len(VNAMES['single'])
GROUP_SIZE = 3
NUM_ROUNDS = 5

# Target channels: indices in the full 268-channel array
def get_target_channel_indices():
    p_count = len(PRESSURE_LEVELS)
    base_single = len(VNAMES['pressure']) * p_count
    return {
        'z500': VNAMES['pressure'].index('z') * p_count + PRESSURE_LEVELS.index(500.),
        't850': VNAMES['pressure'].index('t') * p_count + PRESSURE_LEVELS.index(850.),
        'v10': base_single + VNAMES['single'].index('v10'),
        'u10': base_single + VNAMES['single'].index('u10'),
        't2m': base_single + VNAMES['single'].index('t2m'),
        'msl': base_single + VNAMES['single'].index('msl'),
    }

TARGET_INDICES = get_target_channel_indices()
TARGET_NAMES = list(TARGET_INDICES.keys())
TARGET_IDX_LIST = [TARGET_INDICES[n] for n in TARGET_NAMES]


def get_mean_std():
    cra5_api_dir = os.path.join(os.path.dirname(__file__), 'CRA5', 'cra5', 'api')
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


def get_target_mean_std(full_mean, full_std):
    return full_mean[TARGET_IDX_LIST], full_std[TARGET_IDX_LIST]


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


def pad_to_divisor(data, divisor=128):
    C, H, W = data.shape
    pad_h = (divisor - H % divisor) % divisor
    pad_w = (divisor - W % divisor) % divisor
    if pad_h > 0 or pad_w > 0:
        data = np.pad(data, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
    return data, H, W


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
    scale = (cmax - cmin).clamp(min=1e-8)
    normed = (chunk - cmin) / scale
    return normed, cmin, scale


def minmax_denormalize(chunk, cmin, scale):
    return chunk * scale + cmin


def compute_bitstream_size(strings):
    total = 0
    for s_list in strings:
        for s in s_list:
            total += len(s)
    return total


def denormalize(data_np, mean, std):
    return data_np * std[:, None, None] + mean[:, None, None]


def calculate_psnr(original, reconstructed):
    orig64 = original.astype(np.float64)
    recon64 = reconstructed.astype(np.float64)
    mse = float(np.mean((orig64 - recon64) ** 2))
    if mse < 1e-10:
        return float('inf'), mse
    data_range = float(orig64.max() - orig64.min())
    if data_range < 1e-6:
        data_range = 1.0
    return float(10 * np.log10(data_range ** 2 / mse)), mse


def extract_target_channels(raw_268):
    return raw_268[TARGET_IDX_LIST]


def compute_target_rmse(raw_orig_6ch, raw_recon_6ch):
    rmse = {}
    for i, name in enumerate(TARGET_NAMES):
        diff = raw_orig_6ch[i].astype(np.float64) - raw_recon_6ch[i].astype(np.float64)
        rmse[name] = float(np.sqrt(np.mean(diff ** 2)))
    return rmse


# ─── Grouped compress/decompress with N rounds (for 6-ch input) ───
@torch.no_grad()
def test_compress_decompress_grouped(model, x, orig_H, orig_W, num_rounds=NUM_ROUNDS):
    groups = split_channel_groups(x)
    x_hat_parts = []
    total_bitstream_bytes = 0
    all_encode_times = []
    all_decode_times = []

    for round_idx in range(num_rounds):
        round_enc, round_dec = 0.0, 0.0
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
            round_enc += t1 - t0
            round_dec += t2 - t1
            if round_idx == 0:
                total_bitstream_bytes += compute_bitstream_size(compressed['strings'])
                x_hat_chunk = minmax_denormalize(decompressed['x_hat'], cmin, scale)
                x_hat_parts.append(x_hat_chunk[:, :actual_c, :orig_H, :orig_W])
        all_encode_times.append(round_enc)
        all_decode_times.append(round_dec)

    x_hat = torch.cat(x_hat_parts, dim=1)
    num_ch = x.shape[1]
    original_bytes = num_ch * orig_H * orig_W * 4
    bpp = total_bitstream_bytes * 8.0 / (orig_H * orig_W)
    return {
        'x_hat': x_hat,
        'bpp': bpp,
        'bitstream_bytes': total_bitstream_bytes,
        'original_bytes': original_bytes,
        'compression_ratio': original_bytes / total_bitstream_bytes if total_bitstream_bytes > 0 else float('inf'),
        'encode_time': sum(all_encode_times) / num_rounds,
        'decode_time': sum(all_decode_times) / num_rounds,
        'encode_times': all_encode_times,
        'decode_times': all_decode_times,
    }


# ─── CRA5 compress/decompress (native 268-ch) ───
@torch.no_grad()
def test_cra5_compress_decompress(model, x, orig_H, orig_W, num_rounds=NUM_ROUNDS):
    all_encode_times = []
    all_decode_times = []
    total_bitstream_bytes = 0
    x_hat = None

    for round_idx in range(num_rounds):
        t0 = time.time()
        compressed = model.compress(x)
        if x.is_cuda:
            torch.cuda.synchronize()
        t1 = time.time()
        decompressed = model.decompress(compressed['strings'], compressed['z_shape'])
        if x.is_cuda:
            torch.cuda.synchronize()
        t2 = time.time()
        all_encode_times.append(t1 - t0)
        all_decode_times.append(t2 - t1)
        if round_idx == 0:
            total_bitstream_bytes = sum(len(s[0]) for s in compressed['strings'])
            x_hat = decompressed['x_hat']

    original_bytes = IN_CHANNELS * orig_H * orig_W * 4
    bpp = total_bitstream_bytes * 8.0 / (orig_H * orig_W)
    return {
        'x_hat': x_hat,
        'bpp': bpp,
        'bitstream_bytes': total_bitstream_bytes,
        'original_bytes': original_bytes,
        'compression_ratio': original_bytes / total_bitstream_bytes if total_bitstream_bytes > 0 else float('inf'),
        'encode_time': sum(all_encode_times) / num_rounds,
        'decode_time': sum(all_decode_times) / num_rounds,
        'encode_times': all_encode_times,
        'decode_times': all_decode_times,
    }


# ═══════════════════════════════════════════════════════
# Model loaders
# ═══════════════════════════════════════════════════════

def load_dcae(checkpoint_path, device):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DCAE'))
    from models import DCAE
    net = DCAE().to(device).eval()
    ckpt = torch.load(checkpoint_path, map_location=device)
    sd = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
    net.load_state_dict(sd)
    net.update()
    return net


def load_weconvene(checkpoint_path, device):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'WeConvene'))
    from model import TCM_residual_wave_two_entropy_modified_y_downsample_8
    net = TCM_residual_wave_two_entropy_modified_y_downsample_8(
        config=[2, 2, 2, 2, 2, 2], head_dim=[8, 16, 32, 32, 16, 8],
        drop_path_rate=0.0, N=128, M=320,
    ).to(device).eval()
    ckpt = torch.load(checkpoint_path, map_location=device)
    sd = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
    net.load_state_dict(sd)
    net.update()
    return net


def load_lictcm(checkpoint_path, device, N=64):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'LIC_TCM'))
    from models import TCM
    net = TCM(
        config=[2, 2, 2, 2, 2, 2], head_dim=[8, 16, 32, 32, 16, 8],
        drop_path_rate=0.0, N=N, M=320,
    ).to(device).eval()
    ckpt = torch.load(checkpoint_path, map_location=device)
    sd = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
    net.load_state_dict(sd)
    net.update()
    return net


def load_cra5(device):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CRA5'))
    from cra5.models.compressai.zoo import vaeformer_pretrained
    net = vaeformer_pretrained(quality=268, pretrained=True).eval().to(device)
    net.update()
    return net


def load_compressai(arch, metric, quality, device):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CRA5'))
    from cra5.models.compressai.zoo.image import _load_model
    net = _load_model(arch, metric, quality, pretrained=True, progress=True)
    net = net.to(device).eval()
    net.update()
    return net


# ═══════════════════════════════════════════════════════
# Checkpoint configs
# ═══════════════════════════════════════════════════════

DCAE_CONFIGS = [
    {'metric': 'mse', 'lmbda': '0.05'},
    {'metric': 'mse', 'lmbda': '0.025'},
    {'metric': 'mse', 'lmbda': '0.013'},
    {'metric': 'mse', 'lmbda': '0.0067'},
    {'metric': 'mse', 'lmbda': '0.0035'},
    {'metric': 'mse', 'lmbda': '0.0018'},
]

WECONVENE_CONFIGS = [
    {'metric': 'mse', 'lmbda': '0.05',   'filename': '0.05175_checkpoint.pth.tar'},
    {'metric': 'mse', 'lmbda': '0.025',  'filename': '0.025175_checkpoint.pth.tar'},
    {'metric': 'mse', 'lmbda': '0.013',  'filename': '0.013checkpoint_best.pth.tar'},
    {'metric': 'mse', 'lmbda': '0.0067', 'filename': '0.0067175_checkpoint.pth.tar'},
    {'metric': 'mse', 'lmbda': '0.0035', 'filename': '0.0035175_checkpoint.pth.tar'},
    {'metric': 'mse', 'lmbda': '0.0025', 'filename': '0.0025175_checkpoint.pth.tar'},
]

LICTCM_CONFIGS = [
    {'metric': 'mse', 'lmbda': '0.05',   'filename': '0.05.pth.tar',   'N': 64},
    {'metric': 'mse', 'lmbda': '0.025',  'filename': '0.025.pth.tar',  'N': 64},
    {'metric': 'mse', 'lmbda': '0.013',  'filename': '0.013.pth.tar',  'N': 64},
    {'metric': 'mse', 'lmbda': '0.0067', 'filename': '0.0067.pth.tar', 'N': 64},
    {'metric': 'mse', 'lmbda': '0.0035', 'filename': '0.0035.pth.tar', 'N': 64},
    {'metric': 'mse', 'lmbda': '0.0025', 'filename': '0.0025.pth.tar', 'N': 64},
]


def get_compressai_configs():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CRA5'))
    from cra5.models.compressai.zoo.image import model_urls
    configs = []
    for arch, metric_dict in model_urls.items():
        if arch == 'vaeformer-pretrained':
            continue
        for metric, quality_dict in metric_dict.items():
            for quality, url in quality_dict.items():
                if isinstance(url, str) and url.startswith('http'):
                    configs.append({'arch': arch, 'metric': metric, 'quality': quality})
    return configs


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description='Test all models on selected ERA5 channels')
    parser.add_argument('--data_root', type=str,
                        default='/data/run01/scxj523/zsh/project/AIForCompression/Data/ERA5/2024')
    parser.add_argument('--project_root', type=str,
                        default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--max_samples', type=int, default=1)
    parser.add_argument('--models', nargs='+', default=None,
                        help='Filter: DCAE WeConvene LIC_TCM CRA5 CompressAI')
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    if args.output_dir is None:
        args.output_dir = os.path.join(args.project_root, 'results_selected_channels')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mean, std = get_mean_std()
    target_mean, target_std = get_target_mean_std(mean, std)
    ckpt_base = Path(args.project_root) / 'checkpoints'

    # Load ERA5 data
    file_pairs = find_nc_pairs(args.data_root)
    if not file_pairs:
        print(f"No ERA5 file pairs found in {args.data_root}")
        return
    if args.max_samples > 0:
        file_pairs = file_pairs[:args.max_samples]
    print(f"Using {len(file_pairs)} ERA5 timestamps")

    all_data = []
    for pressure_f, single_f, ts in file_pairs:
        raw_268 = read_nc(pressure_f, single_f)
        norm_268 = (raw_268 - mean[:, None, None]) / std[:, None, None]
        raw_6ch = extract_target_channels(raw_268)
        norm_6ch = (raw_6ch - target_mean[:, None, None]) / target_std[:, None, None]
        padded_6ch, orig_H, orig_W = pad_to_divisor(norm_6ch, divisor=128)
        all_data.append({
            'timestamp': ts, 'raw_268': raw_268, 'norm_268': norm_268,
            'raw_6ch': raw_6ch, 'norm_6ch': norm_6ch,
            'padded_6ch': padded_6ch, 'orig_H': orig_H, 'orig_W': orig_W,
        })

    # Build job list
    allowed = set(args.models) if args.models else {'DCAE', 'WeConvene', 'LIC_TCM', 'CRA5', 'CompressAI'}
    jobs = []

    if 'DCAE' in allowed:
        for cfg in DCAE_CONFIGS:
            ckpt = ckpt_base / 'dcae' / f"{cfg['metric']}_{cfg['lmbda']}.pth.tar"
            if ckpt.exists():
                jobs.append(('DCAE', f"DCAE_lmbda{cfg['lmbda']}", 'grouped',
                             lambda d, c=str(ckpt): load_dcae(c, d), cfg))

    if 'WeConvene' in allowed:
        for cfg in WECONVENE_CONFIGS:
            ckpt = ckpt_base / 'weconvene' / cfg['filename']
            if ckpt.exists():
                jobs.append(('WeConvene', f"WeConvene_lmbda{cfg['lmbda']}", 'grouped',
                             lambda d, c=str(ckpt): load_weconvene(c, d), cfg))

    if 'LIC_TCM' in allowed:
        for cfg in LICTCM_CONFIGS:
            ckpt = ckpt_base / 'lictcm' / cfg['filename']
            if ckpt.exists():
                jobs.append(('LIC_TCM', f"LICTCM_lmbda{cfg['lmbda']}", 'grouped',
                             lambda d, c=str(ckpt), n=cfg['N']: load_lictcm(c, d, N=n), cfg))

    if 'CRA5' in allowed:
        jobs.append(('CRA5', 'CRA5_vaeformer_q268', 'cra5',
                     lambda d: load_cra5(d), {'metric': 'mse', 'lmbda': 'native'}))

    if 'CompressAI' in allowed:
        for cfg in get_compressai_configs():
            mid = f"{cfg['arch']}_{cfg['metric']}_q{cfg['quality']}"
            jobs.append(('CompressAI', mid, 'compressai',
                         lambda d, a=cfg['arch'], m=cfg['metric'], q=cfg['quality']: load_compressai(a, m, q, d),
                         cfg))

    print(f"Total jobs: {len(jobs)}")

    # Resume support
    summary_file = output_dir / 'summary.json'
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        tested = set(r['model_id'] for r in summary if 'error' not in r)
    else:
        summary = []
        tested = set()

    start_time = time.time()

    for i, (model_name, model_id, test_type, loader_fn, cfg) in enumerate(jobs):
        if model_id in tested:
            print(f"[{i+1}/{len(jobs)}] Skip (done): {model_id}")
            continue

        print(f"[{i+1}/{len(jobs)}] Testing: {model_id}")
        model = None
        try:
            model = loader_fn(device)
            total_params = sum(p.numel() for p in model.parameters())

            for data in all_data:
                orig_H, orig_W = data['orig_H'], data['orig_W']
                raw_6ch = data['raw_6ch'][:, :orig_H, :orig_W]

                if test_type == 'cra5':
                    mean_t = torch.from_numpy(mean[:, None, None]).to(device)
                    std_t = torch.from_numpy(std[:, None, None]).to(device)
                    x = torch.from_numpy(data['norm_268'][:, :orig_H, :orig_W][None]).to(device)
                    # warmup
                    cw = model.compress(x)
                    _ = model.decompress(cw['strings'], cw['z_shape'])
                    if x.is_cuda:
                        torch.cuda.synchronize()
                    fwd = test_cra5_compress_decompress(model, x, orig_H, orig_W)
                    x_hat_268 = (fwd['x_hat'].squeeze(0) * std_t + mean_t).cpu().numpy()
                    x_hat_6ch = x_hat_268[TARGET_IDX_LIST]
                elif test_type == 'compressai':
                    padded64, oH64, oW64 = pad_to_divisor(data['norm_6ch'], divisor=64)
                    x = torch.from_numpy(padded64[None]).to(device)
                    fwd = test_compress_decompress_grouped(model, x, oH64, oW64)
                    x_hat_np = fwd['x_hat'].squeeze(0).cpu().numpy()
                    x_hat_6ch = denormalize(x_hat_np, target_mean, target_std)
                else:
                    x = torch.from_numpy(data['padded_6ch'][None]).to(device)
                    fwd = test_compress_decompress_grouped(model, x, orig_H, orig_W)
                    x_hat_np = fwd['x_hat'].squeeze(0).cpu().numpy()
                    x_hat_6ch = denormalize(x_hat_np, target_mean, target_std)

                psnr, mse = calculate_psnr(raw_6ch, x_hat_6ch)
                target_rmse = compute_target_rmse(raw_6ch, x_hat_6ch)

                result = {
                    'model_name': model_name,
                    'model_id': model_id,
                    'metric': cfg.get('metric', ''),
                    'lmbda': cfg.get('lmbda', ''),
                    'params': total_params,
                    'timestamp': data['timestamp'],
                    'psnr': psnr,
                    'mse': mse,
                    'bpp': fwd['bpp'],
                    'compression_ratio': fwd['compression_ratio'],
                    'encode_time_avg': fwd['encode_time'],
                    'decode_time_avg': fwd['decode_time'],
                    'encode_times': fwd['encode_times'],
                    'decode_times': fwd['decode_times'],
                    'target_rmse': target_rmse,
                }
                summary.append(result)

                rmse_str = ' '.join(f"{k}={v:.4f}" for k, v in target_rmse.items())
                print(f"  {data['timestamp']}: PSNR={psnr:.2f}dB BPP={fwd['bpp']:.4f} "
                      f"CR={fwd['compression_ratio']:.2f}x "
                      f"Enc={fwd['encode_time']:.2f}s Dec={fwd['decode_time']:.2f}s")
                print(f"    RMSE: {rmse_str}")

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            summary.append({
                'model_name': model_name, 'model_id': model_id, 'error': str(e),
            })
        finally:
            del model
            torch.cuda.empty_cache()
            gc.collect()

        elapsed = time.time() - start_time
        print(f"  Elapsed: {elapsed/60:.1f}min")

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

    print(f"\nDone. Total time: {(time.time()-start_time)/60:.1f}min")
    print(f"Results: {summary_file}")


if __name__ == '__main__':
    main()
