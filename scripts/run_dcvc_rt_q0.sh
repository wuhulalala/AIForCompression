#!/bin/bash
#SBATCH --job-name=dcvc_rt_q0
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/data/run01/scxj523/zsh/project/AIForCompression/logs/dcvc_rt_q0_%j.log
#SBATCH --error=/data/run01/scxj523/zsh/project/AIForCompression/logs/dcvc_rt_q0_%j.log

set -euo pipefail
export PATH="/data/run01/scxj523/zsh/envs/zsh/bin:$PATH"
export CONDA_PREFIX=/data/run01/scxj523/zsh/envs/zsh
cd /data/run01/scxj523/zsh/project/AIForCompression

python -u -c "
import sys, json, os
sys.path.insert(0, 'scripts')
from test_video_intra_era5 import *

args = parse_args()
args.model = 'DCVC_RT'
args.max_samples = 1
device = 'cuda'

dcvc_root = '/data/run01/scxj523/zsh/project/AIForCompression/models/DCVC'
cpp_root = os.path.join(dcvc_root, 'src', 'cpp')
ext_root = os.path.join(dcvc_root, 'src', 'layers', 'extensions', 'inference')
clear_src_modules()
sys.path.insert(0, dcvc_root)
sys.path.insert(0, cpp_root)
sys.path.insert(0, ext_root)

from src.models.image_model import DMCI

ckpt = torch.load(args.dcvc_rt_checkpoint, map_location='cpu')
if 'state_dict' in ckpt: ckpt = ckpt['state_dict']
if 'net' in ckpt: ckpt = ckpt['net']
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
consume_prefix_in_state_dict_if_present(ckpt, prefix='module.')

model = DMCI()
model.load_state_dict(ckpt)
model = model.to(device).eval()
model.update()
params = sum(p.numel() for p in model.parameters())

pairs = find_nc_pairs(args.data_root)[:1]
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
summary = []

for qp in (0,):  # only q0, skip q21 which crashes in C++ entropy coder
    print(f'[progress] start DCVC_RT_Intra_q{qp}', flush=True)
    try:
        for pressure, single, ts in pairs:
            raw = read_nc(pressure, single)
            x = torch.from_numpy(raw[None]).to(device)
            recon_parts = []
            total_bytes = 0
            total_enc = 0.0
            total_dec = 0.0
            for group_idx, (chunk, actual_c) in enumerate(split_channel_groups(x)):
                print(f'[progress] DCVC_RT_Intra_q{qp} {ts} group {group_idx}', flush=True)
                normed, cmin, scale = minmax_normalize(chunk)
                x_pad, orig_h, orig_w = pad_replicate(normed.float(), 64)
                enc_start = time.time()
                compressed = model.compress(x_pad, qp)
                if x.is_cuda: torch.cuda.synchronize()
                enc_end = time.time()
                sps = {'height': x_pad.shape[-2], 'width': x_pad.shape[-1], 'qp': qp, 'ec_part': 0}
                decompressed = model.decompress(compressed['bit_stream'], sps, qp)
                if x.is_cuda: torch.cuda.synchronize()
                dec_end = time.time()
                x_hat = decompressed['x_hat'].float()[:, :, :orig_h, :orig_w] * scale + cmin
                recon_parts.append(x_hat[:, :actual_c])
                stream_bytes = bitstream_size(compressed['bit_stream'])
                total_bytes += stream_bytes
                total_enc += enc_end - enc_start
                total_dec += dec_end - enc_end
            recon = torch.cat(recon_parts, dim=1).squeeze(0).cpu().numpy()
            psnr_val, mse_val = calculate_psnr(raw, recon)
            original_bytes = raw.size * 4
            summary.append({
                'model_id': f'DCVC_RT_Intra_q{qp}', 'arch': 'DCVC-RT', 'metric': 'mse',
                'params': params, 'timestamp': ts, 'mse': mse_val, 'rmse': math.sqrt(mse_val),
                'psnr': psnr_val, 'bpp': total_bytes * 8.0 / (raw.shape[-2] * raw.shape[-1]),
                'bitstream_bytes': total_bytes, 'original_bytes': original_bytes,
                'compression_ratio': original_bytes / total_bytes if total_bytes > 0 else float('inf'),
                'encode_time_avg': total_enc, 'decode_time_avg': total_dec,
                'encode_throughput': original_bytes / total_enc if total_enc > 0 else None,
                'decode_throughput': original_bytes / total_dec if total_dec > 0 else None,
            })
    except Exception as exc:
        summary.append({'model_id': f'DCVC_RT_Intra_q{qp}', 'arch': 'DCVC-RT', 'metric': 'mse', 'error': str(exc)})
    with (output_dir / 'summary.json').open('w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary[-1], indent=2))

del model; gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
"
