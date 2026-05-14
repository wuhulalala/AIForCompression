#!/bin/bash
#SBATCH --job-name=dcae_diag3
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --export=NONE
#SBATCH --output=/data/run01/scxj523/zsh/project/AIForCompression/logs/dcae_diag3_%j.log
#SBATCH --error=/data/run01/scxj523/zsh/project/AIForCompression/logs/dcae_diag3_%j.log

set -eo pipefail
eval "$(/data/home/scxj523/run/miniconda3/bin/conda shell.bash hook)"
conda activate /data/run01/scxj523/zsh/envs/zsh
set -u

python3 << 'PYEOF'
import sys, numpy as np, torch, time, math
from pathlib import Path

ROOT = Path('/data/run01/scxj523/zsh/project/AIForCompression')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'models' / 'DCAE'))

from models import DCAE as DC
from compressai.ans import BufferedRansEncoder, RansDecoder

device = 'cuda'
ckpt_path = str(ROOT / 'checkpoints' / 'dcae' / 'mse_0.0018.pth.tar')

net = DC().to(device).eval()
ckpt = torch.load(ckpt_path, map_location=device)
sd = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
net.load_state_dict(sd)
net.update()

from PIL import Image
from torchvision import transforms
import os
kodak_dir = '/data/run01/scxj523/zsh/project/Data/Kodac'
img_files = sorted([f for f in os.listdir(kodak_dir) if f.endswith(('.png','.jpg','.jpeg','.peg'))])
img_path = os.path.join(kodak_dir, img_files[0])
img = transforms.ToTensor()(Image.open(img_path).convert('RGB')).to(device)
x = img.unsqueeze(0)
print(f"Image: {img_files[0]}")

# === Monkey-patch compress/decompress to save intermediate values ===
compress_state = {}
decompress_state = {}

orig_compress = net.compress
orig_decompress = net.decompress

def patched_compress(x):
    with torch.no_grad():
        b = x.size(0)
        dt = net.dt.repeat([b, 1, 1])
        y = net.g_a(x)
        y_shape = y.shape[2:]
        z = net.h_a(y)
        z_strings = net.entropy_bottleneck.compress(z)
        z_hat = net.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        latent_scales = net.h_z_s1(z_hat)
        latent_means = net.h_z_s2(z_hat)

        compress_state['z_hat'] = z_hat.clone()
        compress_state['z_strings'] = z_strings
        compress_state['z_shape'] = z.size()[-2:]
        compress_state['latent_scales'] = latent_scales.clone()
        compress_state['latent_means'] = latent_means.clone()
        compress_state['y_shape'] = y_shape
        compress_state['y'] = y.clone()

        y_slices = y.chunk(net.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []
        all_symbols = []
        all_indexes = []
        all_mus = []
        all_scales = []

        cdf = net.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = net.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = net.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if net.max_support_slices < 0 else y_hat_slices[:net.max_support_slices])
            query = torch.cat([latent_scales] + [latent_means] + support_slices, dim=1)
            dict_info = net.dt_cross_attention[slice_index](query, dt)
            support = torch.cat([query] + [dict_info], dim=1)
            mu = net.cc_mean_transforms[slice_index](support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]
            scale = net.cc_scale_transforms[slice_index](support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = net.gaussian_conditional.build_indexes(scale)
            y_q_slice = net.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            all_symbols.append(y_q_slice.clone())
            all_indexes.append(index.clone())
            all_mus.append(mu.clone())
            all_scales.append(scale.clone())

            symbols_flat = y_q_slice.reshape(-1).tolist()
            indexes_flat = index.reshape(-1).tolist()
            encoder.encode_with_indexes(symbols_flat, indexes_flat, cdf, cdf_lengths, offsets)

            lrp_support = torch.cat([support, y_hat_slice], dim=1)
            lrp = net.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice = y_hat_slice + lrp
            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        y_string = encoder.flush()
        y_strings = [y_string]

        compress_state['symbols'] = all_symbols
        compress_state['indexes'] = all_indexes
        compress_state['mus'] = all_mus
        compress_state['scales'] = all_scales
        compress_state['y_string'] = y_string
        compress_state['cdf'] = cdf
        compress_state['cdf_lengths'] = cdf_lengths
        compress_state['offsets'] = offsets

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

def patched_decompress(strings, shape):
    with torch.no_grad():
        z_hat = net.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = net.h_z_s1(z_hat)
        latent_means = net.h_z_s2(z_hat)
        b = z_hat.size(0)
        dt = net.dt.repeat([b, 1, 1])
        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        decompress_state['z_hat'] = z_hat.clone()
        decompress_state['latent_scales'] = latent_scales.clone()
        decompress_state['latent_means'] = latent_means.clone()
        decompress_state['y_shape_dec'] = y_shape

        # Compare z_hat with compress_state
        print(f"\n[State comparison]")
        for key in ['z_hat', 'latent_scales', 'latent_means']:
            c_val = compress_state[key]
            d_val = decompress_state[key]
            diff = (c_val - d_val).abs().max().item()
            print(f"  {key}: compress vs decompress max diff = {diff:.10f}")

        print(f"  z_strings identical: {strings[1] == compress_state['z_strings']}")
        print(f"  shape param: decompress={shape}, compress={compress_state['z_shape']}")

        y_string = strings[0][0]
        y_hat_slices = []
        all_rv = []
        all_mus_dec = []
        all_scales_dec = []

        cdf = net.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = net.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = net.gaussian_conditional.offset.reshape(-1).int().tolist()

        # Compare CDF with compress
        cdf_match = cdf == compress_state['cdf']
        clen_match = cdf_lengths == compress_state['cdf_lengths']
        coff_match = offsets == compress_state['offsets']
        print(f"  CDF match: {cdf_match}, lengths match: {clen_match}, offsets match: {coff_match}")

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(net.num_slices):
            support_slices = (y_hat_slices if net.max_support_slices < 0 else y_hat_slices[:net.max_support_slices])
            query = torch.cat([latent_scales] + [latent_means] + support_slices, dim=1)
            dict_info = net.dt_cross_attention[slice_index](query, dt)
            support = torch.cat([query] + [dict_info], dim=1)
            mu = net.cc_mean_transforms[slice_index](support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]
            scale = net.cc_scale_transforms[slice_index](support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            all_mus_dec.append(mu.clone())
            all_scales_dec.append(scale.clone())

            index = net.gaussian_conditional.build_indexes(scale)
            indexes_flat = index.reshape(-1).tolist()

            # Compare index with compress
            idx_diff = (index - compress_state['indexes'][slice_index]).abs().max().item()
            mu_diff = (mu - compress_state['mus'][slice_index]).abs().max().item()
            scale_diff = (scale - compress_state['scales'][slice_index]).abs().max().item()
            print(f"  slice {slice_index}: mu diff={mu_diff:.10f}, scale diff={scale_diff:.10f}, index diff={idx_diff:.10f}")

            rv = decoder.decode_stream(indexes_flat, cdf, cdf_lengths, offsets)

            # Compare rv with original symbols
            orig_syms = compress_state['symbols'][slice_index].reshape(-1).tolist()
            mismatches = [(i, orig_syms[i], rv[i]) for i in range(len(rv)) if orig_syms[i] != rv[i]]
            if mismatches:
                print(f"    *** SYMBOLS DIFFER! {len(mismatches)}/{len(rv)} ***")
                print(f"    First 10: {mismatches[:10]}")
            else:
                print(f"    symbols match")

            rv_t = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            all_rv.append(rv_t.clone())
            y_hat_slice = net.gaussian_conditional.dequantize(rv_t, mu)

            lrp_support = torch.cat([support, y_hat_slice], dim=1)
            lrp = net.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice = y_hat_slice + lrp
            y_hat_slices.append(y_hat_slice)

            decompress_state[f'rv_{slice_index}'] = rv_t
            decompress_state[f'mu_{slice_index}'] = mu
            decompress_state[f'scale_{slice_index}'] = scale

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = net.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

net.compress = patched_compress
net.decompress = patched_decompress

# Run
with torch.no_grad():
    compressed = net.compress(x)
    torch.cuda.synchronize()
    decompressed = net.decompress(compressed['strings'], compressed['shape'])
    torch.cuda.synchronize()

mse = torch.mean((x - decompressed['x_hat'])**2).item()
psnr = -10 * math.log10(mse) if mse > 0 else float('inf')
print(f"\nResult: PSNR={psnr:.2f} dB")

# Check: if the issue is in y_shape calculation
print(f"\n[y_shape check]")
print(f"  compress y_shape (from g_a output): {compress_state['y_shape']}")
print(f"  decompress y_shape (4 * z_shape): {decompress_state['y_shape_dec']}")
print(f"  match: {list(compress_state['y_shape']) == decompress_state['y_shape_dec']}")
PYEOF
