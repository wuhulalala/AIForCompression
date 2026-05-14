#!/bin/bash
#SBATCH --job-name=dcae_diag2
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --export=NONE
#SBATCH --output=/data/run01/scxj523/zsh/project/AIForCompression/logs/dcae_diag2_%j.log
#SBATCH --error=/data/run01/scxj523/zsh/project/AIForCompression/logs/dcae_diag2_%j.log

set -eo pipefail
eval "$(/data/home/scxj523/run/miniconda3/bin/conda shell.bash hook)"
conda activate /data/run01/scxj523/zsh/envs/zsh
set -u

python3 << 'PYEOF'
import sys, numpy as np, torch, time, math, copy
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

# Forward as reference
with torch.no_grad():
    out_fwd = net(x)
    psnr_fwd = -10 * math.log10(torch.mean((x - out_fwd['x_hat'])**2).item())
    print(f"Forward PSNR: {psnr_fwd:.2f} dB")

# === MIMIC compress() step by step ===
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

    y_slices = y.chunk(net.num_slices, 1)
    y_hat_slices = []
    y_scales = []
    y_means = []

    cdf = net.gaussian_conditional.quantized_cdf.tolist()
    cdf_lengths = net.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
    offsets = net.gaussian_conditional.offset.reshape(-1).int().tolist()

    print(f"CDF table: len={len(cdf)}, cdf_lengths[:5]={cdf_lengths[:5]}, offsets[:5]={offsets[:5]}")
    print(f"total cdf_lengths={len(cdf_lengths)}, total offsets={len(offsets)}")

    encoder = BufferedRansEncoder()
    symbols_list_orig = []  # Store original symbols for comparison
    indexes_list_orig = []

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

        symbols_flat = y_q_slice.reshape(-1).tolist()
        indexes_flat = index.reshape(-1).tolist()
        symbols_list_orig.append(symbols_flat)
        indexes_list_orig.append(indexes_flat)

        encoder.encode_with_indexes(symbols_flat, indexes_flat, cdf, cdf_lengths, offsets)

        lrp_support = torch.cat([support, y_hat_slice], dim=1)
        lrp = net.lrp_transforms[slice_index](lrp_support)
        lrp = 0.5 * torch.tanh(lrp)
        y_hat_slice = y_hat_slice + lrp
        y_hat_slices.append(y_hat_slice)
        y_scales.append(scale)
        y_means.append(mu)

    y_string = encoder.flush()

    # === DECODE step by step ===
    decoder = RansDecoder()
    decoder.set_stream(y_string)

    y_hat_slices_dec = []
    for slice_index in range(net.num_slices):
        support_slices = (y_hat_slices_dec if net.max_support_slices < 0 else y_hat_slices_dec[:net.max_support_slices])
        query = torch.cat([latent_scales] + [latent_means] + support_slices, dim=1)
        dict_info = net.dt_cross_attention[slice_index](query, dt)
        support = torch.cat([query] + [dict_info], dim=1)
        mu = net.cc_mean_transforms[slice_index](support)
        mu = mu[:, :, :y_shape[0], :y_shape[1]]
        scale = net.cc_scale_transforms[slice_index](support)
        scale = scale[:, :, :y_shape[0], :y_shape[1]]

        index = net.gaussian_conditional.build_indexes(scale)
        indexes_flat = index.reshape(-1).tolist()

        # Compare indexes between encode and decode
        idx_diff = sum(1 for a, b in zip(indexes_flat, indexes_list_orig[slice_index]) if a != b)
        if idx_diff > 0:
            print(f"  slice {slice_index}: INDEXES DIFFER! {idx_diff}/{len(indexes_flat)} entries differ")

        rv = decoder.decode_stream(indexes_flat, cdf, cdf_lengths, offsets)

        # Compare decoded symbols with original
        orig_syms = symbols_list_orig[slice_index]
        sym_diff = sum(1 for a, b in zip(rv, orig_syms) if a != b)
        print(f"  slice {slice_index}: orig symbols range=[{min(orig_syms)},{max(orig_syms)}], decoded range=[{min(rv)},{max(rv)}]")
        if sym_diff > 0:
            print(f"    ** SYMBOLS DIFFER! {sym_diff}/{len(rv)} entries differ! **")
            # Show first few differences
            mismatches = [(i, orig_syms[i], rv[i]) for i in range(len(rv)) if orig_syms[i] != rv[i]]
            print(f"    First mismatches: {mismatches[:10]}")
        else:
            print(f"    Symbols match perfectly (0/{len(rv)} differences)")

        rv_t = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
        y_hat_slice = net.gaussian_conditional.dequantize(rv_t, mu)

        lrp_support = torch.cat([support, y_hat_slice], dim=1)
        lrp = net.lrp_transforms[slice_index](lrp_support)
        lrp = 0.5 * torch.tanh(lrp)
        y_hat_slice = y_hat_slice + lrp
        y_hat_slices_dec.append(y_hat_slice)

    y_hat_dec = torch.cat(y_hat_slices_dec, dim=1)
    x_hat_manual = net.g_s(y_hat_dec).clamp_(0, 1)
    mse_manual = torch.mean((x - x_hat_manual)**2).item()
    psnr_manual = -10 * math.log10(mse_manual) if mse_manual > 0 else float('inf')
    print(f"\nManual RANS roundtrip: PSNR={psnr_manual:.2f} dB, MSE={mse_manual:.6f}")

    # === Compare with actual model.compress/decompress ===
    compressed = net.compress(x)
    decompressed = net.decompress(compressed['strings'], compressed['shape'])
    x_hat_actual = decompressed['x_hat']
    mse_actual = torch.mean((x - x_hat_actual)**2).item()
    psnr_actual = -10 * math.log10(mse_actual) if mse_actual > 0 else float('inf')
    print(f"Actual model.compress/decompress: PSNR={psnr_actual:.2f} dB, MSE={mse_actual:.6f}")
    print(f"manual vs actual max diff: {(x_hat_manual - x_hat_actual).abs().max():.6f}")

    # Check y_string sizes
    actual_y_string = compressed['strings'][0][0]
    print(f"\ny_string sizes: manual={len(y_string)}, actual={len(actual_y_string)}")
    print(f"z_strings sizes: manual={len(z_strings[0])}, actual={len(compressed['strings'][1][0])}")

    # Check if y_strings match
    print(f"y_string identical: {y_string == actual_y_string}")

PYEOF
