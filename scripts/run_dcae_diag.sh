#!/bin/bash
#SBATCH --job-name=dcae_diag
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --export=NONE
#SBATCH --output=/data/run01/scxj523/zsh/project/AIForCompression/logs/dcae_diag_%j.log
#SBATCH --error=/data/run01/scxj523/zsh/project/AIForCompression/logs/dcae_diag_%j.log

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

from models import DCAE

device = 'cuda'
ckpt_path = str(ROOT / 'checkpoints' / 'dcae' / 'mse_0.0018.pth.tar')

# Load model
net = DCAE().to(device).eval()
ckpt = torch.load(ckpt_path, map_location=device)
sd = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
net.load_state_dict(sd)
net.update()
print("Model loaded and updated")

# Load one Kodak image
from PIL import Image
from torchvision import transforms
import os

kodak_dir = '/data/run01/scxj523/zsh/project/Data/Kodac'
img_files = sorted([f for f in os.listdir(kodak_dir) if f.endswith(('.png','.jpg','.jpeg','.peg'))])
img_path = os.path.join(kodak_dir, img_files[0])
img = transforms.ToTensor()(Image.open(img_path).convert('RGB')).to(device)
x = img.unsqueeze(0)
print(f"Image: {img_files[0]}, shape={x.shape}, range=[{x.min():.4f},{x.max():.4f}]")

# === 1. Forward pass ===
with torch.no_grad():
    out_fwd = net(x)
    x_hat_fwd = out_fwd['x_hat']
    mse_fwd = torch.mean((x - x_hat_fwd)**2).item()
    psnr_fwd = -10 * math.log10(mse_fwd) if mse_fwd > 0 else float('inf')
    print(f"\n[Forward] PSNR={psnr_fwd:.2f} dB, MSE={mse_fwd:.6f}")

# === 2. Compress/Decompress ===
with torch.no_grad():
    t0 = time.time()
    compressed = net.compress(x)
    torch.cuda.synchronize()
    t1 = time.time()
    decompressed = net.decompress(compressed['strings'], compressed['shape'])
    torch.cuda.synchronize()
    t2 = time.time()
    x_hat_cd = decompressed['x_hat']
    mse_cd = torch.mean((x - x_hat_cd)**2).item()
    psnr_cd = -10 * math.log10(mse_cd) if mse_cd > 0 else float('inf')
    print(f"[Compress/Decompress] PSNR={psnr_cd:.2f} dB, MSE={mse_cd:.6f}")
    print(f"  enc_time={t1-t0:.3f}s, dec_time={t2-t1:.3f}s")
    print(f"  fwd vs cd max diff: {(x_hat_fwd - x_hat_cd).abs().max():.6f}")

# === 3. Trace intermediate latent differences ===
with torch.no_grad():
    # Forward path latents
    y_fwd = net.g_a(x)
    z_fwd = net.h_a(y_fwd)
    z_offset = net.entropy_bottleneck._get_medians()
    z_hat_fwd = (z_fwd - z_offset).round() + z_offset

    # Latent params from forward
    ls_fwd = net.h_z_s1(z_hat_fwd)
    lm_fwd = net.h_z_s2(z_hat_fwd)

    # Compress path latents
    z_strings = net.entropy_bottleneck.compress(z_fwd)
    z_hat_cd = net.entropy_bottleneck.decompress(z_strings, z_fwd.size()[-2:])

    print(f"\n[Latent comparison]")
    print(f"  y_fwd: shape={y_fwd.shape}, range=[{y_fwd.min():.4f},{y_fwd.max():.4f}]")
    print(f"  z_fwd: shape={z_fwd.shape}, range=[{z_fwd.min():.4f},{z_fwd.max():.4f}]")
    print(f"  z_hat_fwd vs z_hat_cd max diff: {(z_hat_fwd - z_hat_cd).abs().max():.6f}")

    # Check hyper-decoder outputs
    ls_cd = net.h_z_s1(z_hat_cd)
    lm_cd = net.h_z_s2(z_hat_cd)
    print(f"  latent_scales fwd vs cd max diff: {(ls_fwd - ls_cd).abs().max():.6f}")
    print(f"  latent_means  fwd vs cd max diff: {(lm_fwd - lm_cd).abs().max():.6f}")

    # === 4. Detailed trace of slice-by-slice ===
    print(f"\n[Slice-by-slice trace]")
    y_shape = y_fwd.shape[2:]
    y_slices_fwd = y_fwd.chunk(net.num_slices, 1)
    y_hat_slices_fwd = []
    b = x.size(0)
    dt = net.dt.repeat([b, 1, 1])

    y_hat_slices_cd = []
    for si in range(net.num_slices):
        y_slice = y_slices_fwd[si]
        support_slices = y_hat_slices_fwd[:net.max_support_slices]
        query = torch.cat([ls_fwd] + [lm_fwd] + support_slices, dim=1)
        dict_info = net.dt_cross_attention[si](query, dt)
        support = torch.cat([query] + [dict_info], dim=1)

        mu = net.cc_mean_transforms[si](support)
        mu = mu[:, :, :y_shape[0], :y_shape[1]]
        scale = net.cc_scale_transforms[si](support)
        scale = scale[:, :, :y_shape[0], :y_shape[1]]

        # Forward: ste_round
        y_hat_slice_fwd = (y_slice - mu).round().detach() + mu
        lrp_support = torch.cat([support, y_hat_slice_fwd], dim=1)
        lrp = net.lrp_transforms[si](lrp_support)
        lrp = 0.5 * torch.tanh(lrp)
        y_hat_slice_fwd = y_hat_slice_fwd + lrp
        y_hat_slices_fwd.append(y_hat_slice_fwd)

        # Compress: quantize + dequantize
        y_q = net.gaussian_conditional.quantize(y_slice, "symbols", mu)
        y_hat_slice_cd = y_q + mu
        lrp_support_cd = torch.cat([support, y_hat_slice_cd], dim=1)
        lrp_cd = net.lrp_transforms[si](lrp_support_cd)
        lrp_cd = 0.5 * torch.tanh(lrp_cd)
        y_hat_slice_cd = y_hat_slice_cd + lrp_cd
        y_hat_slices_cd.append(y_hat_slice_cd)

        diff = (y_hat_slice_fwd - y_hat_slice_cd).abs()
        print(f"  slice {si}: mu range=[{mu.min():.4f},{mu.max():.4f}], scale range=[{scale.min():.4f},{scale.max():.4f}]")
        print(f"          y_slice range=[{y_slice.min():.4f},{y_slice.max():.4f}]")
        print(f"          y_q range=[{y_q.min():.2f},{y_q.max():.2f}], y_hat fwd vs cd max diff={diff.max():.6f}")

# === 5. Check g_s with both y_hat versions ===
with torch.no_grad():
    y_hat_fwd_full = torch.cat(y_hat_slices_fwd, dim=1)
    y_hat_cd_full = torch.cat(y_hat_slices_cd, dim=1)
    recon_fwd = net.g_s(y_hat_fwd_full)
    recon_cd = net.g_s(y_hat_cd_full)
    mse_fwd2 = torch.mean((x - recon_fwd)**2).item()
    mse_cd2 = torch.mean((x - recon_cd)**2).item()
    psnr_fwd2 = -10 * math.log10(mse_fwd2) if mse_fwd2 > 0 else float('inf')
    psnr_cd2 = -10 * math.log10(mse_cd2) if mse_cd2 > 0 else float('inf')
    print(f"\n[Manual reconstruction]")
    print(f"  Forward path: PSNR={psnr_fwd2:.2f} dB, MSE={mse_fwd2:.6f}")
    print(f"  Compress path (manual quantize): PSNR={psnr_cd2:.2f} dB, MSE={mse_cd2:.6f}")
    print(f"  y_hat forward vs compress max diff: {(y_hat_fwd_full - y_hat_cd_full).abs().max():.6f}")

    # Check if g_s output is the same with same input
    recon_cd2 = net.g_s(y_hat_fwd_full)
    diff_recon = (recon_fwd - recon_cd2).abs().max()
    print(f"  g_s determinism (same input): max diff={diff_recon:.10f}")

    # Compare: pipeline's decompress vs our manual recon
    diff_pipeline = (recon_cd - x_hat_cd).abs().max()
    actual_decompress_mse = torch.mean((x - x_hat_cd)**2).item()
    actual_decompress_psnr = -10 * math.log10(actual_decompress_mse) if actual_decompress_mse > 0 else float('inf')
    print(f"  Actual decompress x_hat: PSNR={actual_decompress_psnr:.2f} dB, MSE={actual_decompress_mse:.6f}")
    print(f"  manual_cd vs actual_decompress max diff: {diff_pipeline:.6f}")

# === 6. Check if rans encode/decode is the bottleneck ===
print(f"\n[RANS entropy coder check]")
with torch.no_grad():
    from compressai.ans import BufferedRansEncoder, RansDecoder
    cdf = net.gaussian_conditional.quantized_cdf.tolist()
    cdf_lengths = net.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
    offsets = net.gaussian_conditional.offset.reshape(-1).int().tolist()
    print(f"  CDF lengths: first10={cdf_lengths[:10]}, CDF table len={len(cdf)}")
    print(f"  Offsets: first10={offsets[:10]}")

# === 7. Check the forward() method's internal quantize path ===
print(f"\n[Forward() internals vs compress() internals]")
with torch.no_grad():
    # Replicate forward's internal logic step by step
    b = x.size(0)
    dt2 = net.dt.repeat([b, 1, 1])
    y2 = net.g_a(x)
    z2 = net.h_a(y2)
    _, z_likelihoods = net.entropy_bottleneck(z2)
    z_offset2 = net.entropy_bottleneck._get_medians()
    z_tmp2 = z2 - z_offset2
    z_hat2 = ste_round = (z_tmp2).round() + z_offset2

    latent_scales2 = net.h_z_s1(z_hat2)
    latent_means2 = net.h_z_s2(z_hat2)

    y_slices2 = y2.chunk(net.num_slices, 1)
    y_hat_slices2 = []

    for slice_index, y_slice in enumerate(y_slices2):
        support_slices = y_hat_slices2[:net.max_support_slices]
        query = torch.cat([latent_scales2] + [latent_means2] + support_slices, dim=1)
        dict_info2 = net.dt_cross_attention[slice_index](query, dt2)
        support2 = torch.cat([query] + [dict_info2], dim=1)

        mu2 = net.cc_mean_transforms[slice_index](support2)
        mu2 = mu2[:, :, :y2.shape[2], :y2.shape[3]]

        scale2 = net.cc_scale_transforms[slice_index](support2)
        scale2 = scale2[:, :, :y2.shape[2], :y2.shape[3]]

        # This is what forward does:
        y_hat_slice2 = (y_slice - mu2).round() + mu2  # ste_round
        # Actually forward uses: ste_round(y_slice - mu) + mu

        lrp_support2 = torch.cat([support2, y_hat_slice2], dim=1)
        lrp2 = net.lrp_transforms[slice_index](lrp_support2)
        lrp2 = 0.5 * torch.tanh(lrp2)
        y_hat_slice2 = y_hat_slice2 + lrp2
        y_hat_slices2.append(y_hat_slice2)

    y_hat2 = torch.cat(y_hat_slices2, dim=1)
    x_hat2 = net.g_s(y_hat2)
    mse_fwd_manual = torch.mean((x - x_hat2)**2).item()
    psnr_fwd_manual = -10 * math.log10(mse_fwd_manual) if mse_fwd_manual > 0 else float('inf')
    print(f"  Manual forward replica: PSNR={psnr_fwd_manual:.2f} dB (should match forward PSNR={psnr_fwd:.2f} dB)")

    # Now test: what if we use the compress-path quantization but manually?
    y_hat_slices_cd2 = []
    for slice_index, y_slice in enumerate(y_slices2):
        support_slices = y_hat_slices_cd2[:net.max_support_slices]
        query = torch.cat([latent_scales2] + [latent_means2] + support_slices, dim=1)
        dict_info2 = net.dt_cross_attention[slice_index](query, dt2)
        support2 = torch.cat([query] + [dict_info2], dim=1)

        mu2 = net.cc_mean_transforms[slice_index](support2)
        mu2 = mu2[:, :, :y2.shape[2], :y2.shape[3]]
        scale2 = net.cc_scale_transforms[slice_index](support2)
        scale2 = scale2[:, :, :y2.shape[2], :y2.shape[3]]

        # Use actual quantization (same as compress path)
        y_q2 = net.gaussian_conditional.quantize(y_slice, "symbols", mu2)
        y_hat_slice_cd2 = y_q2 + mu2

        lrp_support2 = torch.cat([support2, y_hat_slice_cd2], dim=1)
        lrp2 = net.lrp_transforms[slice_index](lrp_support2)
        lrp2 = 0.5 * torch.tanh(lrp2)
        y_hat_slice_cd2 = y_hat_slice_cd2 + lrp2
        y_hat_slices_cd2.append(y_hat_slice_cd2)

    y_hat_cd2 = torch.cat(y_hat_slices_cd2, dim=1)
    x_hat_cd2 = net.g_s(y_hat_cd2)
    mse_cd_manual = torch.mean((x - x_hat_cd2)**2).item()
    psnr_cd_manual = -10 * math.log10(mse_cd_manual) if mse_cd_manual > 0 else float('inf')
    print(f"  Manual quantize replica: PSNR={psnr_cd_manual:.2f} dB")

    # The key difference: ste_round vs real quantize
    for slice_index in range(min(2, net.num_slices)):
        y_slice = y_slices2[slice_index]
        # Get the mu for this slice
        support_slices = y_hat_slices2[:slice_index][:net.max_support_slices]
        query = torch.cat([latent_scales2] + [latent_means2] + support_slices, dim=1)
        dict_info2 = net.dt_cross_attention[slice_index](query, dt2)
        support2 = torch.cat([query] + [dict_info2], dim=1)
        mu2 = net.cc_mean_transforms[slice_index](support2)
        mu2 = mu2[:, :, :y2.shape[2], :y2.shape[3]]

        # ste_round
        ste_r = (y_slice - mu2).round() + mu2
        # real quantize
        real_q = net.gaussian_conditional.quantize(y_slice, "symbols", mu2) + mu2
        diff_q = (ste_r - real_q).abs()
        print(f"  slice {slice_index}: ste_round vs real_quantize max diff={diff_q.max():.4f}, mean diff={diff_q.mean():.4f}")
        print(f"          ste_round unique vals={len(ste_r.unique())}, real_q unique vals={len(real_q.unique())}")
PYEOF
