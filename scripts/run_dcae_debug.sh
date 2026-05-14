#!/bin/bash
#SBATCH --job-name=dcae_debug
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --export=NONE
#SBATCH --output=/data/run01/scxj523/zsh/project/AIForCompression/logs/dcae_debug_%j.log
#SBATCH --error=/data/run01/scxj523/zsh/project/AIForCompression/logs/dcae_debug_%j.log

set -eo pipefail
eval "$(/data/home/scxj523/run/miniconda3/bin/conda shell.bash hook)"
conda activate /data/run01/scxj523/zsh/envs/zsh

python3 << 'PYEOF'
import sys, numpy as np, torch, time
from pathlib import Path
sys.path.insert(0, '/data/run01/scxj523/zsh/project/AIForCompression')
from compression_pipeline.views import build_image_groups, reconstruct_from_groups
from compression_pipeline.adapters.kodak import KodakAdapter
from compression_pipeline.torch_codecs import CompressAILikeCodec
from compression_pipeline.model_registry import load_dcae
from compression_pipeline.runner import run_image_grouped_sample

root = Path('/data/run01/scxj523/zsh/project/AIForCompression')
device = 'cuda'

adapter = KodakAdapter('/data/run01/scxj523/zsh/project/Data/Kodac')
sample = next(adapter.iter_samples(max_samples=1))
orig = sample.array.astype(np.float32)
print(f"Sample: {sample.sample_id}, shape={sample.array.shape}")
print(f"Device: {device}, GPU: {torch.cuda.get_device_name(0)}")

model = load_dcae(root, str(root/'checkpoints'/'dcae'/'mse_0.0018.pth.tar'), device)
codec = CompressAILikeCodec(model, device=device, divisor=128)

x = torch.from_numpy(sample.array[None].astype(np.float32) / 255.0).to(device)

# forward
with torch.no_grad():
    out = model(x)
    x_hat_fwd = out['x_hat'].detach().cpu().numpy()[0]
    mse_fwd = np.mean((orig/255.0 - x_hat_fwd)**2)
    psnr_fwd = 10 * np.log10(1.0 / mse_fwd)
    print(f"forward: PSNR={psnr_fwd:.2f} dB, MSE={mse_fwd:.6f} (in [0,1] space)")

# compress/decompress (raw, without codec wrapper)
with torch.no_grad():
    t0 = time.time()
    compressed = model.compress(x)
    torch.cuda.synchronize()
    t1 = time.time()
    decompressed = model.decompress(compressed['strings'], compressed['shape'])
    torch.cuda.synchronize()
    t2 = time.time()
    x_hat_cd = decompressed['x_hat'].detach().cpu().numpy()[0]
    mse_cd = np.mean((orig/255.0 - x_hat_cd)**2)
    psnr_cd = 10 * np.log10(1.0 / mse_cd) if mse_cd > 0 else float('inf')
    print(f"compress/decompress raw: PSNR={psnr_cd:.2f} dB, MSE={mse_cd:.6f} (in [0,1] space)")
    print(f"enc_time={t1-t0:.3f}s, dec_time={t2-t1:.3f}s")
    print(f"forward vs cd diff: max|fwd-cd|={np.abs(x_hat_fwd - x_hat_cd).max():.6f}")

# Through pipeline (uint8 roundtrip)
result = run_image_grouped_sample(sample, codec)
print(f"\nPipeline: PSNR={result['psnr']:.2f} dB, MSE={result['mse']:.1f}, BPP={result['bpp']:.4f}")

# Check reconstruction range
print(f"orig range: [{orig.min()}, {orig.max()}]")
print(f"fwd range: [{x_hat_fwd.min()*255:.2f}, {x_hat_fwd.max()*255:.2f}]")
print(f"cd  range: [{x_hat_cd.min()*255:.2f}, {x_hat_cd.max()*255:.2f}]")

# Debug: trace z_hat, y, mu differences between forward and compress
with torch.no_grad():
    b = x.size(0)

    # --- forward path ---
    y_fwd = model.g_a(x)
    z_fwd = model.h_a(y_fwd)
    z_offset = model.entropy_bottleneck._get_medians()
    z_hat_fwd = (z_fwd - z_offset).round() + z_offset
    print(f"\nz_fwd range: [{z_fwd.min():.4f}, {z_fwd.max():.4f}], z_hat_fwd range: [{z_hat_fwd.min():.4f}, {z_hat_fwd.max():.4f}]")

    # --- compress path ---
    z_strings = model.entropy_bottleneck.compress(z_fwd)
    z_hat_cd = model.entropy_bottleneck.decompress(z_strings, z_fwd.size()[-2:])
    print(f"z_hat_cd range: [{z_hat_cd.min():.4f}, {z_hat_cd.max():.4f}]")
    print(f"z_hat diff: max|fwd-cd| = {(z_hat_fwd - z_hat_cd).abs().max():.6f}")

    # Compare latent_scales and latent_means
    ls_fwd = model.h_z_s1(z_hat_fwd)
    lm_fwd = model.h_z_s2(z_hat_fwd)
    ls_cd = model.h_z_s1(z_hat_cd)
    lm_cd = model.h_z_s2(z_hat_cd)
    print(f"latent_scales diff: max|fwd-cd| = {(ls_fwd - ls_cd).abs().max():.6f}")
    print(f"latent_means diff: max|fwd-cd| = {(lm_fwd - lm_cd).abs().max():.6f}")
PYEOF
