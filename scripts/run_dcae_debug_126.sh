#!/bin/bash
#SBATCH --job-name=dcae_dbg126
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --export=NONE
#SBATCH --output=/data/run01/scxj523/zsh/project/AIForCompression/logs/dcae_debug_126_%j.log
#SBATCH --error=/data/run01/scxj523/zsh/project/AIForCompression/logs/dcae_debug_126_%j.log

set -eo pipefail

/data/run01/scxj523/zsh/project/AIForCompression/.venvs/dcae126/bin/python << 'PYEOF'
import sys, numpy as np, torch, time
from pathlib import Path
sys.path.insert(0, '/data/run01/scxj523/zsh/project/AIForCompression')
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

import compressai
print(f"compressai={compressai.__version__}")
print(f"torch={torch.__version__}")

model = load_dcae(root, str(root/'checkpoints'/'dcae'/'mse_0.0018.pth.tar'), device)
codec = CompressAILikeCodec(model, device=device, divisor=128)

x = torch.from_numpy(sample.array[None].astype(np.float32) / 255.0).to(device)

with torch.no_grad():
    out = model(x)
    x_hat_fwd = out['x_hat'].detach().cpu().numpy()[0]
    mse_fwd = np.mean((orig/255.0 - x_hat_fwd)**2)
    psnr_fwd = 10 * np.log10(1.0 / mse_fwd)
    print(f"forward: PSNR={psnr_fwd:.2f} dB, MSE={mse_fwd:.6f}")

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
    print(f"compress/decompress raw: PSNR={psnr_cd:.2f} dB, MSE={mse_cd:.6f}")
    print(f"enc_time={t1-t0:.3f}s, dec_time={t2-t1:.3f}s")
    print(f"forward vs cd diff: max|fwd-cd|={np.abs(x_hat_fwd - x_hat_cd).max():.6f}")

result = run_image_grouped_sample(sample, codec)
print(f"Pipeline: PSNR={result['psnr']:.2f} dB, MSE={result['mse']:.1f}, BPP={result['bpp']:.4f}")
PYEOF
