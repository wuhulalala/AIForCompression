#!/bin/bash
#SBATCH --job-name=dcae_fix
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --export=NONE
#SBATCH --output=/data/run01/scxj523/zsh/project/AIForCompression/logs/dcae_fix_%j.log
#SBATCH --error=/data/run01/scxj523/zsh/project/AIForCompression/logs/dcae_fix_%j.log

set -eo pipefail
eval "$(/data/home/scxj523/run/miniconda3/bin/conda shell.bash hook)"
conda activate /data/run01/scxj523/zsh/envs/zsh
set -u

python3 << 'PYEOF'
import torch, math, sys, os, time
from pathlib import Path

# === TEST 1: With cuDNN (default) ===
torch.backends.cudnn.enabled = True
print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
print(f"cuDNN deterministic: {torch.backends.cudnn.deterministic}")

ROOT = Path('/data/run01/scxj523/zsh/project/AIForCompression')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'models' / 'DCAE'))
from models import DCAE

device = 'cuda'
ckpt_path = str(ROOT / 'checkpoints' / 'dcae' / 'mse_0.0018.pth.tar')

net = DCAE().to(device).eval()
ckpt = torch.load(ckpt_path, map_location=device)
sd = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
net.load_state_dict(sd)
net.update()

from PIL import Image
from torchvision import transforms
kodak_dir = '/data/run01/scxj523/zsh/project/Data/Kodac'
img_files = sorted([f for f in os.listdir(kodak_dir) if f.endswith(('.png','.jpg','.jpeg','.peg'))])
img_path = os.path.join(kodak_dir, img_files[0])
img = transforms.ToTensor()(Image.open(img_path).convert('RGB')).to(device)
x = img.unsqueeze(0)

# Run compress/decompress 3 times
print("\n=== WITH cuDNN ===")
for run in range(3):
    with torch.no_grad():
        compressed = net.compress(x)
        decompressed = net.decompress(compressed['strings'], compressed['shape'])
    mse = torch.mean((x - decompressed['x_hat'])**2).item()
    psnr = -10 * math.log10(mse) if mse > 0 else float('inf')
    print(f"  run {run+1}: PSNR={psnr:.2f} dB")

# Test cuDNN determinism: call h_z_s1 twice with same input
print("\n[Determinism test]")
with torch.no_grad():
    y = net.g_a(x)
    z = net.h_a(y)
    offset = net.entropy_bottleneck._get_medians()
    z_hat = (z - offset).round() + offset

    out1 = net.h_z_s1(z_hat)
    out2 = net.h_z_s1(z_hat)
    print(f"  cuDNN enabled: h_z_s1(z_hat) twice, max diff = {(out1 - out2).abs().max():.10f}")

    out1m = net.h_z_s2(z_hat)
    out2m = net.h_z_s2(z_hat)
    print(f"  cuDNN enabled: h_z_s2(z_hat) twice, max diff = {(out1m - out2m).abs().max():.10f}")

# === TEST 2: Disable cuDNN ===
del net
torch.cuda.empty_cache()

torch.backends.cudnn.enabled = False
print(f"\ncuDNN enabled: {torch.backends.cudnn.enabled}")

net2 = DCAE().to(device).eval()
net2.load_state_dict(sd)
net2.update()

print("\n[Determinism test - cuDNN disabled]")
with torch.no_grad():
    y2 = net2.g_a(x)
    z2 = net2.h_a(y2)
    z_hat2 = (z2 - offset).round() + offset

    out1 = net2.h_z_s1(z_hat2)
    out2 = net2.h_z_s1(z_hat2)
    print(f"  h_z_s1(z_hat) twice, max diff = {(out1 - out2).abs().max():.10f}")

    out1m = net2.h_z_s2(z_hat2)
    out2m = net2.h_z_s2(z_hat2)
    print(f"  h_z_s2(z_hat) twice, max diff = {(out1m - out2m).abs().max():.10f}")

print("\n=== WITHOUT cuDNN ===")
for run in range(3):
    with torch.no_grad():
        compressed = net2.compress(x)
        decompressed = net2.decompress(compressed['strings'], compressed['shape'])
    mse = torch.mean((x - decompressed['x_hat'])**2).item()
    psnr = -10 * math.log10(mse) if mse > 0 else float('inf')
    print(f"  run {run+1}: PSNR={psnr:.2f} dB")

# Also test with deterministic=true
del net2
torch.cuda.empty_cache()

torch.backends.cudnn.enabled = True
torch.use_deterministic_algorithms(True)
print(f"\ncuDNN deterministic mode: {torch.backends.cudnn.deterministic}")

net3 = DCAE().to(device).eval()
net3.load_state_dict(sd)
net3.update()

print("\n=== WITH deterministic algorithms ===")
for run in range(3):
    with torch.no_grad():
        compressed = net3.compress(x)
        decompressed = net3.decompress(compressed['strings'], compressed['shape'])
    mse = torch.mean((x - decompressed['x_hat'])**2).item()
    psnr = -10 * math.log10(mse) if mse > 0 else float('inf')
    print(f"  run {run+1}: PSNR={psnr:.2f} dB")
PYEOF
