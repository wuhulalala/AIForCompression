#!/bin/bash
#SBATCH --job-name=dcae_k_bin
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output=dcae_k_bin_%j.log
#SBATCH --error=dcae_k_bin_%j.log

set -eo pipefail

ROOT=/data/run01/scxj523/zsh/project/AIForCompression/models/DCAE
PY=/data/run01/scxj523/zsh/project/AIForCompression/.venvs/dcae126/bin/python
DATA=/data/run01/scxj523/zsh/project/Data/Kodac
CKPT=/data/run01/scxj523/zsh/project/AIForCompression/checkpoints/dcae/mse_0.0018.pth.tar
OUT_BASE=/data/run01/scxj523/zsh/project/AIForCompression/models/DCAE/kodak_bin_roundtrip

mkdir -p "$OUT_BASE"

cd "$ROOT"

$PY compress_and_decompress.py \
  --cuda \
  --data "$DATA" \
  --save_path "$OUT_BASE" \
  --mode compress \
  --checkpoint "$CKPT"

$PY compress_and_decompress.py \
  --cuda \
  --data "$OUT_BASE/bin" \
  --save_path "$OUT_BASE/decompressed" \
  --mode decompress \
  --checkpoint "$CKPT"
