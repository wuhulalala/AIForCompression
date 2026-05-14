#!/bin/bash
#SBATCH --job-name=kodak_dcae_fwd
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/data/run01/scxj523/zsh/project/AIForCompression/logs/kodak_dcae_fwd_%j.log
#SBATCH --error=/data/run01/scxj523/zsh/project/AIForCompression/logs/kodak_dcae_fwd_%j.log

set -eo pipefail

ROOT=/data/run01/scxj523/zsh/project/AIForCompression
KODAK=/data/run01/scxj523/zsh/project/Data/Kodac
OUTDIR=$ROOT/unified_results/kodak_dcae_forward

mkdir -p "$ROOT/logs" "$OUTDIR"

eval "$(/data/home/scxj523/run/miniconda3/bin/conda shell.bash hook)"
conda activate /data/run01/scxj523/zsh/envs/zsh

cd "$ROOT"

python -u scripts/run_dataset_compression.py \
  --dataset kodak \
  --data_root "$KODAK" \
  --output_dir "$OUTDIR" \
  --models DCAE \
  --image_eval_mode forward \
  --max_samples -1
