#!/bin/bash
#SBATCH --job-name=caesar_kodak
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --export=NONE
#SBATCH --output=/data/run01/scxj523/zsh/project/AIForCompression/logs/caesar_kodak_%j.log
#SBATCH --error=/data/run01/scxj523/zsh/project/AIForCompression/logs/caesar_kodak_%j.log

set -eo pipefail

ROOT=/data/run01/scxj523/zsh/project/AIForCompression
KODAK=/data/run01/scxj523/zsh/project/Data/Kodac
OUTDIR=$ROOT/unified_results/kodak_caesar_d

mkdir -p "$ROOT/logs" "$OUTDIR"

eval "$(/data/home/scxj523/run/miniconda3/bin/conda shell.bash hook)"
conda activate /data/run01/scxj523/zsh/envs/zsh
set -u

cd "$ROOT"

echo "host=$(hostname)"
echo "python=$(which python)"
nvidia-smi || true

python -u scripts/run_dataset_compression.py \
  --dataset kodak \
  --data_root "$KODAK" \
  --output_dir "$OUTDIR" \
  --models caesar_d \
  --max_samples -1 \
  --resolution 512 512 \
  --caesar_eb 3e-4 5e-4 8e-4 1e-3 1.5e-3 2e-3 3e-3 \
  --batch_size 1

echo "Done. Results in $OUTDIR"
