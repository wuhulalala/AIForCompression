#!/bin/bash
#SBATCH --job-name=caesar_eb_sweep
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-6%2
#SBATCH --output=/data/run01/scxj523/zsh/project/AIForCompression/logs/caesar_eb_sweep_%A_%a.log
#SBATCH --error=/data/run01/scxj523/zsh/project/AIForCompression/logs/caesar_eb_sweep_%A_%a.log

set -eo pipefail

ROOT=/data/run01/scxj523/zsh/project/AIForCompression
DATA=/data/run01/scxj523/zsh/project/Data/ERA5/2024
LOG_ROOT="$ROOT/logs/results"

EB_VALUES=(3e-4 5e-4 8e-4 1e-3 1.5e-3 2e-3 3e-3)
EB="${EB_VALUES[$SLURM_ARRAY_TASK_ID]}"
EB_LABEL="${EB//./p}"
EB_LABEL="${EB_LABEL//-/m}"
OUT_DIR="$LOG_ROOT/CAESAR_eb_${EB_LABEL}"

mkdir -p "$ROOT/logs" "$OUT_DIR"

eval "$(/data/home/scxj523/run/miniconda3/bin/conda shell.bash hook)"
conda activate /data/run01/scxj523/zsh/envs/zsh
set -u

cd "$ROOT"
echo "Running CAESAR eb sweep task ${SLURM_ARRAY_TASK_ID}: eb=${EB}"
echo "Output dir: ${OUT_DIR}"

python -u scripts/test_caesar_era5.py \
  --data_root "$DATA" \
  --output_dir "$OUT_DIR" \
  --ckpt_dir "$ROOT/checkpoints/caesar" \
  --model both \
  --gpu 0 \
  --batch_size 1 \
  --max_channels 32 \
  --eb "$EB"
