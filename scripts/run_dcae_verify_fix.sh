#!/bin/bash
#SBATCH --job-name=dcae_vrfy
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --export=NONE
#SBATCH --output=/data/run01/scxj523/zsh/project/AIForCompression/logs/dcae_vrfy_%j.log
#SBATCH --error=/data/run01/scxj523/zsh/project/AIForCompression/logs/dcae_vrfy_%j.log

set -eo pipefail
eval "$(/data/home/scxj523/run/miniconda3/bin/conda shell.bash hook)"
conda activate /data/run01/scxj523/zsh/envs/zsh
set -u

cd /data/run01/scxj523/zsh/project/AIForCompression

python -u scripts/run_dataset_compression.py \
  --dataset kodak \
  --data_root /data/run01/scxj523/zsh/project/Data/Kodac \
  --output_dir /data/run01/scxj523/zsh/project/AIForCompression/unified_results/kodak_dcae_fixed \
  --models DCAE \
  --max_samples 3
