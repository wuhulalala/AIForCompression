#!/bin/bash
#SBATCH --job-name=cra5_stk_kodak
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/data/run01/scxj523/zsh/project/AIForCompression/logs/cra5_stk_kodak_%j.log
#SBATCH --error=/data/run01/scxj523/zsh/project/AIForCompression/logs/cra5_stk_kodak_%j.log

eval "$(/data/home/scxj523/run/miniconda3/bin/conda shell.bash hook)"
conda activate /data/run01/scxj523/zsh/envs/zsh

cd /data/run01/scxj523/zsh/project/AIForCompression

python scripts/test_cra5_stacked.py \
  --dataset kodak \
  --output_dir unified_results/cra5_kodak_stacked
