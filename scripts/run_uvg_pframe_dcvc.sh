#!/bin/bash
#SBATCH --job-name=uvg_dcvc_p
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output=../logs/uvg_dcvc_p_%j.log
#SBATCH --error=../logs/uvg_dcvc_p_%j.log

eval "$(/data/home/scxj523/run/miniconda3/bin/conda shell.bash hook)"
conda activate /data/run01/scxj523/zsh/envs/zsh

cd /data/run01/scxj523/zsh/project/AIForCompression

python scripts/test_uvg_pframe.py \
  --model dcvc \
  --data_dir /data/run01/scxj523/zsh/project/Data/UVG_png/Twilight \
  --output_dir unified_results/uvg_dcvc_pframe \
  --max_frames 30
