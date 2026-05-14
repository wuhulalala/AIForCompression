#!/bin/bash
#SBATCH --job-name=uvg_dace_hpcm
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output=../logs/uvg_dace_hpcm_%j.log
#SBATCH --error=../logs/uvg_dace_hpcm_%j.log

eval "$(/data/home/scxj523/run/miniconda3/bin/conda shell.bash hook)"
conda activate /data/run01/scxj523/zsh/envs/zsh

cd /data/run01/scxj523/zsh/project/AIForCompression

python scripts/run_dataset_compression.py \
  --dataset uvg \
  --data_root /data/run01/scxj523/zsh/project/Data/UVG \
  --output_dir unified_results/uvg \
  --models DCAE LIC-HPCM \
  --max_samples 30
