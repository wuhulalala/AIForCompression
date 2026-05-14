#!/bin/bash
#SBATCH --job-name=hurr_video
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output=../logs/hurr_video_%j.log
#SBATCH --error=../logs/hurr_video_%j.log

eval "$(/data/home/scxj523/run/miniconda3/bin/conda shell.bash hook)"
conda activate /data/run01/scxj523/zsh/envs/zsh

cd /data/run01/scxj523/zsh/project/AIForCompression

python scripts/run_dataset_compression.py \
  --dataset hurricane \
  --data_root /data/run01/scxj523/zsh/project/Data/hurricane/100x500x500 \
  --output_dir unified_results/hurricane_video \
  --models DCVC-RT DCMVC \
  --max_samples -1
