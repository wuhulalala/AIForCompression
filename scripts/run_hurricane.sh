#!/bin/bash
#SBATCH --job-name=hurricane_dace_hpcm
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output=../logs/hurricane_dace_hpcm_%j.log
#SBATCH --error=../logs/hurricane_dace_hpcm_%j.log

eval "$(/data/home/scxj523/run/miniconda3/bin/conda shell.bash hook)"
conda activate /data/run01/scxj523/zsh/envs/zsh

cd /data/run01/scxj523/zsh/project/AIForCompression

python scripts/run_dataset_compression.py \
  --dataset hurricane \
  --data_root /data/run01/scxj523/zsh/project/Data/hurricane/100x500x500 \
  --output_dir unified_results/hurricane \
  --models DCAE LIC-HPCM \
  --max_samples -1
