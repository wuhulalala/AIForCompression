#!/bin/bash
#SBATCH --job-name=isot1024_dace_hpcm
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output=../logs/isot1024_%j.log
#SBATCH --error=../logs/isot1024_%j.log

eval "$(/data/home/scxj523/run/miniconda3/bin/conda shell.bash hook)"
conda activate /data/run01/scxj523/zsh/envs/zsh

cd /data/run01/scxj523/zsh/project/AIForCompression

python scripts/run_dataset_compression.py \
  --dataset isot1024 \
  --data_root /data/run01/scxj523/zsh/project/Data/isot1024/isotropic1024-coarse-pressure.h5 \
  --output_dir unified_results/isot1024 \
  --models DCAE LIC-HPCM DCVC-RT DCMVC \
  --max_samples 30
