#!/bin/bash
#SBATCH --job-name=s2c_dace_hpcm
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output=../logs/s2c_dace_hpcm_%j.log
#SBATCH --error=../logs/s2c_dace_hpcm_%j.log

eval "$(/data/home/scxj523/run/miniconda3/bin/conda shell.bash hook)"
conda activate /data/run01/scxj523/zsh/envs/zsh

cd /data/run01/scxj523/zsh/project/AIForCompression

python scripts/run_dataset_compression.py \
  --dataset s2c \
  --data_root /data/run01/scxj523/zsh/project/Data/s2c/S2C_MSIL2A_20260509T022531_N0512_R046_T51RUQ_20260509T055911.SAFE \
  --output_dir unified_results/s2c \
  --models DCAE LIC-HPCM \
  --tile_size 1024 \
  --max_samples -1
