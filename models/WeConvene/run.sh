#!/bin/bash
#SBATCH --job-name=weconvene_era5
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output=weconvene_era5_%j.log
#SBATCH --error=weconvene_era5_%j.log

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate /data/run01/scxj523/zsh/envs/zsh

cd /data/run01/scxj523/zsh/project/AIForCompression/WeConvene
python test_era5.py \
    --data_root /data/run01/scxj523/zsh/project/AIForCompression/Data/ERA5/2024 \
    --ckpt_dir /data/run01/scxj523/zsh/project/AIForCompression/checkpoints/weconvene \
    --gpu 0 \
    --compress
