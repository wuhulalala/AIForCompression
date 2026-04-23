#!/bin/bash
#SBATCH --job-name=dcae_era5
#SBATCH --partition=gpu_5090
#SBATCH --gpus=2
#SBATCH --cpus-per-task=4
#SBATCH --output=dcae_era5_%j.log
#SBATCH --error=dcae_era5_%j.log

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate /data/run01/scxj523/zsh/envs/zsh

cd /data/run01/scxj523/zsh/project/AIForCompression/DCAE
python test_era5.py \
    --data_root /data/run01/scxj523/zsh/project/AIForCompression/Data/ERA5/2024 \
    --ckpt_dir /data/run01/scxj523/zsh/project/AIForCompression/checkpoints/dcae \
    --gpu 0 \
    --compress
