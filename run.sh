#!/bin/bash
#SBATCH --job-name=dcae_era5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=dcae_era5_%j.log
#SBATCH --error=dcae_era5_%j.log

module load compilers/cuda/12.1 cudnn/8.8.1.3_cuda12.x compilers/gcc/9.3.0
source ~/.bashrc
conda activate zsh
cd /home/bingxing2/home/scx9kvs/zsh/backup/AIForCompression/DCAE
python test_era5.py \
    --data_root /home/bingxing2/home/scx9kvs/zsh/backup/data/ERA5/2024 \
    --gpu 0
