#!/bin/bash
#SBATCH --job-name=lictcm_era5
#SBATCH --partition=gpu_5090
#SBATCH --gpus=2
#SBATCH --cpus-per-task=4
#SBATCH --output=lictcm_era5_%j.log
#SBATCH --error=lictcm_era5_%j.log

set -ex

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate /data/run01/scxj523/zsh/envs/zsh

echo "Hostname: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'no gpu')"
echo "Python: $(which python)"

cd /data/run01/scxj523/zsh/project/AIForCompression/LIC_TCM
python test_era5.py \
    --data_root /data/run01/scxj523/zsh/project/AIForCompression/Data/ERA5/2024 \
    --ckpt_dir /data/run01/scxj523/zsh/project/AIForCompression/checkpoints/lictcm \
    --gpu 0 \
    --compress
