#!/bin/bash
#SBATCH --job-name=dcae_eval
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --export=NONE
#SBATCH --output=/data/run01/scxj523/zsh/project/AIForCompression/logs/dcae_eval_%j.log
#SBATCH --error=/data/run01/scxj523/zsh/project/AIForCompression/logs/dcae_eval_%j.log

set -eo pipefail

eval "$(/data/home/scxj523/run/miniconda3/bin/conda shell.bash hook)"
conda activate /data/run01/scxj523/zsh/envs/zsh

cd /data/run01/scxj523/zsh/project/AIForCompression/models/DCAE

python -u eval.py \
  --checkpoint /data/run01/scxj523/zsh/project/AIForCompression/checkpoints/dcae/mse_0.0018.pth.tar \
  --data /data/run01/scxj523/zsh/project/Data/Kodac \
  --cuda \
  --real
