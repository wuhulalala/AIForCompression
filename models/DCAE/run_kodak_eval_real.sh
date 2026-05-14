#!/bin/bash
#SBATCH --job-name=dcae_k_eval_real
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output=dcae_k_eval_real_%j.log
#SBATCH --error=dcae_k_eval_real_%j.log

set -eo pipefail

cd /data/run01/scxj523/zsh/project/AIForCompression/models/DCAE

/data/run01/scxj523/zsh/project/AIForCompression/.venvs/dcae126/bin/python eval.py \
  --checkpoint /data/run01/scxj523/zsh/project/AIForCompression/checkpoints/dcae/mse_0.0018.pth.tar \
  --data /data/run01/scxj523/zsh/project/Data/Kodac \
  --cuda \
  --real
