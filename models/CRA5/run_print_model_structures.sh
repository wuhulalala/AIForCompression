#!/bin/bash
#SBATCH --job-name=print_model_structures
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output=print_model_structures_%j.log
#SBATCH --error=print_model_structures_%j.log

set -ex
eval "$(/data/home/scxj523/run/miniconda3/bin/conda shell.bash hook)"
conda activate /data/run01/scxj523/zsh/envs/zsh

cd /data/run01/scxj523/zsh/project/AIForCompression/CRA5

which python
python --version
nvidia-smi

mkdir -p analysis_outputs

python -u print_model_structures.py \
  --output analysis_outputs/model_structures_${SLURM_JOB_ID}.txt
