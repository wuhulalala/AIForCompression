#!/bin/bash
#SBATCH --job-name=xray_dace_hpcm
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output=../logs/xray_dace_hpcm_%j.log
#SBATCH --error=../logs/xray_dace_hpcm_%j.log

eval "$(/data/home/scxj523/run/miniconda3/bin/conda shell.bash hook)"
conda activate /data/run01/scxj523/zsh/envs/zsh

cd /data/run01/scxj523/zsh/project/AIForCompression

python scripts/run_dataset_compression.py \
  --dataset shanghai_xray \
  --data_root "/data/run01/scxj523/zsh/project/Data/shanghai_xray/Shanghai Synchrotron Radiation Facility X-ray Small-Angle Scattering and Wide-Angle Diffraction Image Dataset" \
  --output_dir unified_results/shanghai_xray \
  --models DCAE LIC-HPCM \
  --max_samples 30
