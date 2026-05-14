#!/bin/bash
#SBATCH --job-name=dcmvc_p_uvg
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/data/run01/scxj523/zsh/project/AIForCompression/logs/dcmvc_p_uvg_%j.log
#SBATCH --error=/data/run01/scxj523/zsh/project/AIForCompression/logs/dcmvc_p_uvg_%j.log

eval "$(/data/home/scxj523/run/miniconda3/bin/conda shell.bash hook)"
conda activate /data/run01/scxj523/zsh/envs/zsh

python /data/run01/scxj523/zsh/project/AIForCompression/scripts/run_dcmvc_pframe.py \
  --i_frame_model_path /data/run01/scxj523/zsh/project/AIForCompression/checkpoints/dcmvc/cvpr2023_image_psnr.pth.tar \
  --p_frame_model_path /data/run01/scxj523/zsh/project/AIForCompression/checkpoints/dcmvc/dcmvc_p_frame.pth.tar \
  --test_config /data/run01/scxj523/zsh/project/AIForCompression/scripts/uvg_dcmvc_pframe_config.json \
  --output_dir /data/run01/scxj523/zsh/project/AIForCompression/unified_results/uvg_dcmvc_pframe
