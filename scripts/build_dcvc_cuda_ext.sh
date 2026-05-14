#!/bin/bash
#SBATCH --job-name=dcvc_cuda_ext
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/data/run01/scxj523/zsh/project/AIForCompression/logs/dcvc_cuda_ext_%j.log
#SBATCH --error=/data/run01/scxj523/zsh/project/AIForCompression/logs/dcvc_cuda_ext_%j.log

set -euo pipefail

export PATH="/data/run01/scxj523/zsh/envs/zsh/bin:$PATH"
# setup.py uses CONDA_PREFIX to locate nvidia CUDA headers
export CONDA_PREFIX=/data/run01/scxj523/zsh/envs/zsh

EXT_DIR=/data/run01/scxj523/zsh/project/AIForCompression/models/DCVC/src/layers/extensions/inference

echo "=== Building DCVC-RT CUDA inference extension (sm_120) ==="
cd "$EXT_DIR"
rm -rf build/ *.so
python setup.py build_ext --inplace 2>&1

echo "=== Verification ==="
ls -la "$EXT_DIR"/*.so 2>/dev/null && echo "BUILD SUCCESS" || echo "BUILD FAILED"
