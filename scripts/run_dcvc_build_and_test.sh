#!/bin/bash
#SBATCH --job-name=dcvc_build
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/data/run01/scxj523/zsh/project/AIForCompression/logs/dcvc_build_%j.log
#SBATCH --error=/data/run01/scxj523/zsh/project/AIForCompression/logs/dcvc_build_%j.log

set -uo pipefail

export PATH="/data/run01/scxj523/zsh/envs/zsh/bin:$PATH"

# 1. 编译 C++ rans 扩展
echo "=== Building C++ rans extension ==="
cd /data/run01/scxj523/zsh/project/AIForCompression/models/DCVC/src/cpp
rm -rf build/ *.so
python3 setup.py build_ext --inplace 2>&1

# 2. 编译 CUDA inference 扩展
echo "=== Building CUDA inference extension ==="
cd /data/run01/scxj523/zsh/project/AIForCompression/models/DCVC/src/layers/extensions/inference
rm -rf build/ *.so
python3 setup.py build_ext --inplace 2>&1

# 3. 验证编译结果
echo "=== Verifying builds ==="
ls -la /data/run01/scxj523/zsh/project/AIForCompression/models/DCVC/src/cpp/*.so 2>/dev/null || echo "No .so in cpp/"
find /data/run01/scxj523/zsh/project/AIForCompression/models/DCVC -name "*.so" -type f 2>/dev/null

# 4. 运行 DCVC-RT 测试
echo "=== Running DCVC-RT test ==="
cd /data/run01/scxj523/zsh/project/AIForCompression
python -u scripts/test_video_intra_era5.py \
  --data_root "/data/run01/scxj523/zsh/project/Data/ERA5/2024" \
  --output_dir "/data/run01/scxj523/zsh/project/AIForCompression/logs/results/video_intra" \
  --model DCVC \
  --gpu 0 \
  --max_samples 1
