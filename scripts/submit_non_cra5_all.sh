#!/bin/bash
set -euo pipefail

ROOT=/data/run01/scxj523/zsh/project/AIForCompression

scripts=(
  "$ROOT/scripts/run_image_models_268.sh"
  "$ROOT/scripts/run_video_models_268.sh"
  "$ROOT/scripts/run_caesar_models_268.sh"
)

for script in "${scripts[@]}"; do
  echo "Submitting $script"
  sbatch -p gpu_5090 "$script"
done
