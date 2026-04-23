#!/bin/bash
#SBATCH --job-name=aifc_smoke
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --export=NONE
#SBATCH --output=/data/run01/scxj523/zsh/project/AIForCompression/logs/framework_smoke/%x_%j.log
#SBATCH --error=/data/run01/scxj523/zsh/project/AIForCompression/logs/framework_smoke/%x_%j.log

set -eo pipefail


ROOT=/data/run01/scxj523/zsh/project/AIForCompression
ERA5=/data/run01/scxj523/zsh/project/Data/ERA5/2024
KODAK=/data/run01/scxj523/zsh/project/Data/Kodac
CASE_NAME="${1:?usage: sbatch scripts/run_framework_smoke_model.sh <case> [max_model_jobs|all] [max_samples|all]}"
MAX_MODEL_JOBS="${2:-1}"
if [[ "$MAX_MODEL_JOBS" == "all" ]]; then
  MAX_MODEL_JOBS=-1
fi
MAX_SAMPLES="${3:-1}"
if [[ "$MAX_SAMPLES" == "all" ]]; then
  MAX_SAMPLES=-1
fi

mkdir -p "$ROOT/logs/framework_smoke" "$ROOT/unified_results/framework_smoke"

eval "$(/data/home/scxj523/run/miniconda3/bin/conda shell.bash hook)"
conda activate /data/run01/scxj523/zsh/envs/zsh
set -u

cd "$ROOT"
echo "case=${CASE_NAME}"
echo "max_model_jobs=${MAX_MODEL_JOBS}"
echo "max_samples=${MAX_SAMPLES}"
echo "host=$(hostname)"
echo "python=$(which python)"
nvidia-smi || true

common=(--project_root "$ROOT" --max_model_jobs "$MAX_MODEL_JOBS" --max_samples "$MAX_SAMPLES")

case "$CASE_NAME" in
  kodak_dcae)
    python -u scripts/run_dataset_compression.py --dataset kodak --data_root "$KODAK" --output_dir "$ROOT/unified_results/framework_smoke/kodak_dcae" --models DCAE "${common[@]}"
    ;;
  kodak_lictcm)
    python -u scripts/run_dataset_compression.py --dataset kodak --data_root "$KODAK" --output_dir "$ROOT/unified_results/framework_smoke/kodak_lictcm" --models LIC_TCM "${common[@]}"
    ;;
  kodak_hpcm)
    python -u scripts/run_dataset_compression.py --dataset kodak --data_root "$KODAK" --output_dir "$ROOT/unified_results/framework_smoke/kodak_hpcm" --models LIC-HPCM "${common[@]}"
    ;;
  kodak_rwkv)
    python -u scripts/run_dataset_compression.py --dataset kodak --data_root "$KODAK" --output_dir "$ROOT/unified_results/framework_smoke/kodak_rwkv" --models RwkvCompress "${common[@]}"
    ;;
  kodak_weconvene)
    python -u scripts/run_dataset_compression.py --dataset kodak --data_root "$KODAK" --output_dir "$ROOT/unified_results/framework_smoke/kodak_weconvene" --models WeConvene "${common[@]}"
    ;;
  era5_dcae)
    python -u scripts/run_dataset_compression.py --dataset era5 --data_root "$ERA5" --output_dir "$ROOT/unified_results/framework_smoke/era5_dcae" --models DCAE --max_channels 6 --resolution 128 128 "${common[@]}"
    ;;
  era5_lictcm)
    python -u scripts/run_dataset_compression.py --dataset era5 --data_root "$ERA5" --output_dir "$ROOT/unified_results/framework_smoke/era5_lictcm" --models LIC_TCM --max_channels 6 --resolution 128 128 "${common[@]}"
    ;;
  era5_hpcm)
    python -u scripts/run_dataset_compression.py --dataset era5 --data_root "$ERA5" --output_dir "$ROOT/unified_results/framework_smoke/era5_hpcm" --models LIC-HPCM --max_channels 6 --resolution 256 256 "${common[@]}"
    ;;
  era5_rwkv)
    python -u scripts/run_dataset_compression.py --dataset era5 --data_root "$ERA5" --output_dir "$ROOT/unified_results/framework_smoke/era5_rwkv" --models RwkvCompress --max_channels 6 --resolution 128 128 "${common[@]}"
    ;;
  era5_weconvene)
    python -u scripts/run_dataset_compression.py --dataset era5 --data_root "$ERA5" --output_dir "$ROOT/unified_results/framework_smoke/era5_weconvene" --models WeConvene --max_channels 6 --resolution 128 128 "${common[@]}"
    ;;
  era5_cra5)
    python -u scripts/run_dataset_compression.py --dataset era5 --data_root "$ERA5" --output_dir "$ROOT/unified_results/framework_smoke/era5_cra5" --models CRA5 --resolution 256 256 "${common[@]}"
    ;;
  era5_caesar_v)
    python -u scripts/run_dataset_compression.py --dataset era5 --data_root "$ERA5" --output_dir "$ROOT/unified_results/framework_smoke/era5_caesar_v" --models caesar_v "${common[@]}" --max_samples 8 --max_channels 6 --resolution 256 256 --batch_size 1 --caesar_eb 1e-4
    ;;
  era5_caesar_d)
    python -u scripts/run_dataset_compression.py --dataset era5 --data_root "$ERA5" --output_dir "$ROOT/unified_results/framework_smoke/era5_caesar_d" --models caesar_d "${common[@]}" --max_samples 16 --max_channels 6 --resolution 256 256 --batch_size 1 --caesar_eb 1e-4
    ;;
  *)
    echo "Unknown case: $CASE_NAME" >&2
    exit 2
    ;;
esac
