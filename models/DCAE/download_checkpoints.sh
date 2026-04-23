#!/bin/bash
# Download DCAE pretrained checkpoints from Google Drive
# Run this script in an environment with access to Google Drive
# Usage: bash download_checkpoints.sh

set -e

CKPT_DIR="$(dirname "$0")/checkpoints"
mkdir -p "$CKPT_DIR"
cd "$CKPT_DIR"

echo "Downloading DCAE checkpoints to $CKPT_DIR ..."

# MSE checkpoints (6 lambdas)
gdown "1jCsRJq7Ttx22-yWQbEQAHJWbdIDtc30k" -O mse_0.05.pth.tar
gdown "1-6ZZ-bScGYj448h1sqMTX4w2Q75MQQ1q" -O mse_0.025.pth.tar
gdown "1kXfvxsljdN3EfXDGqzknFc2Ecsgf8qgS" -O mse_0.013.pth.tar
gdown "1LdycatKcGXHvFjoR-NE-GWnPlL9-BRWX" -O mse_0.0067.pth.tar
gdown "1JE0SO876a-btXzOQLTilj7D0vJdePlB4" -O mse_0.0035.pth.tar
gdown "1JzVuERiZe8cStgLnE5TJii_ssppgY1p-" -O mse_0.0018.pth.tar

# MS-SSIM checkpoints (6 lambdas)
gdown "1S81POfELTNyWmy2mMRlL70vuKjGC5QC_" -O msssim_60.5.pth.tar
gdown "1208tRiJw37ruKON1JVPj2YLp23l2O42U" -O msssim_31.73.pth.tar
gdown "1hzgbNPGxfXrwQN-OmTIujGBXCIXYsnW1" -O msssim_16.64.pth.tar
gdown "1U-1f24E6IrKjdHvzObnMslYBLA2YAKXe" -O msssim_8.73.pth.tar
gdown "1br6rf4WtwLY9NPvRCY3GyVP_lvrXRi9y" -O msssim_4.58.pth.tar
gdown "1BIgupje5UcEwzOch3QG1pooCxRxE6Rtt" -O msssim_2.40.pth.tar

echo "Done. Downloaded $(ls -1 *.pth.tar | wc -l) checkpoints."
ls -lh *.pth.tar
