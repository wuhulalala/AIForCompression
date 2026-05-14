#!/bin/bash
# Usage: submit all tomo compression tests to Slurm
#   bash scripts/run_tomo_all.sh

SCRIPT_DIR=/data/run01/scxj523/zsh/project/AIForCompression/scripts

echo "Submitting image models..."
jid1=$(sbatch --parsable $SCRIPT_DIR/run_tomo_image_models.sh)
echo "  Image models job: $jid1"

echo "Submitting video models..."
jid2=$(sbatch --parsable $SCRIPT_DIR/run_tomo_video_models.sh)
echo "  Video models job: $jid2"

echo "Submitting CAESAR models..."
jid3=$(sbatch --parsable $SCRIPT_DIR/run_tomo_caesar.sh)
echo "  CAESAR models job: $jid3"

echo ""
echo "All jobs submitted: $jid1 $jid2 $jid3"
echo "Check status: squeue -u \$USER"
