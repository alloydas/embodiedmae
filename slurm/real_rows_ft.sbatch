#!/bin/bash
#SBATCH --job-name=maize_real_ft
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=(a100&NVLINK)|h200|rtxpro6000
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
# Fine-tune the maize row model on 75 % of the real maize plants (train_real_rows.py), then score the held-out
# 25 % with eval/eval_real_rows.py: the fine-tuned model and the row fine-tune it started from, on the same stops.
# usage: [SPECIES=sorghum] [SOURCE=rows2026] sbatch slurm/real_rows_ft.sbatch [extra train_real_rows.py args]
# SOURCE defaults to rows3000 (the 2026-11-13 upload); its runs are <species>_real_ft_r3000, never the first upload's dirs.
set -euo pipefail
cd /work/mech-ai-scratch/alloy/embodiedmae
source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
G=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1); echo "GPU: $G"
case "$G" in *"RTX PRO 6000"*) conda activate /work/mech-ai-scratch/alloy/.conda/envs/det_cu128;; *) conda activate /work/mech-ai/alloy/.conda/envs/det;; esac
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled
SPECIES=${SPECIES:-maize}                                   # SPECIES=sorghum for the 30 scanned sorghum plants
case $SPECIES in maize) BASE=maize_row_ft_e1:checkpoints/checkpoint_epoch_100.pth; TAG=e10_real_ft;;
                 sorghum) BASE=sorghum_row_ft:checkpoints/checkpoint_epoch_100.pth; TAG=e10_real_ft_sorghum;; esac
SOURCE=${SOURCE:-rows3000}                                  # SOURCE=rows2026: the first upload (split by row)
RUN=${SPECIES}_real_ft$([ "$SOURCE" = rows3000 ] && echo _r3000)${RUN_SUFFIX:-}
python train_real_rows.py --species $SPECIES --source $SOURCE --run $RUN --num-workers 14 "$@"
TEST=$(awk -F, '$2=="test"{print $1}' outputs/$RUN/split.csv | tr '\n' ' ')
if [ "$SOURCE" = rows3000 ]; then SEL="--source rows3000 --plants-file outputs/$RUN/split.csv"; else SEL="--source rows2026 --rows $TEST"; fi
python eval/eval_real_rows.py --species $SPECIES --num-workers 14 --batch-size 16 $SEL \
  --models $RUN:best_model.pth $BASE --out reports/${TAG}$([ "$SOURCE" = rows3000 ] && echo _r3000)${RUN_SUFFIX:-}_test_${SLURM_JOB_ID}
