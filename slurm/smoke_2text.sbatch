#!/bin/bash
#SBATCH --job-name=smoke2text
#SBATCH --partition=scavenger
#SBATCH --account=mech-ai-scavenger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=00:45:00
#SBATCH --output=logs/smoke2text_%j.out
#SBATCH --error=logs/smoke2text_%j.err
#
# Twenty steps of each ->text arm on one GPU, to answer one question before the
# full runs go out: with target [text], recon_term is the Smooth-L1 param loss
# scaled by spline_loss_weight (5.0), and it has to sit on a sane scale against
# feat_distill_weight 1.0 and cls_distill_weight 0.5. If recon is orders below
# feat, the student would optimise the teacher's features and ignore the
# parameters, and the weight needs raising before burning ~12 GPU-h per arm.
# Also confirms the val metric prints as text->param_mae, not a Chamfer.
set -uo pipefail
cd /work/mech-ai-scratch/alloy/embodiedmae
source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU" in *"PRO 6000"*) conda activate /work/mech-ai-scratch/alloy/.conda/envs/det_cu128 ;;
                *) conda activate /work/mech-ai/alloy/.conda/envs/det ;; esac
export PYTHONUNBUFFERED=1
echo "node $(hostname) | GPU ${GPU} | $(date)"

for SRC in pc rgb; do
  echo ""
  echo "================ ${SRC} -> text ================"
  python train_sorghum_4m_distill.py \
    --config configs/config_4m_distill_15k_${SRC}2text.yaml \
    --world_size 1 --batch_size 8 --num_workers 8 \
    --epochs 1 --max_steps 20 --val_max_batches 3 --no_wandb \
    --output_dir ./outputs/_smoke_${SRC}2text \
    || echo "!!! ${SRC}2text FAILED"
done
echo ""
echo "Done at $(date)"
