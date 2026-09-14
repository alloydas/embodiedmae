#!/bin/bash
#SBATCH --job-name=smokeaccum
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:45:00
#SBATCH --output=logs/smokeaccum_%j.out
#SBATCH --error=logs/smokeaccum_%j.err
#
# Validates the gradient-accumulation path added to train_sorghum_4m_distill.py
# before it is trusted with a ~24h run.
#
# Why 2 GPUs and not 1: accumulation interacts with DDP. Non-final micro-batches
# run inside student.no_sync() so the all-reduce happens once per optimiser step
# rather than once per micro-batch, and DDP arms its reducer during the FORWARD
# pass -- so the forward has to be inside the context too. A single-GPU smoke
# never enters that code path and would prove nothing about it.
#
# Two arms, same config, same per-GPU batch, differing only in accum:
#   A: bs4 x 2 GPUs x accum1 -> global 8,  40 optimiser steps
#   B: bs4 x 2 GPUs x accum2 -> global 16, 20 optimiser steps
#
# What to check in the output:
#   1. Both arms complete without a DDP hang, desync, or bucket-rebuild error.
#   2. The header prints the intended GLOBAL BATCH and optimiser-steps/epoch.
#   3. Arm B does HALF the optimiser steps of arm A for the same micro-batches.
#   4. PEAKMEM is ~equal across arms: accumulation must not raise activation
#      memory, since each micro-batch's graph is freed by its own backward. If B
#      is materially higher, something is holding graphs alive and the 4-GPU plan
#      (which relies on bs16 fitting exactly as it does today) is off.
#   5. Losses are finite and on the same scale in both arms.
#
# bs4 (not the production 16) so this fits the smallest card it might land on:
# the config's own scaling is ~3.26 GiB/sample + ~2.2 GiB fixed, so bs4 is
# ~15 GiB and fits a 24 GB rtx_6000. This smoke tests mechanics, not capacity.
set -uo pipefail
cd /work/mech-ai-scratch/alloy/embodiedmae
mkdir -p logs

source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU" in
  *"PRO 6000"*) conda activate /work/mech-ai-scratch/alloy/.conda/envs/det_cu128 ;;
  *)            conda activate /work/mech-ai/alloy/.conda/envs/det ;;
esac

export PYTHONUNBUFFERED=1
export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$((10000 + RANDOM % 50000))
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN

echo "node $(hostname) | GPU ${GPU} | $(date)"
nvidia-smi -L || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cc', torch.cuda.get_device_capability(0))"

CONFIG=configs/config_4m_distill_15k_rgb2text.yaml

for ACCUM in 1 2; do
  echo ""
  echo "================ accum_steps=${ACCUM} ================"
  torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    train_sorghum_4m_distill.py \
      --config "${CONFIG}" \
      --batch_size 4 --accum_steps "${ACCUM}" \
      --num_workers 6 --epochs 1 --max_steps 40 \
      --val_max_batches 2 --no_wandb \
      --output_dir "./outputs/_smoke_accum${ACCUM}" \
    || echo "!!! accum=${ACCUM} FAILED"
done

echo ""
echo "Done at $(date)"
