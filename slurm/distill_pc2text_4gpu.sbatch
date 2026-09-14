#!/bin/bash
# PC -> SPLINE PARAMETERS specialist distillation on the FULL Sorghum_15K.
# Config: configs/config_4m_distill_15k_pc2text.yaml   (target: [text] only, source_mask_ratio 0.5)
#
# Recovers the plant's procedural generation parameters from the point cloud alone.
# `mean_gen` in this run's history IS param_mae_masked and nothing else, so the
# headline number is the metric with no Chamfer mixed in.
#
# 4 GPUs, not the 8 the sibling arms used, because the two rtx_pro_6000 nodes are
# the most contested resource on the cluster and an 8-GPU ask sat PENDING with a
# backfill estimate of 2026-09-15 (jobs 12247381/12247382, submitted 2026-09-08).
# Global batch is held at 128 by accumulating 2 micro-batches per optimiser step
# (accum_steps: 2 in the config), so this is the SAME experiment as the 8-GPU
# recipe: same 820 optimiser steps/epoch, same lr, and batch_size stays 16 so the
# non-synced BatchNorm1d in PointCloudEmbed sees the same 16-sample micro-batches.
#
# partition=scavenger: waits for genuinely free cards rather than preempting
# anyone. PreemptMode=REQUEUE there, which this script already handles --
# --requeue plus --open-mode=append plus the auto-resume block below, which
# restarts from the highest-epoch checkpoint.
#
# 1-day walltime, deliberately shorter than the ~24-27h this needs. Half the GPUs
# at the same global batch is ~2x the 12h the 8-GPU arms took, so 100 epochs will
# very likely NOT finish inside 24h -- that is accepted. A short walltime is what
# makes the job backfillable ahead of the higher-priority rtx_pro_6000 reservation
# sitting at 2026-09-13 07:42, and getting on the cards days earlier is worth more
# than finishing in one submission. On TIMEOUT the auto-resume block below restarts
# from the highest-epoch checkpoint, so a continuation run loses at most save_freq=5
# epochs. Note TIMEOUT does not trigger --requeue: the continuation is a manual
# resubmit of this same script.
#
# Loss balance was checked before submitting (job 12247376, 20 steps each):
# recon 0.047 against feat 0.040-0.049, i.e. the parameter objective sits level
# with the feature-distillation term rather than being drowned by it, so
# spline_loss_weight stays at the teacher's own 5.0.
#SBATCH --job-name=pc2text4g
#SBATCH --partition=scavenger
#SBATCH --account=mech-ai-scavenger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_pro_6000:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
# Append on requeue: SLURM truncates by default, which erased the epoch 1-3
# history when this job was first preempted.
#SBATCH --open-mode=append
#SBATCH --output=logs/pc2text4g_%j.out
#SBATCH --error=logs/pc2text4g_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=alloydas@iastate.edu
#
# --mem=100G, not the 600G every prior arm asked for. That 600G was never close
# to used, and it was the only thing keeping this job off nova26-gpu-1, which has
# 4 idle rtx_pro_6000 but only ~104G of free allocatable RAM.
# Measured, not guessed:
#   job 12248284 (2 ranks x 6 workers = 14 procs)   MaxRSS  14.3G  -> ~1.0G/proc
#   job 12209228 (8 ranks x 14 workers = 120 procs) MaxRSS  73.0G  -> ~0.6G/proc
# This job is 4 ranks x 14 workers = 60 procs, so ~40-64G plus ~3.6G of prefetch
# buffers. (job 12209162 recorded 332G for the same shape as 12209228 -- that is
# page cache charged to the cgroup, which is reclaimed under pressure, not OOM.)
# If this OOMs it exits FAILED rather than requeueing, losing at most save_freq=5
# epochs; resubmit at 300G in that case.

set -euo pipefail

cd /work/mech-ai-scratch/alloy/embodiedmae
mkdir -p logs

source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
conda activate /work/mech-ai-scratch/alloy/.conda/envs/det_cu128

CONFIG=configs/config_4m_distill_15k_pc2text_4gpu.yaml
CKPT_DIR=./outputs/4m_distill_15k_pc2text/checkpoints

echo "JobID         : ${SLURM_JOB_ID}"
echo "Node          : $(hostname)"
echo "Date          : $(date)"
echo "GPUs visible  : ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count(), 'cc', torch.cuda.get_device_capability(0))"

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$((10000 + RANDOM % 50000))
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN
# Unbuffered: a preempted job otherwise logs nothing at all.
export PYTHONUNBUFFERED=1

# SLURM_GPUS_ON_NODE is a COUNT (e.g. "8"), not a list — use it directly.
if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
  NPROC=${SLURM_GPUS_ON_NODE}
else
  NPROC=$(echo "${CUDA_VISIBLE_DEVICES//,/ }" | wc -w)
fi
[ "${NPROC}" -gt 0 ] 2>/dev/null || NPROC=8
echo "Launching torchrun with ${NPROC} processes"

# Auto-resume from the highest-epoch checkpoint (survives requeue/preemption).
RESUME_ARG=""
if [ -d "${CKPT_DIR}" ]; then
  LATEST=$(for f in "${CKPT_DIR}"/checkpoint_epoch_*.pth; do
             [ -e "$f" ] || continue
             b=$(basename "$f" .pth); echo "${b##*_} $f"
           done | sort -n | tail -1 | cut -d' ' -f2-)
  if [ -n "${LATEST:-}" ]; then
    echo "Resuming from latest checkpoint: ${LATEST}"
    RESUME_ARG="--resume ${LATEST}"
  fi
fi

srun --kill-on-bad-exit=1 \
  torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC} \
    train_sorghum_4m_distill.py --config "${CONFIG}" ${RESUME_ARG}

echo "Done at $(date)"
