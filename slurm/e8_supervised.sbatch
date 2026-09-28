#!/bin/bash
#SBATCH --job-name=e8_supervised
#SBATCH --partition=scavenger
#SBATCH --account=mech-ai-scavenger
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=48
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# E8: the supervised-from-scratch baseline (train_supervised_4m.py), on Nova.
#
#   sbatch slurm/e8_supervised.sbatch
#
# Same shape as every reference arm: 16 per GPU x 2 GPUs = global 32, 197,400 steps.
# The trainer refuses any other shape (PointCloudEmbed's BatchNorm is per rank), so do
# not "speed it up" with more GPUs. A100 + det_cu128 is the placement the two
# token-budget control arms ran on.
#
# NEVER launch this through e2_arm_blackwell.sbatch / scale_arm_blackwell.sbatch /
# slurm/delta/train_arm.sbatch: they run train_sorghum_4m.py, and the E8 YAML makes that
# die at config parse on purpose.
#
# scavenger requeues on preemption and this script then resumes from the newest
# checkpoint that opens. A walltime TIMEOUT is NOT requeued: re-run the same sbatch
# line and it resumes (loss <= save_freq = 25 epochs).
#
# Score it afterwards at checkpoint_epoch_600.pth, in this order:
#   python train_supervised_4m.py --config configs/config_e8_supervised.yaml --score_head
#   CKPT=checkpoints/checkpoint_epoch_600.pth sbatch slurm/linear_probe.sbatch e8_supervised

set -euo pipefail

cd /work/mech-ai-scratch/alloy/embodiedmae
mkdir -p logs

source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
conda activate /work/mech-ai-scratch/alloy/.conda/envs/det_cu128

CONFIG=configs/config_e8_supervised.yaml
CKPT_DIR=./outputs/e8_supervised/checkpoints

echo "JobID   : ${SLURM_JOB_ID}"
echo "Node    : $(hostname)"
echo "Date    : $(date)"
nvidia-smi -L || true

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$((10000 + RANDOM % 50000))
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1

# Newest checkpoint that actually opens: a preemption can leave the last one truncated.
RESUME_ARG=""
if [ -d "${CKPT_DIR}" ]; then
  for f in $(ls "${CKPT_DIR}"/checkpoint_epoch_*.pth 2>/dev/null | sed 's/.*_\([0-9]*\)\.pth$/\1 &/' | sort -rn | cut -d' ' -f2); do
    if python -c "import torch,sys; torch.load(sys.argv[1], map_location='cpu', weights_only=False, mmap=True)" "$f" 2>/dev/null; then
      RESUME_ARG="--resume $f"; echo "Resuming from $f"; break
    fi
    echo "Skipping unreadable checkpoint $f"
  done
fi

# 48 CPUs: 2 ranks x 22 workers + main processes.
srun --kill-on-bad-exit=1 \
  torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    train_supervised_4m.py --config "${CONFIG}" --batch_size 16 --num_workers 22 ${RESUME_ARG} "$@"

echo "Done at $(date)"
