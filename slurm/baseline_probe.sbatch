#!/bin/bash
#SBATCH --job-name=e8_base
#SBATCH --partition=scavenger
#SBATCH --account=mech-ai-scavenger
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/probe_%x_%j.out
#SBATCH --error=logs/probe_%x_%j.err

# E8: the frozen baselines, scored by the 6.4 probe protocol (eval/baseline_probe.py).
#
#   sbatch --job-name=e8_base_sorghum slurm/baseline_probe.sbatch sorghum
#   sbatch --job-name=e8_base_maize   slurm/baseline_probe.sbatch maize
#
# Weights come from E8_WEIGHTS, prefetched once on a login node with network:
#   python eval/baseline_probe.py --prefetch --weights-dir /work/mech-ai-scratch/alloy/e8_weights
# The job runs HF_HUB_OFFLINE=1, so a missing weight fails loudly instead of downloading
# on a compute node.
#
# --num-workers 32 is NOT a tuning knob: each worker's numpy stream picks every plant's
# point subset, so it must equal the worker count the arm caches were extracted with
# (slurm/linear_probe.sbatch: --cpus-per-task=32 -> 32 workers), or the point-cloud rows
# are scored on different inputs from the arms they are compared with. The baseline
# caches record this and refuse a mismatch.
#
# Both poolings (cls, mean) come from one data pass. Rows are only comparable to arm
# rows of the same `feature`; the arms' mean rows come from
#   FEATURE=mean CKPT=checkpoints/checkpoint_epoch_600.pth sbatch slurm/linear_probe.sbatch ...
#
# Env is picked from the GPU actually granted, as in slurm/linear_probe.sbatch.

set -euo pipefail

SPECIES="${1:?usage: baseline_probe.sbatch <sorghum|maize> [baseline names...]}"
shift || true
BASELINES="${*:-random_init embodiedmae multimae dinov2_vitb14 pointmae pointmae_upright}"

cd /work/mech-ai-scratch/alloy/embodiedmae
mkdir -p logs reports
source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
case "${GPU_NAME}" in
  *"RTX PRO 6000"*|*Blackwell*) CONDA_ENV=/work/mech-ai-scratch/alloy/.conda/envs/det_cu128 ;;
  *)                            CONDA_ENV=/work/mech-ai/alloy/.conda/envs/det ;;
esac
conda activate "${CONDA_ENV}"

E8_WEIGHTS="${E8_WEIGHTS:-/work/mech-ai-scratch/alloy/e8_weights}"
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
OUT="reports/probe_${SLURM_JOB_NAME:-e8_base}_${SLURM_JOB_ID:-manual}.csv"

echo "JobID     : ${SLURM_JOB_ID:-none}"
echo "Node      : $(hostname)"
echo "GPU / env : ${GPU_NAME} / ${CONDA_ENV}"
echo "Species   : ${SPECIES}"
echo "Baselines : ${BASELINES}"
echo "Weights   : ${E8_WEIGHTS}"
echo "Date      : $(date)"

# Features are cached atomically per (baseline, split), so a preempted and requeued
# job resumes at the split it lost.
python eval/baseline_probe.py \
  --species "${SPECIES}" \
  --baselines ${BASELINES} \
  --weights-dir "${E8_WEIGHTS}" \
  --batch-size 32 --num-workers 32 \
  --out "${OUT}"

echo "Done at $(date)  ->  ${OUT}"
