#!/bin/bash
#SBATCH --job-name=pcview_eval
#SBATCH --partition=scavenger
#SBATCH --account=mech-ai-scavenger
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# 1-view vs 3-view point clouds, sorghum, evaluation only (eval/pc_view_eval.py).
# Needs the visibility masks from slurm/pc_view_masks.sbatch.
#
#   sbatch slurm/pc_view_eval.sbatch                        # default run set below
#   sbatch slurm/pc_view_eval.sbatch e2_pc e2_pcrgbd        # or name the runs
#
# One pass over each split's data scores every run under every condition, so the
# cost is mostly data loading: ~15,000 plants x (RGB + depth + cloud). One GPU of
# any type; the env is picked from the GPU granted, as in slurm/linear_probe.sbatch.
# Features cache per (run, ckpt, split) and are written atomically, so a preempted
# and requeued job resumes at the split it lost.

set -euo pipefail

cd /work/mech-ai-scratch/alloy/embodiedmae
mkdir -p logs reports

source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
# `|| true`: on a CPU-only node nvidia-smi is absent, and under set -e -o pipefail a
# failing command substitution would end the job here.
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
case "${GPU_NAME}" in
  *"RTX PRO 6000"*|*Blackwell*) CONDA_ENV=/work/mech-ai-scratch/alloy/.conda/envs/det_cu128 ;;
  *)                            CONDA_ENV=/work/mech-ai/alloy/.conda/envs/det ;;
esac
conda activate "${CONDA_ENV}"
export PYTHONUNBUFFERED=1

RUNS=("$@")
if [ "${#RUNS[@]}" -eq 0 ]; then
  RUNS=(e2_pc e2_pcrgb e2_pcrgbd e2_pcrgbdt
        4m_pretrain_15k_v2_depthfix_qal:checkpoints/checkpoint_epoch_1000.pth)
fi

echo "JobID : ${SLURM_JOB_ID:-none}"
echo "Node  : $(hostname)"
echo "GPU   : ${GPU_NAME}"
echo "Env   : ${CONDA_ENV}"
echo "Runs  : ${RUNS[*]}"
echo "Date  : $(date)"

python eval/pc_view_eval.py --runs "${RUNS[@]}" \
  --num-workers "${SLURM_CPUS_PER_TASK:-32}" --batch-size 32 \
  --out-prefix "reports/pcview_${SLURM_JOB_ID:-manual}"

echo "Done at $(date)"
