#!/bin/bash
#SBATCH --job-name=e8_score
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

# E8's supervised-from-scratch row, scored at checkpoint_epoch_600.pth in the order
# train_supervised_4m.py's docstring requires:
#   1. --score_head: its own regression head, applied to the CLS features that
#      eval/linear_probe.py extracts. It writes the shared e8_supervised CLS cache
#      with the arm caches' loader shape (batch 32, 32 workers, seed 0), which is
#      why it goes FIRST: a probe run first would write that cache with its own
#      shape, and the cache name does not record the shape.
#   2. the ridge probe on the same CLS features (a cache hit).
#   3. the mean-pooled probe. E8 compares every model on mean pooling, the only
#      pooling all baselines have.
#
#   sbatch --dependency=afterany:<last e8_supervised training job> \
#          slurm/e8_supervised_score.sbatch
#
# One GPU of any type; the env is picked from the GPU granted, as in
# slurm/linear_probe.sbatch. Keep --cpus-per-task at 32: the worker count picks
# each plant's point subset, and every arm cache was extracted with 32 workers.
# scavenger bypasses the mech-ai GPU cap; a preempted job is requeued and resumes
# from the feature caches, which are written atomically.

set -euo pipefail

cd /work/mech-ai-scratch/alloy/embodiedmae
mkdir -p logs reports

RUN=e8_supervised
CKPT=checkpoints/checkpoint_epoch_600.pth
if [ ! -f "outputs/${RUN}/${CKPT}" ]; then
  echo "outputs/${RUN}/${CKPT} does not exist: training has not reached epoch 600." >&2
  echo "Resubmit this script once it has." >&2
  exit 3
fi

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

TAG="${SLURM_JOB_NAME:-e8_score}_${SLURM_JOB_ID:-manual}"
echo "JobID : ${SLURM_JOB_ID:-none}"
echo "Node  : $(hostname)"
echo "GPU   : ${GPU_NAME}"
echo "Env   : ${CONDA_ENV}"
echo "Ckpt  : outputs/${RUN}/${CKPT}"
echo "Date  : $(date)"

python train_supervised_4m.py --config configs/config_e8_supervised.yaml --score_head \
  --score_ckpt "${CKPT}" --score_out "reports/probe_${TAG}_head.csv"

python eval/linear_probe.py --runs "${RUN}" --ckpt "${CKPT}" \
  --batch-size 32 --num-workers 32 --out "reports/probe_${TAG}_cls.csv"

python eval/linear_probe.py --runs "${RUN}" --ckpt "${CKPT}" --feature mean \
  --batch-size 32 --num-workers 32 --out "reports/probe_${TAG}_mean.csv"

echo "Done at $(date)  ->  reports/probe_${TAG}_{head,cls,mean}.csv"
