#!/bin/bash
#SBATCH --job-name=pcfscore_maize
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|l40s|h200|rtxpro6000
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Point-cloud precision / recall / F1 of maize runs on Nova (eval/pc_fscore_maize.py; the Delta
# twin is slurm/delta/pc_fscore_maize.sbatch). Each run's data root comes from its config.json
# (maize final data on Nova: /work/mech-ai-scratch/alloy/Maize_1).
#   sbatch slurm/pc_fscore_maize_nova.sbatch maize_4m_pretrain_15k_d2:checkpoints/checkpoint_epoch_1000.pth
set -uo pipefail
cd /work/mech-ai-scratch/alloy/embodiedmae
source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
case "${GPU_NAME}" in
  *"RTX PRO 6000"*|*Blackwell*) conda activate /work/mech-ai-scratch/alloy/.conda/envs/det_cu128 ;;
  *)                            conda activate /work/mech-ai/alloy/.conda/envs/det ;;
esac
export PYTHONUNBUFFERED=1
echo "JobID ${SLURM_JOB_ID}  node $(hostname)  GPU ${GPU_NAME}  runs: $*"
python eval/pc_fscore_maize.py --runs "$@" --splits val test --num-workers "${SLURM_CPUS_PER_TASK:-8}" \
  --out "reports/pc_fscore_maize_nova_${SLURM_JOB_ID}.csv"
echo "Done at $(date)"
