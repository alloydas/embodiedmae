#!/bin/bash
#SBATCH --job-name=pcview_maize
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|l40s|h200|rtxpro6000
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Maize partial-cloud study on Nova (eval/pc_view_eval_maize.py + eval/pc_view_bootstrap.py),
# the Nova twin of slurm/delta/pc_view_maize.sbatch: conditions 1view / 3view / full (unions of
# the per-view clouds in camera 00's frame). Data: the v2 maize tree on Nova.
#   sbatch slurm/pc_view_maize_nova.sbatch maize_4m_pretrain_15k_d2:checkpoints/checkpoint_epoch_1000.pth
set -uo pipefail
cd /work/mech-ai-scratch/alloy/embodiedmae
source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
case "${GPU_NAME}" in
  *"RTX PRO 6000"*|*Blackwell*) conda activate /work/mech-ai-scratch/alloy/.conda/envs/det_cu128 ;;
  *)                            conda activate /work/mech-ai/alloy/.conda/envs/det ;;
esac
export PYTHONUNBUFFERED=1
ROOT="${MAIZE_ROOT:-/work/mech-ai-scratch/alloy/Maize_1}"
TAG="pcview_maize_nova_${SLURM_JOB_ID:-manual}"
echo "JobID ${SLURM_JOB_ID}  node $(hostname)  GPU ${GPU_NAME}  data ${ROOT}  runs: $*"
python eval/pc_view_eval_maize.py --runs "$@" --data-root "${ROOT}" \
  --num-workers "${SLURM_CPUS_PER_TASK:-16}" --batch-size 32 --out-prefix "reports/${TAG}"
LOKY_MAX_CPU_COUNT="${SLURM_CPUS_PER_TASK:-16}" python eval/pc_view_bootstrap.py --species maize --runs "$@" \
  --data-root "${ROOT}" --out "reports/${TAG}_bootstrap.csv"
echo "Done at $(date)  ->  reports/${TAG}_{coverage,recon,probe,bootstrap}.csv"
