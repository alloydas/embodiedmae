#!/bin/bash
#SBATCH --job-name=e10_real_rows
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
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# E10: the maize models on the rows built from the 520 real plants (eval/eval_real_rows.py, data eval/real_rows.py).
#   sbatch slurm/eval_real_rows.sbatch [extra args for eval/eval_real_rows.py]
set -uo pipefail
cd /work/mech-ai-scratch/alloy/embodiedmae
source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
case "${GPU_NAME}" in
  *"RTX PRO 6000"*|*Blackwell*) conda activate /work/mech-ai-scratch/alloy/.conda/envs/det_cu128 ;;
  *)                            conda activate /work/mech-ai/alloy/.conda/envs/det ;;
esac
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
echo "JobID ${SLURM_JOB_ID}  node $(hostname)  GPU ${GPU_NAME}  args: $*"
python -I eval/eval_real_rows.py --num-workers "${SLURM_CPUS_PER_TASK:-16}" --batch-size 16 --out "reports/e10_real_rows_${SLURM_JOB_ID}" "$@"
echo "Done at $(date)"
