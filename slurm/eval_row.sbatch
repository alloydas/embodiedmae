#!/bin/bash
#SBATCH --job-name=eval_row
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Regime III evaluation (eval/eval_row.py): a checkpoint on the row data, row vs isolated,
# every input set. SPECIES env (default maize) picks the data root and recipe parser.
#
#   sbatch slurm/eval_row.sbatch maize_4m checkpoints/checkpoint_epoch_600.pth val
#   SPECIES=sorghum sbatch slurm/eval_row.sbatch e2_pcrgbdt_d2 checkpoints/checkpoint_epoch_600.pth val
#   sbatch slurm/eval_row.sbatch maize_row_ft best_model.pth test --n-plants 200

set -euo pipefail
SPECIES="${SPECIES:-maize}"
RUN="${1:?usage: [SPECIES=maize|sorghum] eval_row.sbatch <run> [ckpt] [split] [extra args]}"
CKPT="${2:-best_model.pth}"
SPLIT="${3:-val}"
shift 3 2>/dev/null || shift $#
cd /work/mech-ai-scratch/alloy/embodiedmae
mkdir -p logs reports
source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
case "${GPU_NAME}" in
  *"RTX PRO 6000"*|*Blackwell*) CONDA_ENV=/work/mech-ai-scratch/alloy/.conda/envs/det_cu128 ;;
  *)                            CONDA_ENV=/work/mech-ai/alloy/.conda/envs/det ;;
esac
conda activate "${CONDA_ENV}"
export PYTHONUNBUFFERED=1
TAG="${SPECIES}_${RUN}_$(basename "${CKPT}" .pth)_${SPLIT}_${SLURM_JOB_ID:-manual}"
echo "JobID ${SLURM_JOB_ID:-none}  node $(hostname)  GPU ${GPU_NAME}  run ${RUN} @ ${CKPT}  split ${SPLIT}"
python eval/eval_row.py --species "${SPECIES}" --run "${RUN}" --ckpt "${CKPT}" --split "${SPLIT}" \
  --num-workers "${SLURM_CPUS_PER_TASK:-8}" --out "reports/row_${TAG}" "$@"
echo "Done at $(date)  ->  reports/row_${TAG}.{csv,json}"
