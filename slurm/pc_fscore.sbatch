#!/bin/bash
#SBATCH --job-name=pcfscore
#SBATCH --partition=scavenger
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Point-cloud precision / recall / F1 (eval/pc_fscore.py) of the sorghum runs given as arguments
# (slug or slug:ckpt), val + test. Writes reports/pcfscore_<job>_{items,summary}.csv.
#   sbatch slurm/pc_fscore.sbatch e2_pc_d2 e2_pcrgb_d2 ... 4m_pretrain_15k_d2_sink:checkpoints/checkpoint_epoch_1000.pth
set -euo pipefail
cd /work/mech-ai-scratch/alloy/embodiedmae
mkdir -p logs reports
source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
case "${GPU_NAME}" in
  *"RTX PRO 6000"*|*Blackwell*) conda activate /work/mech-ai-scratch/alloy/.conda/envs/det_cu128 ;;
  *)                            conda activate /work/mech-ai/alloy/.conda/envs/det ;;
esac
export PYTHONUNBUFFERED=1
echo "JobID ${SLURM_JOB_ID}  node $(hostname)  GPU ${GPU_NAME}  runs: $*"
python eval/pc_fscore.py --runs "$@" --splits val test --num-workers "${SLURM_CPUS_PER_TASK:-8}" \
  --out-prefix "reports/pcfscore_${SLURM_JOB_ID}"
echo "Done at $(date)"
