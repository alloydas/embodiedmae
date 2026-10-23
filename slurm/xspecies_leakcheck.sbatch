#!/bin/bash
#SBATCH --job-name=xspecies_leak
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Leak check for the cross-species probe: the sorghum E1 reads maize plants with METRIC depth
# (re-encoded (z-0.0238)/50), while native maize models only ever see per-plant-normalised depth.
# (1) the same sorghum E1 on maize with depth HIDDEN; (2) the native maize_4m through the same
# harness (anchor). If (1) falls to the native level, metric depth carries the size signal.
set -euo pipefail
cd /work/mech-ai-scratch/alloy/embodiedmae
source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
case "${GPU_NAME}" in
  *"RTX PRO 6000"*|*Blackwell*) conda activate /work/mech-ai-scratch/alloy/.conda/envs/det_cu128 ;;
  *)                            conda activate /work/mech-ai/alloy/.conda/envs/det ;;
esac
export PYTHONUNBUFFERED=1; J="${SLURM_JOB_ID:-manual}"
python eval/cross_species_probe.py --model-run 4m_pretrain_15k_d2_sink:checkpoints/checkpoint_epoch_1000.pth \
  --data-species maize --drop depth --num-workers "${SLURM_CPUS_PER_TASK:-8}" --out reports/xprobe_sorghum_on_maize_nodepth_${J}.csv
python eval/cross_species_probe.py --model-run maize_4m:checkpoints/checkpoint_epoch_600.pth \
  --data-species maize --num-workers "${SLURM_CPUS_PER_TASK:-8}" --out reports/xprobe_maize_on_maize_anchor_${J}.csv
echo "Done at $(date)"
