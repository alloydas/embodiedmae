#!/bin/bash
#SBATCH --job-name=xspecies_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Cross-species Z1 (frozen probe) + Z2 (photograph->cloud) in both directions
# (PLAN_cross_species.md). Writes reports/xprobe_*_<job>.csv and reports/xquant_*_<job>.json.
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
J="${SLURM_JOB_ID:-manual}"
echo "JobID ${J}  node $(hostname)  GPU ${GPU_NAME}"
python eval/cross_species_probe.py --model-run e2_pcrgbd_d2 4m_pretrain_15k_d2_sink:checkpoints/checkpoint_epoch_1000.pth \
  --data-species maize --num-workers "${SLURM_CPUS_PER_TASK:-8}" --out reports/xprobe_sorghum_on_maize_${J}.csv
python eval/cross_species_probe.py --model-run maize_4m:checkpoints/checkpoint_epoch_600.pth \
  --data-species sorghum --num-workers "${SLURM_CPUS_PER_TASK:-8}" --out reports/xprobe_maize_on_sorghum_${J}.csv
python eval_rgb2pc_quant.py --checkpoint outputs/4m_pretrain_15k_d2_sink/checkpoints/checkpoint_epoch_1000.pth \
  --model_species sorghum --data_species maize --label xquant_sorghum_on_maize --split test --n_plants 2250 \
  --batch_size 8 --device cuda --out reports/xquant_sorghum_on_maize_${J}.json
python eval_rgb2pc_quant.py --checkpoint outputs/maize_4m/checkpoints/checkpoint_epoch_600.pth \
  --model_species maize --data_species sorghum --label xquant_maize_on_sorghum --split test --n_plants 2250 \
  --batch_size 8 --device cuda --out reports/xquant_maize_on_sorghum_${J}.json
echo "Done at $(date)"
