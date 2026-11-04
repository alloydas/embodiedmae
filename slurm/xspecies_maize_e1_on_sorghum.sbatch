#!/bin/bash
#SBATCH --job-name=xsp_mE1_on_s
#SBATCH --partition=scavenger
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|l40s|h200|rtxpro6000
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Cross-species, maize HEADLINE model (maize E1 d2, epoch 1000) on SORGHUM data (PLAN_cross_species.md, 2026-11-04):
#   Z1 frozen probe: with depth / depth hidden; sorghum images as stored (x1.85) / at gain 1.0;
#      native anchors e2_pcrgbd_d2 and sorghum E1 are cache hits.
#   Z2 photograph -> cloud on 2,250 sorghum test plants: maize E1 and the maize RGB student (best_model).
# Feature caches resume after a preemption (outputs/_probe_cache_xspecies); quant reports are skipped if present.
set -euo pipefail
cd /work/mech-ai-scratch/alloy/embodiedmae
mkdir -p logs reports
source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
case "${GPU_NAME}" in
  *"RTX PRO 6000"*|*Blackwell*) conda activate /work/mech-ai-scratch/alloy/.conda/envs/det_cu128 ;;
  *)                            conda activate /work/mech-ai/alloy/.conda/envs/det ;;
esac
export PYTHONUNBUFFERED=1 WANDB__SERVICE_WAIT=600
J="${SLURM_JOB_ID:-manual}"; NW="${SLURM_CPUS_PER_TASK:-8}"
echo "JobID ${J}  node $(hostname)  GPU ${GPU_NAME}"
ME1=maize_4m_pretrain_15k_d2:checkpoints/checkpoint_epoch_1000.pth
python eval/cross_species_probe.py --model-run ${ME1} --data-species sorghum --num-workers ${NW} \
  --out reports/xprobe_maizeE1_on_sorghum_${J}.csv
python eval/cross_species_probe.py --model-run ${ME1} --data-species sorghum --num-workers ${NW} --drop depth \
  --out reports/xprobe_maizeE1_on_sorghum_nodepth_${J}.csv
python eval/cross_species_probe.py --model-run ${ME1} --data-species sorghum --num-workers ${NW} --rgb-gain 1.0 \
  --out reports/xprobe_maizeE1_on_sorghum_g1_${J}.csv
python eval/cross_species_probe.py --model-run e2_pcrgbd_d2 4m_pretrain_15k_d2_sink:checkpoints/checkpoint_epoch_1000.pth \
  --data-species sorghum --num-workers ${NW} --out reports/xprobe_sorghum_on_sorghum_anchor_${J}.csv
for spec in "maize_4m_pretrain_15k_d2/checkpoints/checkpoint_epoch_1000.pth:xquant_maizeE1_on_sorghum" \
            "maize_distill_rgb2pc_e1_d2/best_model.pth:xquant_maize_rgbstudent_on_sorghum"; do
  ck="outputs/${spec%%:*}"; lab="${spec##*:}"
  ls reports/${lab}_*.json >/dev/null 2>&1 && { echo "skip ${lab} (exists)"; continue; }
  python eval_rgb2pc_quant.py --checkpoint "${ck}" --model_species maize --data_species sorghum --label "${lab}" \
    --split test --n_plants 2250 --batch_size 8 --device cuda --out reports/${lab}_${J}.json
done
echo "Done at $(date)"
