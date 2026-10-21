#!/bin/bash
#SBATCH --job-name=r2pquant_d2
#SBATCH --partition=scavenger
#SBATCH --account=mech-ai-scavenger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/r2pquant_%j.out
#SBATCH --error=logs/r2pquant_%j.err
#
# DATA VERSION 2 twin of eval_rgb2pc_test.sbatch (E1 Sinkhorn teacher + its d2 students; RGB served at each
# run's recorded gain via config.json).
# Quantitative RGB->PC generation eval on the HELD-OUT TEST split, with the val
# split run alongside so the val->test gap is measured rather than assumed.
# val is also the split best_model.pth was selected on, so it is optimistic by
# an unknown amount -- that amount is the point of this job.
#
# One seeded view of every one of the 2,250 test plants (and 2,250 val plants).
# Each report carries its own controls: `shuffled` (prediction vs a different
# plant) and `gt_vs_gt` (one plant's ground truth as another's prediction),
# which is the honest "predict the average sorghum" floor.
#
# Re-runnable: a report that already exists is skipped, so a scavenger requeue
# resumes instead of starting over.
set -uo pipefail
cd /work/mech-ai-scratch/alloy/embodiedmae

source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU" in
  *"PRO 6000"*) conda activate /work/mech-ai-scratch/alloy/.conda/envs/det_cu128 ;;
  *)            conda activate /work/mech-ai/alloy/.conda/envs/det ;;
esac
export PYTHONUNBUFFERED=1
echo "=========================================================="
echo "JobID   : ${SLURM_JOB_ID:-none}"
echo "Node    : $(hostname)"
echo "GPU     : ${GPU}"
echo "Date    : $(date)"
echo "=========================================================="

N_PLANTS=2250
OUTDIR=reports
mkdir -p "$OUTDIR"

# label | checkpoint | source modality
RUNS=(
  "teacher_undistilled_d2|outputs/4m_pretrain_15k_d2_sink/checkpoints/checkpoint_epoch_1000.pth|rgb"
  "smr00_full_d2|outputs/4m_distill_15k_rgb2pc_full_d2/best_model.pth|rgb"
  "smr50_seed1_d2|outputs/4m_distill_15k_rgb2pc_smr50_d2/best_model.pth|rgb"
  "smr50_seed2_d2|outputs/4m_distill_15k_rgb2pc_smr50b_d2/best_model.pth|rgb"
  "generalist_depth_d2|outputs/4m_distill_15k_all_d2/best_model.pth|depth"
  "generalist_rgb_d2|outputs/4m_distill_15k_all_d2/best_model.pth|rgb"
)

for SPLIT in test val; do
  for R in "${RUNS[@]}"; do
    IFS='|' read -r LABEL CKPT SRC <<< "$R"
    OUT="${OUTDIR}/quant_${LABEL}_${SPLIT}.json"
    if [ -f "$OUT" ]; then
      echo ">>> SKIP ${LABEL} / ${SPLIT} (already have ${OUT})"
      continue
    fi
    echo ""
    echo ">>> ${LABEL}  split=${SPLIT}  source=${SRC}"
    python eval_rgb2pc_quant.py \
      --checkpoint "$CKPT" \
      --label "$LABEL" \
      --split "$SPLIT" \
      --n_plants "$N_PLANTS" \
      --batch_size 4 \
      --source "$SRC" \
      --device cuda \
      --out "$OUT" || echo "!!! FAILED ${LABEL} / ${SPLIT} (continuing)"
  done
done

echo ""
echo "=========================================================="
echo "Reports written:"
ls -la "${OUTDIR}"/quant_*.json 2>/dev/null
echo "Done at $(date)"
echo "=========================================================="
