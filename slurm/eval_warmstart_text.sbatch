#!/bin/bash
# EPOCH-0 BASELINE for the two ->params distillation arms.
#
# Both arms' first validation point is already one epoch into training, and on the
# rgb2pc specialist runs a single epoch captured most of the total gain -- so the
# run's own ep1 number understates what distillation bought. This measures the
# undistilled warm start (teacher_final.pth, the v2 pretrain best at epoch 960)
# on exactly the same val subset the runs validated on:
#   * val_max_batches 100 x batch_size 16 = the same first 1600 val samples
#   * the trainer's val_loader has NO DistributedSampler and shuffle=False, so the
#     2-GPU runs scored this identical subset on every rank -- directly comparable.
#   * evaluate_crossmodal passes visible={src} and never forwards source_mask_ratio,
#     so the source is handed over whole here exactly as it was during training val.
#
# Single GPU, both arms sequentially: ~1600 samples x 1 source each, minutes.
#SBATCH --job-name=ws_text
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --open-mode=append
#SBATCH --output=logs/ws_text_%j.out
#SBATCH --error=logs/ws_text_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alloydas@iastate.edu

set -euo pipefail

cd /work/mech-ai-scratch/alloy/embodiedmae
mkdir -p logs

source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU" in
  *"PRO 6000"*) conda activate /work/mech-ai-scratch/alloy/.conda/envs/det_cu128 ;;
  *)            conda activate /work/mech-ai/alloy/.conda/envs/det ;;
esac

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

echo "GPU: ${GPU}"
echo "JobID : ${SLURM_JOB_ID}"
echo "Node  : $(hostname)"
echo "Date  : $(date)"

for CFG in configs/config_4m_distill_15k_rgb2text.yaml \
           configs/config_4m_distill_15k_pc2text.yaml; do
  echo ""
  echo "================================================================"
  echo "WARM-START BASELINE :: ${CFG}"
  echo "================================================================"
  python eval_warmstart.py "${CFG}"
done

echo "Done at $(date)"
