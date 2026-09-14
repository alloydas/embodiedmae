#!/bin/bash
# RGB -> SPLINE PARAMETERS specialist distillation on the FULL Sorghum_15K.
# Config: configs/config_4m_distill_15k_pc2text.yaml   (target: [text] only, source_mask_ratio 0.5)
#
# Recovers the plant's procedural generation parameters from the point cloud alone.
# `mean_gen` in this run's history IS param_mae_masked and nothing else, so the
# headline number is the metric with no Chamfer mixed in.
#
# 2x A100-80GB on the *nova* partition, with gradient accumulation holding the
# experiment fixed: batch_size 16 x 2 GPUs x accum 4 = global batch 128, the same
# number every sibling arm used, and batch_size stays 16 so the non-synced
# BatchNorm1d in PointCloudEmbed still sees 16-sample micro-batches.
#
# Why 2 GPUs on nova rather than 8 on scavenger:
#   * The mech-ai account is capped at gres/gpu=18 for the WHOLE lab, and ~15 were
#     in use, so anything above ~3 GPUs on nova is rejected with AssocGrpGRES.
#   * scavenger bypasses that cap but is PriorityTier=0, behind every nova and
#     interactive job. An 8-GPU scavenger ask is a treadmill: the 8x A100 job
#     12248620 (this arm, 8-GPU) slipped 17:50 -> 19:12 on 2026-09-09 as
#     higher-tier jobs took each GPU the moment it freed (07:18 -> 16:00 -> 19:54).
#   * nova is PriorityTier=100, so once this is RUNNING it cannot be bumped by the
#     jobs that keep stealing the scavenger slot. Verified 2026-09-09 09:22-09:23:
#     real 1-, 2- and 3-GPU nova probes all started within ~60s.
#
# ~4x slower than the 8-GPU shape (~80h vs ~20h for 100 epochs), hence the 4-day
# walltime. If it is still short, the auto-resume block below continues from the
# highest-epoch checkpoint on resubmit.
#
# Loss balance was checked before submitting (job 12247376, 20 steps each):
# recon 0.047 against feat 0.040-0.049, i.e. the parameter objective sits level
# with the feature-distillation term rather than being drowned by it, so
# spline_loss_weight stays at the teacher's own 5.0.
#SBATCH --job-name=rgb2txtN2
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=150G
#SBATCH --time=4-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
# Append on requeue: SLURM truncates by default, which erased the epoch 1-3
# history when this job was first preempted.
#SBATCH --open-mode=append
#SBATCH --output=logs/rgb2textN2_%j.out
#SBATCH --error=logs/rgb2textN2_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
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
echo "GPU: ${GPU}"

CONFIG=configs/config_4m_distill_15k_rgb2text.yaml
CKPT_DIR=./outputs/4m_distill_15k_rgb2text/checkpoints

echo "JobID         : ${SLURM_JOB_ID}"
echo "Node          : $(hostname)"
echo "Date          : $(date)"
echo "GPUs visible  : ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count(), 'cc', torch.cuda.get_device_capability(0))"

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$((10000 + RANDOM % 50000))
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN
# Unbuffered: a preempted job otherwise logs nothing at all.
export PYTHONUNBUFFERED=1

# SLURM_GPUS_ON_NODE is a COUNT (e.g. "8"), not a list — use it directly.
if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
  NPROC=${SLURM_GPUS_ON_NODE}
else
  NPROC=$(echo "${CUDA_VISIBLE_DEVICES//,/ }" | wc -w)
fi
[ "${NPROC}" -gt 0 ] 2>/dev/null || NPROC=8
echo "Launching torchrun with ${NPROC} processes"

# Auto-resume from the highest-epoch checkpoint (survives requeue/preemption).
RESUME_ARG=""
if [ -d "${CKPT_DIR}" ]; then
  LATEST=$(for f in "${CKPT_DIR}"/checkpoint_epoch_*.pth; do
             [ -e "$f" ] || continue
             b=$(basename "$f" .pth); echo "${b##*_} $f"
           done | sort -n | tail -1 | cut -d' ' -f2-)
  if [ -n "${LATEST:-}" ]; then
    echo "Resuming from latest checkpoint: ${LATEST}"
    RESUME_ARG="--resume ${LATEST}"
  fi
fi

srun --kill-on-bad-exit=1 \
  torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC} \
    train_sorghum_4m_distill.py --config "${CONFIG}" --num_workers 10 --accum_steps 4 ${RESUME_ARG}

echo "Done at $(date)"
