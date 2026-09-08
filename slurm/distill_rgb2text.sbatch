#!/bin/bash
# RGB -> SPLINE PARAMETERS specialist distillation on the FULL Sorghum_15K.
# Config: configs/config_4m_distill_15k_rgb2text.yaml   (target: [text] only, source_mask_ratio 0.5)
#
# Recovers the plant's procedural generation parameters from a single RGB render alone.
# `mean_gen` in this run's history IS param_mae_masked and nothing else, so the
# headline number is the metric with no Chamfer mixed in.
#
# Shape copied from the rgb2pc_smr50 run that this is the sibling of: 8 GPUs for
# global batch 128, matching every prior run and the pretrain. GPU count is what
# gates placement on scavenger; CPU count only matters for staying under the
# 128-CPU cliff.
#
# 24h walltime: the rgb2pc arms ran 6.8-8.0 min/epoch on 8x RTX PRO 6000, so 100
# epochs is ~12h. Walltime does not affect placement.
#
# Loss balance was checked before submitting (job 12247376, 20 steps each):
# recon 0.047 against feat 0.040-0.049, i.e. the parameter objective sits level
# with the feature-distillation term rather than being drowned by it, so
# spline_loss_weight stays at the teacher's own 5.0.
#SBATCH --job-name=rgb2text
#SBATCH --partition=scavenger
#SBATCH --account=mech-ai-scavenger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_pro_6000:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=600G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
# Append on requeue: SLURM truncates by default, which erased the epoch 1-3
# history when this job was first preempted.
#SBATCH --open-mode=append
#SBATCH --output=logs/rgb2text_%j.out
#SBATCH --error=logs/rgb2text_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=alloydas@iastate.edu

set -euo pipefail

cd /work/mech-ai-scratch/alloy/embodiedmae
mkdir -p logs

source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
conda activate /work/mech-ai-scratch/alloy/.conda/envs/det_cu128

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
    train_sorghum_4m_distill.py --config "${CONFIG}" ${RESUME_ARG}

echo "Done at $(date)"
