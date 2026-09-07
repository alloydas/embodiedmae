#!/bin/bash
# RGB -> POINT CLOUD specialist distillation on the FULL Sorghum_15K.
# Config: configs/config_4m_distill_15k_rgb2pc_smr50_nova.yaml   (target: [pc] only, source_mask_ratio 0.5)
#
# Shape chosen from an `sbatch --test-only` sweep on 2026-09-02 19:12:
#   8 GPU, any CPU count <=128 -> nova26-gpu-1, start 2026-09-03 05:10
#   8 GPU, 200 CPU             -> nova26-gpu-2, start 2026-09-07  (CPU cliff)
#   4 GPU                      -> nova26-gpu-1, start 2026-09-02 23:14
# GPU COUNT is the gate here (other scavenger users hold gpu-1's GPUs and free
# them at staggered times); CPU count only matters for staying under the cliff.
# 8 GPUs chosen for global batch 128, matching every prior run and the pretrain.
#
# 24h walltime: ~7.4 min/epoch measured on 8x RTX PRO 6000 (job 12132446 did 31
# epochs in 3h51m), so 100 epochs is ~12h. Walltime does not affect placement.
#SBATCH --job-name=smr50nova
#SBATCH --partition=nova
#SBATCH --account=mech-ai
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
#SBATCH --output=logs/smr50nova_%j.out
#SBATCH --error=logs/smr50nova_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=alloydas@iastate.edu

set -euo pipefail

cd /work/mech-ai-scratch/alloy/embodiedmae
mkdir -p logs

source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
conda activate /work/mech-ai-scratch/alloy/.conda/envs/det_cu128

CONFIG=configs/config_4m_distill_15k_rgb2pc_smr50_nova.yaml
CKPT_DIR=./outputs/4m_distill_15k_rgb2pc_smr50_nova/checkpoints

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
