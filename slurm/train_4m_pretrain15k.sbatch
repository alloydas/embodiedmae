#!/bin/bash
#SBATCH --job-name=emb4m15k
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=10-00:00:00
#SBATCH --requeue
#SBATCH --output=logs/emb4m15k_%j.out
#SBATCH --error=logs/emb4m15k_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=alloydas@iastate.edu

set -euo pipefail

# ── Workdir ───────────────────────────────────────────────────────────────────
cd /work/mech-ai-scratch/alloy/embodiedmae
mkdir -p logs

# ── Conda env (det) ───────────────────────────────────────────────────────────
source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
conda activate det

CONFIG=configs/config_4m_pretrain15k.yaml
CKPT_DIR=./outputs/4m_pretrain_15k/checkpoints

# ── Diagnostics ───────────────────────────────────────────────────────────────
echo "JobID         : ${SLURM_JOB_ID}"
echo "Node          : $(hostname)"
echo "Date          : $(date)"
echo "GPUs visible  : ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count())"

# ── Distributed env ───────────────────────────────────────────────────────────
export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$((10000 + RANDOM % 50000))
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN

# SLURM_GPUS_ON_NODE is a COUNT (e.g. "4"), not a list — use it directly.
# Fall back to counting CUDA_VISIBLE_DEVICES entries if it is unset.
if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
  NPROC=${SLURM_GPUS_ON_NODE}
else
  NPROC=$(echo "${CUDA_VISIBLE_DEVICES//,/ }" | wc -w)
fi
[ "${NPROC}" -gt 0 ] 2>/dev/null || NPROC=4
echo "Launching torchrun with ${NPROC} processes"

# ── Auto-resume from latest checkpoint if one exists (survives requeue/wall-time)
RESUME_ARG=""
if [ -d "${CKPT_DIR}" ]; then
  # Pick the highest-epoch checkpoint by the NUMBER in its basename. Sorting the
  # full path on '_' is wrong — the dir name (4m_pretrain_15k) also has '_', so
  # the epoch field shifts and 80 sorts after 220. Extract the number instead.
  LATEST=$(for f in "${CKPT_DIR}"/checkpoint_epoch_*.pth; do
             [ -e "$f" ] || continue
             b=$(basename "$f" .pth); echo "${b##*_} $f"
           done | sort -n | tail -1 | cut -d' ' -f2-)
  if [ -n "${LATEST:-}" ]; then
    echo "Resuming from latest checkpoint: ${LATEST}"
    RESUME_ARG="--resume ${LATEST}"
  fi
fi

# ── Launch ────────────────────────────────────────────────────────────────────
srun --kill-on-bad-exit=1 \
  torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=${NPROC} \
    train_sorghum_4m.py \
    --config "${CONFIG}" \
    ${RESUME_ARG}

echo "Done at $(date)"
