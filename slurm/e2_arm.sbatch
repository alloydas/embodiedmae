#!/bin/bash
#SBATCH --job-name=e2arm
#SBATCH --partition=scavenger
#SBATCH --account=mech-ai-scavenger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=logs/e2_%x_%j.out
#SBATCH --error=logs/e2_%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alloydas@iastate.edu

# E2 modality value-add -- one arm per job.
#   sbatch --job-name=e2_pc slurm/e2_arm.sbatch pc
# Extra args after the arm slug are passed straight to train_sorghum_4m.py, so a
# timing probe is just:
#   sbatch --job-name=e2_probe slurm/e2_arm.sbatch pc --epochs 3 --no_wandb
#
# Every arm must run with the SAME resources: batch size per GPU and GPU count
# both change the global batch, and a global-batch difference between arms would
# confound the one thing E2 is trying to measure. So the gres is pinned to
# a100:4 rather than left open:
#   * scavenger's 8-GPU nodes are nearly all draining; its a100:4 pool is ~18
#     healthy nodes, so all four arms place at once on identical hardware.
#   * 'a100' excludes 'a100-pcie' (different memory bandwidth) and excludes the
#     rtx_pro_6000 nodes, which are sm_120 and need the det_cu128 env, not det.
#
# scavenger is preemptible by design -- that is the point, it is not subject to
# the lab's 18-GPU AssocGrpGRES cap. --requeue plus the auto-resume below means a
# preemption costs at most save_freq epochs.

set -euo pipefail

ARM="${1:?usage: e2_arm.sbatch <pc|pcrgb|pcrgbd|pcrgbdt> [extra train args...]}"
shift || true

cd /work/mech-ai-scratch/alloy/embodiedmae
mkdir -p logs

source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
conda activate det

CONFIG="configs/config_e2_${ARM}.yaml"
[ -f "${CONFIG}" ] || { echo "no such config: ${CONFIG}"; exit 2; }
CKPT_DIR="./outputs/e2_${ARM}/checkpoints"

echo "JobID   : ${SLURM_JOB_ID}"
echo "Arm     : ${ARM}"
echo "Config  : ${CONFIG}"
echo "Node    : $(hostname)"
echo "Date    : $(date)"
nvidia-smi -L || true

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$((10000 + RANDOM % 50000))
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1

if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
  NPROC=${SLURM_GPUS_ON_NODE}
else
  NPROC=$(echo "${CUDA_VISIBLE_DEVICES//,/ }" | wc -w)
fi
[ "${NPROC}" -gt 0 ] 2>/dev/null || NPROC=4
echo "torchrun with ${NPROC} processes"

# Auto-resume: pick the highest-epoch checkpoint by its NUMBER, not by sorting
# the path (the output dir name also contains '_', which shifts the field).
RESUME_ARG=""
if [ -d "${CKPT_DIR}" ]; then
  LATEST=$(for f in "${CKPT_DIR}"/checkpoint_epoch_*.pth; do
             [ -e "$f" ] || continue
             b=$(basename "$f" .pth); echo "${b##*_} $f"
           done | sort -n | tail -1 | cut -d' ' -f2-)
  if [ -n "${LATEST:-}" ]; then
    echo "Resuming from ${LATEST}"
    RESUME_ARG="--resume ${LATEST}"
  fi
fi

srun --kill-on-bad-exit=1 \
  torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC} \
    train_sorghum_4m.py --config "${CONFIG}" ${RESUME_ARG} "$@"

echo "Done at $(date)"
