#!/bin/bash
#SBATCH --job-name=e2armbw
#SBATCH --partition=scavenger
#SBATCH --account=mech-ai-scavenger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=4G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=logs/e2bw_%x_%j.out
#SBATCH --error=logs/e2bw_%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alloydas@iastate.edu

# E2 modality value-add on the RTX PRO 6000 (Blackwell) nodes -- one arm per job.
#   sbatch --job-name=e2_pc slurm/e2_arm_blackwell.sbatch pc
# Extra args after the arm slug are passed straight to train_sorghum_4m.py, so a
# timing probe is just:
#   sbatch --job-name=e2_probe slurm/e2_arm_blackwell.sbatch pc --epochs 3 --no_wandb
#
# Every arm must run with the SAME resources: batch size per GPU and GPU count
# both set the global batch, and a global-batch difference between arms would
# confound the one thing E2 exists to measure.
#
# Why 1 GPU per arm when 8 sit free: the GPUs are idle because the node's MEMORY
# is gone, not because the node is unloaded. nova26-gpu-1 is RealMemory 1494G
# with AllocMem 1475G -- 98.7% reserved by other jobs, leaving ~19G schedulable
# for everything. (FreeMem reads ~1.4T; that is physical RAM other jobs reserved
# and are not touching. SLURM schedules on AllocMem, so FreeMem is a red herring
# and asking for 200G here pends forever on ReqNodeNotAvail.) CPUs were never
# tight: 130 of 384 allocated.
#
# So host memory, not GPU count, sets the arm width. One rank per arm instead of
# two halves the per-job CUDA host context and drops the NCCL buffers entirely,
# which is what lets four concurrent arms fit in 19G. --batch_size 32 on a single
# rank gives global batch 32 -- the same value the a100:4 variant reaches with 8
# per GPU on 4 ranks, so the two launchers stay directly comparable.
#
# This leaves 4 of the 8 GPUs unused. That is the node's memory reservation
# talking, not a scheduling mistake: widening to 2 GPUs per arm needs the other
# jobs to release memory first.
#
# sm_120 needs det_cu128 (torch 2.11+cu128). The det env is torch 2.5+cu124 and
# has no sm_120 kernels, so it dies at the first kernel launch. det_cu128 is the
# same env the v2 headline pretrain used on these nodes.
#
# scavenger is preemptible by design -- it is not subject to the lab's 18-GPU
# AssocGrpGRES cap. --requeue plus the auto-resume below caps a preemption at
# save_freq epochs.

set -euo pipefail

ARM="${1:?usage: e2_arm_blackwell.sbatch <pc|pcrgb|pcrgbd|pcrgbdt> [extra train args...]}"
shift || true

cd /work/mech-ai-scratch/alloy/embodiedmae
mkdir -p logs

source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
conda activate /work/mech-ai-scratch/alloy/.conda/envs/det_cu128

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
    train_sorghum_4m.py --config "${CONFIG}" --batch_size 32 --num_workers 3 ${RESUME_ARG} "$@"

echo "Done at $(date)"
