#!/bin/bash
# Sorghum cross-modal distillation on DATA VERSION 2 from the Sinkhorn E1
# (configs/config_4m_distill_15k_all_d2.yaml). Global batch 128 = 16 x GPUs x accum, accum
# set from the GPU count (1, 2, 4 or 8 GPUs). Auto-resumes from the newest checkpoint and
# requeues itself 10 min before the wall (USR1), so one submission runs to epoch 100.
#
#   sbatch --dependency=afterok:16929801 --kill-on-invalid-dep=yes slurm/distill_15k_all_d2.sbatch
#   DISTILL_CONFIG=<yaml> NUM_WORKERS=16 sbatch ... slurm/distill_15k_all_d2.sbatch [extra trainer args]
#
# v1 measured 5.8 min/epoch on 8x RTX PRO 6000 (~10 h); expect ~12 min/epoch on 4 -> ~20 h.
#SBATCH --job-name=distill_d2
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_pro_6000:4
#SBATCH --cpus-per-task=80
#SBATCH --mem=280G
#SBATCH --time=4-00:00:00
#SBATCH --time-min=04:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=alloydas@iastate.edu

set -euo pipefail
cd /work/mech-ai-scratch/alloy/embodiedmae
mkdir -p logs
source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
case "${GPU_NAME}" in
  *"RTX PRO 6000"*|*Blackwell*) conda activate /work/mech-ai-scratch/alloy/.conda/envs/det_cu128 ;;
  *)                            conda activate /work/mech-ai/alloy/.conda/envs/det ;;
esac

CONFIG="${DISTILL_CONFIG:-configs/config_4m_distill_15k_all_d2.yaml}"
OUT_DIR=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['checkpointing']['output_dir'])" "${CONFIG}")
TEACHER=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['distill']['teacher_checkpoint'])" "${CONFIG}")
CKPT_DIR="${OUT_DIR}/checkpoints"
NUM_WORKERS="${NUM_WORKERS:-16}"

echo "JobID   : ${SLURM_JOB_ID}   restarts ${SLURM_RESTART_COUNT:-0}"
echo "Config  : ${CONFIG} -> ${OUT_DIR}"
echo "Node    : $(hostname)   GPU ${GPU_NAME}   CPUs ${SLURM_CPUS_PER_TASK:-?}   workers/rank ${NUM_WORKERS}"
echo "Date    : $(date)"
nvidia-smi -L || true
if [ ! -f "${TEACHER}" ]; then
  echo "teacher checkpoint ${TEACHER} is missing: E1 has not reached its last epoch. Exiting 3."
  exit 3
fi

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$((10000 + RANDOM % 50000))
export OMP_NUM_THREADS=2
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-600}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
  NPROC=${SLURM_GPUS_ON_NODE}
else
  NPROC=$(echo "${CUDA_VISIBLE_DEVICES//,/ }" | wc -w)
fi
case "${NPROC}" in
  1|2|4|8) ACCUM=$((8 / NPROC)) ;;
  *) echo "the global batch of 128 (8 x 16) needs 1, 2, 4 or 8 GPUs; got ${NPROC}"; exit 2 ;;
esac
echo "GPUs ${NPROC} x batch 16 x accum ${ACCUM} = global batch 128"

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

requeue_before_wall() {
  echo "$(date) USR1: walltime in ~10 min; requeueing job ${SLURM_JOB_ID:-?} to resume"
  scontrol requeue "${SLURM_JOB_ID}" && echo "$(date) requeued ${SLURM_JOB_ID}" \
    || echo "$(date) scontrol requeue FAILED: the job will end TIMEOUT; re-run this sbatch line by hand"
}
trap requeue_before_wall USR1

srun --kill-on-bad-exit=1 \
  torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC} \
    train_sorghum_4m_distill.py --config "${CONFIG}" --num_workers "${NUM_WORKERS}" \
      --accum_steps "${ACCUM}" ${RESUME_ARG} "$@" &
SRUN_PID=$!
rc=0
wait "${SRUN_PID}" || rc=$?
if [ "${rc}" -gt 128 ] && kill -0 "${SRUN_PID}" 2>/dev/null; then
  rc=0
  wait "${SRUN_PID}" || rc=$?
fi
echo "srun exit ${rc} at $(date)"
exit "${rc}"
