#!/bin/bash
#SBATCH --job-name=e1_pretrain_d2
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_pro_6000:8
#SBATCH --cpus-per-task=180
#SBATCH --mem=600G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append
#SBATCH --output=logs/e1_pretrain_d2_%j.out
#SBATCH --error=logs/e1_pretrain_d2_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alloydas@iastate.edu

# E1, the headline 4M pretrain, on DATA VERSION 2 (rgb_nobg.png + the width leaf
# layout): configs/config_4m_pretrain15k_d2.yaml, which is the v2 recipe
# (config_4m_pretrain15k_v2_blackwell.yaml) unchanged -- 8 GPUs x 32 = global batch
# 256, 1000 epochs over all 105,000 views, lr 6e-4 -- plus rgb_file and pruning.
#
# WHY 8 RTX PRO 6000 ON NOVA. E1 reads every view every epoch, so it is bound by
# the data loader (~1 s per item per worker on 2026-10-01, four NFS files per
# item), not the GPUs. The v2 run's 3.1 min/epoch came from ~350 workers on a
# 384-CPU Blackwell node. Scavenger's 8-GPU A100 nodes have 96-128 CPUs (an
# estimated 10-14 days), and both Blackwell nodes host nova CPU jobs, which a
# scavenger job cannot share. 180 CPUs = 8 ranks x 21 workers + the main
# processes; about 7 days at the measured rate.
#
# 4-HOUR SEGMENTS THAT REQUEUE THEMSELVES, as slurm/distill_maize.sbatch does:
# the Blackwell nodes are often PLANNED, which admits only jobs that end before
# the reservation. Slurm sends USR1 ~10 min before the wall; the trap requeues
# the same job id, which resumes from the newest checkpoint (save_freq 5 epochs,
# pruned to the newest 3 + every 100th). --requeue alone does NOT cover TIMEOUT.
#
#   sbatch slurm/pretrain_15k_d2.sbatch                      # nova, 8x RTX PRO 6000
#   NUM_WORKERS=10 sbatch --partition=scavenger --account=mech-ai-scavenger --qos=scavenger \
#       --gres=gpu:a100:8 --cpus-per-task=90 --mem=440G slurm/pretrain_15k_d2.sbatch
#   (scavenger, 8x A100 80 GB on a 96-CPU node: the user's choice on 2026-10-02; about
#   half the loader throughput of the Blackwell node, so ~6-10 days of running)
set -euo pipefail
cd /work/mech-ai-scratch/alloy/embodiedmae
mkdir -p logs
source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
conda activate /work/mech-ai-scratch/alloy/.conda/envs/det_cu128

CONFIG=configs/config_4m_pretrain15k_d2.yaml
OUT_DIR=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['checkpointing']['output_dir'])" "${CONFIG}")
CKPT_DIR="${OUT_DIR}/checkpoints"
NUM_WORKERS="${NUM_WORKERS:-21}"

echo "JobID   : ${SLURM_JOB_ID}   restarts ${SLURM_RESTART_COUNT:-0}"
echo "Config  : ${CONFIG} -> ${OUT_DIR}"
echo "Node    : $(hostname)   CPUs ${SLURM_CPUS_PER_TASK:-?}   workers/rank ${NUM_WORKERS}"
echo "Date    : $(date)"
nvidia-smi -L || true

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$((10000 + RANDOM % 50000))
export OMP_NUM_THREADS=2
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-600}"
if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
  NPROC=${SLURM_GPUS_ON_NODE}
else
  NPROC=$(echo "${CUDA_VISIBLE_DEVICES//,/ }" | wc -w)
fi
[ "${NPROC}" -eq 8 ] || { echo "E1 needs 8 GPUs for its global batch of 256 (8 x 32); got ${NPROC}"; exit 2; }

# newest checkpoint by epoch; train_sorghum_4m.resolve_resume walks an unreadable one down
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
  if scontrol requeue "${SLURM_JOB_ID}"; then
    echo "$(date) requeued ${SLURM_JOB_ID}"
  else
    echo "$(date) scontrol requeue FAILED: the job will end TIMEOUT; re-run this sbatch line by hand"
  fi
}
trap requeue_before_wall USR1

srun --kill-on-bad-exit=1 \
  torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC} \
    train_sorghum_4m.py --config "${CONFIG}" --num_workers "${NUM_WORKERS}" ${RESUME_ARG} "$@" &
SRUN_PID=$!
rc=0
wait "${SRUN_PID}" || rc=$?
if [ "${rc}" -gt 128 ] && kill -0 "${SRUN_PID}" 2>/dev/null; then
  # woken by the signal, not by srun: keep waiting (the requeue kills the step)
  rc=0
  wait "${SRUN_PID}" || rc=$?
fi
echo "srun exit ${rc} at $(date)"
exit "${rc}"
