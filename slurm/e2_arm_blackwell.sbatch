#!/bin/bash
#SBATCH --job-name=e2armbw
#SBATCH --partition=scavenger
#SBATCH --account=mech-ai-scavenger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_pro_6000:2
#SBATCH --cpus-per-task=80
#SBATCH --mem=320G
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
# confound the one thing E2 exists to measure. 2 GPUs x --batch_size 16 = global
# batch 32, the same value the a100:4 variant reaches with 8 per GPU on 4 ranks,
# so the two launchers stay directly comparable.
#
# On memory: scale it WITH num_workers. Measured at 12 workers x 2 ranks the job
# held 65.8G RSS; backing out ~10G of model+CUDA context per node gives ~2.3G per
# worker, so 38 x 2 = 76 workers projects to ~187G -- which is why 192G was not
# enough headroom and this is 320G. 4 arms x 320G = 1280G of the node's 1494G.
# Memory is placement-neutral here (see the --test-only note below), so buy the
# headroom rather than discover it as an oom-kill 200 epochs in.
#
# On CPUs: this pipeline is DATALOADER-BOUND, so cpus-per-task/num_workers set
# throughput, not the GPU. Measured 2026-09-15: ~1.6 items/s per worker process.
#   headline v2: 44 workers x 8 ranks = 352 -> ~563 items/s (observed 565)
#   E2 at 12 x 2 =  24 workers          -> ~ 38 items/s (observed 33), 319 s/epoch
# 38 x 2 = 76 workers should give ~120 items/s, ~86 s/epoch. Extra CPUs are free
# here: `sbatch --test-only` puts 48, 80 and 96 CPUs at the same placement time.
# 4 arms x 80 leaves 64 of the node's 384 CPUs for everyone else.
#
# On sizing: do NOT shrink this request to chase the idle GPUs on nova26-gpu-1.
# Those 8 GPUs are genuinely unallocated, but nothing can land on that node --
# `sbatch --test-only` for 1 CPU / 1G / no GPU is pushed out just as far as a
# 200G request, so the node is held by a backfill reservation for a higher
# priority pending job, not by anything about our request. (nova26-gpu-1 also
# has 1475G of 1494G reserved by three CPU-only jobs, the largest being a 2-CPU
# 1000G job ending Sep 25 -- which is exactly the date the scheduler offers.)
# Measured start estimates on 2026-09-15: 1 or 2 GPUs both land Sep 16 17:09 on
# nova26-gpu-2, 4 GPUs slips to Sep 17 10:25, and asking for MORE cpu/mem did
# not delay placement at all. So take 2 GPUs and a comfortable host allocation.
#
# sm_120 needs det_cu128 (torch 2.11+cu128). The det env is torch 2.5+cu124 and
# has no sm_120 kernels, so it dies at the first kernel launch. det_cu128 is the
# same env the v2 headline pretrain used on these nodes.
#
# scavenger is preemptible by design -- it is not subject to the lab's 18-GPU
# AssocGrpGRES cap. --requeue plus the auto-resume below caps a preemption at
# save_freq epochs.
#
# ON NOVA (2026-10-02, data version 2). Scavenger cannot reach the Blackwell nodes
# while nova jobs sit on them (dheerajt's 20-day CPU jobs, since 09-28), so E2 runs
# there through nova, inside the mech-ai 17-GPU cap. Both nodes are PLANNED for
# another user's 12 h CPU jobs: a 2-day request never got a start time, while 4 h
# and 11 h requests were placed within 5 min. So submit long with --time-min, let
# backfill cut the limit to what fits, and requeue before the wall:
#   sbatch --job-name=e2_<arm>_d2 --partition=nova --account=mech-ai --qos=normal \
#     --gres=gpu:rtx_pro_6000:2 --cpus-per-task=80 --mem=160G \
#     --time=2-00:00:00 --time-min=04:00:00 --signal=B:USR1@600 --requeue \
#     --open-mode=append slurm/e2_arm_blackwell.sbatch <arm>_d2
# 160G: the v1 arms peaked at ~61 GiB (sacct MaxRSS) with 38 workers x 2 ranks.

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
# wandb.init starts a wandb-core server and waits x_service_wait (30 s by
# default) for it. With /work (NFS) slow, loading that binary from the conda env
# took longer, and on 2026-09-29 all five running jobs died at wandb.init with
# ServicePollForTokenError before their first step. Allow 10 minutes.
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-600}"

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

# Self-requeue before the wall, opt-in: only a job submitted with
# --signal=B:USR1@600 gets USR1, ~10 min before its limit. The trap then requeues
# the same job id, which resumes from the newest checkpoint (as
# slurm/distill_maize.sbatch and pretrain_15k_d2.sbatch do). --requeue alone does
# NOT cover TIMEOUT. A requeue is not an end, so a chunk queued afterany on this job
# keeps waiting (checked 2026-10-02). Without the signal nothing here changes.
requeue_before_wall() {
  echo "$(date) USR1: walltime in ~10 min; requeueing job ${SLURM_JOB_ID:-?} to resume"
  if scontrol requeue "${SLURM_JOB_ID}"; then
    echo "$(date) requeued ${SLURM_JOB_ID}"
  else
    echo "$(date) scontrol requeue FAILED: the job will end TIMEOUT and its chain carries on"
  fi
}
trap requeue_before_wall USR1

srun --kill-on-bad-exit=1 \
  torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC} \
    train_sorghum_4m.py --config "${CONFIG}" --batch_size 16 --num_workers 38 ${RESUME_ARG} "$@" &
SRUN_PID=$!
rc=0
wait "${SRUN_PID}" || rc=$?
if [ "${rc}" -gt 128 ] && kill -0 "${SRUN_PID}" 2>/dev/null; then
  # woken by the signal, not by srun: keep waiting (the requeue kills the step)
  rc=0
  wait "${SRUN_PID}" || rc=$?
fi
echo "srun exit ${rc}; done at $(date)"
exit "${rc}"
