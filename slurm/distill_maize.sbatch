#!/bin/bash
#SBATCH --job-name=distill_maize
#SBATCH --partition=scavenger
#SBATCH --account=mech-ai-scavenger
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=48
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
# Append on requeue: SLURM truncates by default, which erased the epoch 1-3
# history of the sorghum distill run the first time it was preempted.
#SBATCH --open-mode=append
# USR1 to this batch shell ~10 min before the wall; the trap below requeues the
# job, because --requeue alone does NOT cover a walltime TIMEOUT.
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# MAIZE cross-modal distillation — the maize twin of the sorghum run
# outputs/4m_distill_15k_all (slurm/distill_15k_all_blackwell.sbatch).
#
#   sbatch slurm/distill_maize.sbatch
#
# GPU smoke (40 micro-batches = 10 optimiser steps, then val + viz; prints PEAKMEM
# and steady it/s). Give it its own output_dir so nothing lands in the real run's:
#   sbatch --time=01:00:00 slurm/distill_maize.sbatch --epochs 1 --max_steps 40 \
#          --val_max_batches 10 --no_wandb --output_dir ./outputs/maize_distill_smoke
# (--max_steps also turns off the self-requeue below, so a smoke never loops.)
# (Run it before the real run has checkpoints: the auto-resume below always looks
# in outputs/maize_distill_all.)
#
# The 'before' number alone, on one GPU (or CPU) and without the teacher:
#   python train_maize_4m_distill.py --config configs/config_maize_distill_all.yaml --eval_only
# writes outputs/maize_distill_all/warmstart_eval.json -- the same number the run
# itself records as epoch 0 (history['val'][0]) before its first step.
#
# Shape: 16 per GPU x 2 GPUs x accum 4 = global 128, the reference's 16 x 8 x 1.
# The trainer refuses any other product (training.global_batch in the YAML), so a
# launch on a different GPU count needs accum_steps changed with it; keep
# batch_size at 16 either way (PointCloudEmbed's BatchNorm is per rank).
#
# RUNTIME: the same shape on 2x A100 (sorghum pc2textN2, job 12249835: 16 x 2 x 4,
# 3,282 micro-batches/epoch, crossmodal 0.8) measured 42.9 min/epoch mean (median
# 41, first three epochs 61-65 cold, fastest 30.6) = 71.5 h for 100 epochs. Maize
# should be comparable (lighter loader and a teacher that encodes 308 tokens, not
# 617; a student that encodes more on average), plus ~21 val passes: plan on
# ~60-75 h. It will NOT fit one 2-day wall; expect ONE requeue.
# --requeue covers preemption only, not a walltime TIMEOUT, so this script
# requeues ITSELF: Slurm sends USR1 ~10 min before the wall (--signal above) and
# the trap runs `scontrol requeue`. The requeued job re-enters the queue under the
# same job id, appends to the same log, and the block below resumes from the newest
# checkpoint that opens (loss <= save_freq = 2 epochs). If the requeue fails (the
# log says so) the job ends TIMEOUT as before: re-run the same sbatch line.
#
# DO NOT point this at train_sorghum_4m_distill.py: its dataset globs
# '*_nc_cam.ply' (matches nothing in the maize tree) and its model is 9 params
# over 24 leaves.

set -euo pipefail

cd /work/mech-ai-scratch/alloy/embodiedmae
mkdir -p logs

source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
conda activate /work/mech-ai-scratch/alloy/.conda/envs/det_cu128

CONFIG=configs/config_maize_distill_all.yaml
CKPT_DIR=./outputs/maize_distill_all/checkpoints

echo "JobID   : ${SLURM_JOB_ID:-none}"
echo "Node    : $(hostname)"
echo "Config  : ${CONFIG}"
echo "Date    : $(date)"
nvidia-smi -L || true

# Refuse a partially transferred split, exactly as slurm/train_maize.sbatch does.
# This run reads every view folder each epoch, so a half-copied split would train
# on a silent subset and the 821-steps/epoch budget would stop matching the
# sorghum reference -- the thing this run exists to hold fixed.
for split in train val; do
  # Count folders from the directory's link count (subdirectories + 2), not by
  # listing it -- same guard as slurm/scale_arm_maize.sbatch. /work is NFS: on
  # 2026-09-29 an `ls` of the 105,000-entry train split sat in uninterruptible NFS
  # wait for 45+ min while `stat` on the same directory returned in 0.2 s. The split
  # roots hold only plant_* folders plus _params.json, a file, which adds no link --
  # so links - 2 is the view-folder count exactly (train 105002 -> 105000).
  links=$(stat -c %h "/work/mech-ai-scratch/alloy/Maize/${split}" 2>/dev/null || echo 2)
  n=$(( links - 2 ))
  case "${split}" in
    train) want=105000 ;;
    val)   want=22500  ;;
  esac
  if [ "${n}" -lt "${want}" ]; then
    echo "REFUSING TO START: ${split} has ${n} view folders, expected ${want}."
    exit 3
  fi
  echo "split ${split}: ${n}/${want} ✅"
done

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$((10000 + RANDOM % 50000))
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN
# Unbuffered: a preempted job otherwise logs nothing at all.
export PYTHONUNBUFFERED=1
# wandb.init starts a wandb-core server and waits x_service_wait (30 s by
# default) for it. With /work (NFS) slow, loading that binary from the conda env
# took longer, and on 2026-09-29 all five running jobs died at wandb.init with
# ServicePollForTokenError before their first step. Allow 10 minutes.
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-600}"

# Newest checkpoint that actually opens: a preemption can leave the last one
# truncated (the trainer writes temp + os.replace, but a file from any other
# writer, or a filesystem fault, can still be torn). Same walk-down as
# slurm/e8_supervised.sbatch. keep_last=2 in the YAML guarantees a fallback.
RESUME_ARG=""
if [ -d "${CKPT_DIR}" ]; then
  for f in $(ls "${CKPT_DIR}"/checkpoint_epoch_*.pth 2>/dev/null | sed 's/.*_\([0-9]*\)\.pth$/\1 &/' | sort -rn | cut -d' ' -f2); do
    if python -c "import torch,sys; torch.load(sys.argv[1], map_location='cpu', weights_only=False, mmap=True)" "$f" 2>/dev/null; then
      RESUME_ARG="--resume $f"; echo "Resuming from $f"; break
    fi
    echo "Skipping unreadable checkpoint $f"
  done
fi

# Self-requeue at the wall (see RUNTIME above). bash runs a trap only between
# commands, so srun goes to the background and the shell waits on it: USR1 then
# interrupts `wait`, the handler requeues, and Slurm tears the step down.
# A smoke or eval-only launch (--max_steps / --eval_only) must never requeue itself:
# a smoke that overran its short --time would otherwise loop.
SELF_REQUEUE=1
for a in "$@"; do
  case "${a}" in --max_steps|--max_steps=*|--eval_only) SELF_REQUEUE=0 ;; esac
done
requeue_before_wall() {
  if [ "${SELF_REQUEUE}" != 1 ]; then
    echo "$(date) USR1: walltime in ~10 min; smoke/eval launch, NOT requeueing"
    return 0
  fi
  echo "$(date) USR1: walltime in ~10 min; requeueing job ${SLURM_JOB_ID:-?} to resume"
  if scontrol requeue "${SLURM_JOB_ID}"; then
    echo "$(date) requeued ${SLURM_JOB_ID}"
  else
    echo "$(date) scontrol requeue FAILED: the job will end TIMEOUT; re-run this sbatch line by hand"
  fi
}
trap requeue_before_wall USR1

# 48 CPUs: 2 ranks x 16 persistent train workers (num_workers in the YAML) + 8 val
# workers on rank 0 during val epochs + the two main processes.
srun --kill-on-bad-exit=1 \
  torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    train_maize_4m_distill.py --config "${CONFIG}" ${RESUME_ARG} "$@" &
SRUN_PID=$!
rc=0
wait "${SRUN_PID}" || rc=$?
if [ "${rc}" -gt 128 ] && kill -0 "${SRUN_PID}" 2>/dev/null; then
  # woken by the signal, not by srun: keep waiting (Slurm's requeue kills the
  # step; without a requeue, training uses its last ~10 min and srun's own exit
  # code is what this script reports)
  rc=0
  wait "${SRUN_PID}" || rc=$?
fi
echo "srun exit ${rc} at $(date)"
exit "${rc}"
