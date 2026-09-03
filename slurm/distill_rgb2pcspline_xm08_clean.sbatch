#!/bin/bash
# CLEAN crossmodal_prob test: run B (job 12087915) rerun with xm 0.8 instead of 1.0
# and nothing else changed. 8x RTX PRO 6000 Blackwell, so the global batch matches
# run B's 128. Requires the cu128 env: the `det` env's torch 2.5 tops out at sm_90
# and cannot emit kernels for these sm_120 cards.
#
# CANCEL THIS BY HAND around epoch 60. `epochs: 500` in the config is there to keep
# the cosine schedule identical to run B's, not because 500 epochs are wanted.
#SBATCH --job-name=xm08clean
#SBATCH --partition=scavenger
#SBATCH --account=mech-ai-scavenger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# 4, not 8: global batch becomes 16 x 4 = 64, HALF run B's 128. This is a deliberate
# regime change, not a match to run B -- see the run E note in
# SPECIALIST_DISTILL_rgb2pc_spline.md. NOTE it buys no queue time: a --test-only sweep
# on 2026-09-01 gave an identical projected start for 1/2/3/4/6/8 GPUs, because all 16
# rtx_pro_6000 GPUs were idle and the wait was for CPUs held by `nova` jobs.
#SBATCH --gres=gpu:rtx_pro_6000:2
# CPU count is the ONLY scheduling lever here -- established 2026-09-01 with a
# `sbatch --test-only` sweep (submits nothing; vary one axis and read the projected
# start). The cliff sits between 128 and 144 CPUs:
#     <=128 CPUs -> nova26-gpu-1, start 2026-09-02T20:57
#     >=144 CPUs -> nova26-gpu-2, start 2026-09-07T22:05
# so 360 CPUs cost FIVE DAYS of queue for CPUs the job never used. Everything at or
# below 128 gets the SAME slot, so 12 is not faster to schedule than 120 -- it is set
# this low only because the run is capped at 12 CPU / 120G by request. Memory and
# walltime are irrelevant to placement: 800G vs 120G and 4h vs 7d all produce
# byte-identical start times. Neither is the GPU count -- even --gres=...:1 waits for
# the same slot, because no other job on the cluster requests rtx_pro_6000; the wait is
# 452 pending `nova` jobs (PriorityTier 100 vs scavenger's 0) projected to take
# nova26-gpu-1's CPUs the moment the 96-CPU MPI job 12172786 releases them.
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
# 2h: a short feasibility run, not the full continuation. NOTE --requeue covers
# PREEMPTION, not TIMEOUT -- at 2h this job simply ends and does not resume itself.
#SBATCH --time=02:00:00
#SBATCH --requeue
# Append on requeue: SLURM truncates by default, which erased the epoch 1-3
# history when this job was first preempted.
#SBATCH --open-mode=append
#SBATCH --output=logs/xm08clean_%j.out
#SBATCH --error=logs/xm08clean_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=alloydas@iastate.edu

set -euo pipefail

cd /work/mech-ai-scratch/alloy/embodiedmae
mkdir -p logs

source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
conda activate /work/mech-ai-scratch/alloy/.conda/envs/det_cu128

CONFIG=configs/config_4m_distill_15k_rgb2pc_spline_xm08_clean.yaml
CKPT_DIR=./outputs/4m_distill_15k_rgb2pc_spline_xm08_clean/checkpoints

echo "JobID         : ${SLURM_JOB_ID}"
echo "Node          : $(hostname)"
echo "Date          : $(date)"
echo "GPUs visible  : ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count(), 'cc', torch.cuda.get_device_capability(0))"

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$((10000 + RANDOM % 50000))
# 1, not 4: 8 ranks x 4 threads = 32 on a 12-CPU allocation is 2.7x oversubscribed and
# thrashes. With 12 CPUs the loader is the bottleneck anyway, not intra-op parallelism.
export OMP_NUM_THREADS=1
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
