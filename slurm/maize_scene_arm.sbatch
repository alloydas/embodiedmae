#!/bin/bash
#SBATCH --job-name=maize_scene
#SBATCH --partition=scavenger
#SBATCH --account=mech-ai-scavenger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --constraint=a100|l40s|h200|rtxpro6000
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=logs/maize_%x_%j.out
#SBATCH --error=logs/maize_%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yongyun@iastate.edu

# MAIZE occluded-scene arms (train_maize_4m.py), the twin of sm_arm.sbatch:
# one arm x one seed per job, identical resources so the species compare.
#   sbatch --job-name=maize_scene_s1        slurm/maize_scene_arm.sbatch scene 1
#   sbatch --job-name=maize_scene_struct_s1 slurm/maize_scene_arm.sbatch scene_struct 1
# Config configs/config_maize_<arm>.yaml, output outputs/maize_<arm>_s<seed>.
# Extra args after the seed go straight to train_maize_4m.py.
#
# Always 2 GPUs x 16 (global 32), as every reference run: PointCloudEmbed's
# BatchNorm is not synced, so the per-GPU batch sets its statistics.
# GPU type: myenv (torch 2.11 + CUDA 13) runs on A100 / L40S / H200 / RTX PRO
# 6000 but not on the V100s a bare --gres=gpu:2 can land on -- hence the
# constraint. scavenger + --requeue + the auto-resume below bound a preemption
# at save_freq epochs; a walltime TIMEOUT is NOT requeued -- rerun the line.

set -euo pipefail

ARM="${1:?usage: maize_scene_arm.sbatch <scene|scene_struct> <seed> [extra train args...]}"
SEED="${2:?usage: maize_scene_arm.sbatch <scene|scene_struct> <seed> [extra train args...]}"
shift 2

REPO=/work/mech-ai-scratch/yongyun/embodiedmae4m_distilation
cd "${REPO}"
mkdir -p logs

CONFIG="configs/config_maize_${ARM}.yaml"
[ -f "${CONFIG}" ] || { echo "no such config: ${CONFIG}"; exit 2; }
RUN="maize_${ARM}_s${SEED}"
OUT="./outputs/${RUN}"
PYTHON_BIN=/work/mech-ai-scratch/yongyun/envs/myenv/bin/python

echo "JobID   : ${SLURM_JOB_ID}  (restart ${SLURM_RESTART_COUNT:-0})"
echo "Run     : ${RUN}  config ${CONFIG}"
echo "Node    : $(hostname)"
echo "Date    : $(date)"
nvidia-smi -L || true

export OMP_NUM_THREADS=2
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
# wandb-core start-up on the slow /work disk can exceed the 30 s default.
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-600}"
export WANDB_INIT_TIMEOUT=600
# Job 16686223 (sm_scene_off_s1, A100-40GB) died mid-validation with 13 GiB
# reserved-but-unallocated: two val passes (clean + occluded) per val epoch
# fragment the cache. Expandable segments let freed blocks be reused.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/xdg_${SLURM_JOB_ID}"
mkdir -p "${XDG_CACHE_HOME}"

NPROC=${SLURM_GPUS_ON_NODE:-2}
[ "${NPROC}" -eq 2 ] || { echo "expected 2 GPUs, got ${NPROC}"; exit 3; }
WORKERS=$(( (SLURM_CPUS_PER_TASK - 4) / NPROC ))

# Auto-resume from the highest-numbered checkpoint that actually opens.
RESUME_ARG=""
if [ -d "${OUT}/checkpoints" ]; then
  for f in $(for c in "${OUT}"/checkpoints/checkpoint_epoch_*.pth; do
               [ -e "$c" ] || continue
               b=$(basename "$c" .pth); echo "${b##*_} $c"
             done | sort -rn | cut -d' ' -f2-); do
    if "${PYTHON_BIN}" -c "import torch,sys; torch.load(sys.argv[1], map_location='cpu', weights_only=False, mmap=True)" "$f" 2>/dev/null; then
      RESUME_ARG="--resume $f"; break
    fi
    echo "unreadable checkpoint, skipping: $f"
  done
fi
# INIT_CKPT (env): continue ANOTHER run's checkpoint into this run's own
# directory -- used only while this run has no checkpoint of its own, so a
# preempted continuation resumes from its latest epoch, not from INIT_CKPT.
if [ -z "${RESUME_ARG}" ] && [ -n "${INIT_CKPT:-}" ]; then
  [ -f "${INIT_CKPT}" ] || { echo "INIT_CKPT not found: ${INIT_CKPT}"; exit 4; }
  RESUME_ARG="--resume ${INIT_CKPT}"
fi
echo "Resume  : ${RESUME_ARG:-<scratch>}"

srun --kill-on-bad-exit=1 \
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
    train_maize_4m.py --config "${CONFIG}" --seed "${SEED}" \
      --output_dir "${OUT}" --wandb_name "${RUN}" \
      --batch_size 16 --num_workers "${WORKERS}" ${RESUME_ARG} "$@"

echo "Done at $(date)"
