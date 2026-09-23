#!/bin/bash
#SBATCH --job-name=maizescale
#SBATCH --partition=scavenger
#SBATCH --account=mech-ai-scavenger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_pro_6000:2
#SBATCH --cpus-per-task=80
#SBATCH --mem=320G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=logs/maizescale_%x_%j.out
#SBATCH --error=logs/maizescale_%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alloydas@iastate.edu

# E3 (data scaling) and E4 (model scaling) for MAIZE.
#
#   sbatch --job-name=maize_e3_1k    slurm/scale_arm_maize.sbatch maize_e3_1k
#   sbatch --job-name=maize_e4_large slurm/scale_arm_maize.sbatch maize_e4_large
#   sbatch --job-name=smoke slurm/scale_arm_maize.sbatch maize_e3_1k --epochs 3 --no_wandb
#
# The maize twin of slurm/scale_arm_blackwell.sbatch. Resources are byte-identical
# to it and to slurm/train_maize.sbatch ON PURPOSE: maize_4m is the shared
# reference point for both experiments (E3's full-data arm and E4's base-model
# arm), so a resource difference between that run and these would confound both
# curves -- and the sorghum arms use the same box, which is what makes the
# cross-species comparison a comparison.
#
# IT RUNS train_maize_4m.py. Pointing this at train_sorghum_4m.py would build a
# 9-param / 24-leaf ParamEmbed and glob '*_nc_cam.ply', which matches nothing in
# the maize tree -- "No valid samples found" at best, a silently mis-sized model
# at worst.
#
# GLOBAL BATCH IS 32: --batch_size 16 x 2 GPUs. The configs carry 8 (the a100:4
# value); this launcher overrides it, exactly as the sorghum launcher does.
# Verify: jq '.batch_size, .world_size, .num_points, .max_leaves' outputs/<slug>/config.json
#
# EQUAL STEPS, NOT EQUAL EPOCHS. The E3 configs carry very different `epochs`
# (6169 / 2100 / 631) precisely so every arm runs ~197,400 optimizer steps. Do
# NOT "fix" the epoch counts to match each other -- that would unfix the thing
# the experiment controls. See the header of any configs/config_maize_e3_*.yaml.
#
# On maize_e3_1k: 6169 epochs over a 1000-item epoch, so worker respawn is paid
# 6169 times (persistent_workers is deliberately off -- a persistent worker holds
# a stale dataset copy and freezes set_epoch(), silently pinning view sampling to
# epoch 0). Measured on the sorghum arms that overhead is only ~0.7 s/epoch
# (solving e3_1k's 13.8 s/ep at 32 steps against e3_3k's 39.2 s/ep at 94 steps),
# i.e. ~1.2 h over the whole arm -- real, but not the long pole.
#
# WALLTIME: --requeue does NOT cover a TIMEOUT. Slurm requeues on preemption,
# node failure or an admin requeue only, so a job that hits 2-00:00:00 ends
# TIMEOUT and nothing resubmits it -- the arm stops with its last checkpoint on
# disk and looks, from outside, exactly like one still running. Re-run the same
# sbatch line by hand; the auto-resume below then caps the loss at save_freq
# epochs. --requeue + auto-resume covers PREEMPTION, which on scavenger is the
# common case, and is what killed sorghum's e3_1k and e4_large at
# 2026-09-19T15:51:27.
#
# No arm is expected to need that. The maize reference (job 16551951) runs
# 329 steps in ~85 s = 0.258 s/step, putting every arm's ~197,400 steps at ~14 h.
# The long pole is maize_e4_large, NOT e3_1k -- sorghum's e4_large took ~20.7 h
# against e4_small's 10.7 h. Watch e4_large's first epochs: past ~0.5 s/step it
# is the one that could approach the wall.
#
# On maize_e4_large: 1024-dim, 24-layer encoder. If it ever OOMs at batch 16 the
# fix is the a100:4 launcher at --batch_size 8, NOT a smaller batch here and NOT
# fewer GPUs with a larger per-GPU batch -- global batch must stay 32.

set -euo pipefail

SLUG="${1:?usage: scale_arm_maize.sbatch <maize_e3_1k|maize_e3_3k|maize_e3_10k|maize_e4_small|maize_e4_large> [extra train args...]}"
shift || true

cd /work/mech-ai-scratch/alloy/embodiedmae
mkdir -p logs outputs

case "${SLUG}" in
  maize_*) ;;
  *) echo "REFUSING: slug '${SLUG}' is not a maize arm. This launcher runs"
     echo "train_maize_4m.py; use slurm/scale_arm_blackwell.sbatch for sorghum."
     exit 2 ;;
esac

CONFIG="configs/config_${SLUG}.yaml"
[ -f "${CONFIG}" ] || { echo "no such config: ${CONFIG}"; exit 2; }
CKPT_DIR="./outputs/${SLUG}/checkpoints"

source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh

# Pick the env from the GPU we land on rather than pinning the gres: `det` is a
# CUDA 12.4 build and dies on sm_120 with "no kernel image is available", not at
# import but at the first kernel launch, deep inside the PC embedder's FPS.
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
case "${GPU_NAME}" in
  *"RTX PRO 6000"*|*Blackwell*) CONDA_ENV=/work/mech-ai-scratch/alloy/.conda/envs/det_cu128 ;;
  *)                            CONDA_ENV=/work/mech-ai/alloy/.conda/envs/det ;;
esac
conda activate "${CONDA_ENV}"

echo "JobID   : ${SLURM_JOB_ID:-none}"
echo "Slug    : ${SLUG}"
echo "Config  : ${CONFIG}"
echo "Node    : $(hostname)"
echo "GPU     : ${GPU_NAME}"
echo "Env     : ${CONDA_ENV}"
echo "Date    : $(date)"
nvidia-smi -L || true

# Refuse a partially transferred split, exactly as slurm/train_maize.sbatch does.
# Under view_sampling an epoch is one view per plant, so a half-copied split
# trains on a silent subset and the step budget stops matching E2's -- which is
# the single thing these arms exist to hold fixed.
for split in train val; do
  n=$(ls "/work/mech-ai-scratch/alloy/Maize/${split}" 2>/dev/null | grep -c '^plant_' || true)
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
export PYTHONUNBUFFERED=1

if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
  NPROC=${SLURM_GPUS_ON_NODE}
else
  NPROC=$(echo "${CUDA_VISIBLE_DEVICES//,/ }" | wc -w)
fi
[ "${NPROC}" -gt 0 ] 2>/dev/null || NPROC=2
echo "torchrun with ${NPROC} processes"

# Auto-resume: highest-epoch checkpoint by its NUMBER (not by sorting the path --
# the output dir name also contains '_', which shifts the field), and it must
# actually OPEN. A preempted job can be killed mid-write, leaving a truncated
# .pth that is the highest-numbered one and therefore exactly what a naive pick
# would choose; resuming from it fails at torch.load after the queue wait, so the
# run makes no progress and fails identically on every retry. Walk down.
RESUME_ARG=""
if [ -d "${CKPT_DIR}" ]; then
  LATEST=$(python - "${CKPT_DIR}" <<'PYEOF'
import glob, os, sys, torch
fs = sorted(glob.glob(os.path.join(sys.argv[1], 'checkpoint_epoch_*.pth')),
            key=lambda f: int(f.rsplit('_', 1)[1].split('.')[0]), reverse=True)
for f in fs:
    try:
        torch.load(f, map_location='cpu', weights_only=False)
        print(f); break
    except Exception as e:
        print(f"skipping unreadable {os.path.basename(f)}: "
              f"{type(e).__name__}", file=sys.stderr)
PYEOF
)
  if [ -n "${LATEST:-}" ]; then
    echo "Resuming from ${LATEST}"
    RESUME_ARG="--resume ${LATEST}"
  fi
fi

srun --kill-on-bad-exit=1 \
  torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC} \
    train_maize_4m.py --config "${CONFIG}" --batch_size 16 --num_workers 38 \
    ${RESUME_ARG} "$@"

echo "Done at $(date)"
