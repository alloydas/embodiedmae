#!/bin/bash
#SBATCH --job-name=maize4m
#SBATCH --partition=scavenger
#SBATCH --account=mech-ai-scavenger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_pro_6000:2
#SBATCH --cpus-per-task=80
#SBATCH --mem=320G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=logs/maize_%x_%j.out
#SBATCH --error=logs/maize_%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alloydas@iastate.edu

# MAIZE 4M pretraining.
#
#   sbatch slurm/train_maize.sbatch
#   sbatch slurm/train_maize.sbatch --epochs 3 --no_wandb        # smoke test
#   sbatch --job-name=maize_small slurm/train_maize.sbatch --model_size small
#
# Resources are byte-identical to slurm/scale_arm_blackwell.sbatch ON PURPOSE:
# the maize run is matched to the E2/E3/E4 budget (600 epochs, global batch 32,
# ~197,000 optimizer steps) so the cross-species comparison is controlled. A
# resource difference would confound it.
#
# GLOBAL BATCH IS 32: --batch_size 16 x 2 GPUs. configs/config_maize.yaml carries
# 8 (the a100:4 value); this launcher overrides it, exactly as the E2/E3/E4
# launchers do. Verify after launch:
#   jq '.batch_size, .world_size, .num_points, .max_leaves' outputs/maize_4m/config.json
#
# DO NOT point this at train_sorghum_4m.py. Maize is 14 params over 28 leaf
# slots; sorghum is 9 over 24. The sorghum entry point would build the wrong
# ParamEmbed width and its dataset globs '*_nc_cam.ply', which matches nothing
# in the maize tree.

set -euo pipefail

cd /work/mech-ai-scratch/alloy/embodiedmae
mkdir -p logs outputs

source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh

# Pick the env from the GPU we landed on. `det` is a CUDA 12.4 build and dies on
# sm_120 with "no kernel image is available for execution on the device" — not
# at import, but at the first kernel launch, deep inside the PC embedder's FPS.
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
case "${GPU_NAME}" in
  *"RTX PRO 6000"*|*Blackwell*) CONDA_ENV=/work/mech-ai-scratch/alloy/.conda/envs/det_cu128 ;;
  *)                            CONDA_ENV=/work/mech-ai/alloy/.conda/envs/det ;;
esac
conda activate "${CONDA_ENV}"

CONFIG=configs/config_maize.yaml
OUT=./outputs/maize_4m
CKPT_DIR="${OUT}/checkpoints"

echo "JobID : ${SLURM_JOB_ID:-none}"
echo "Node  : $(hostname)"
echo "GPU   : ${GPU_NAME}"
echo "Env   : ${CONDA_ENV}"
echo "Config: ${CONFIG}"
echo "Date  : $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# The split must be fully transferred before a real run. An epoch is one view
# per plant, so a half-copied train split trains on a silent subset and the
# step budget no longer matches E2's. Refuse rather than produce a run that
# looks fine and is not comparable to anything.
for split in train val; do
  n=$(ls "/work/mech-ai-scratch/alloy/Maize/${split}" 2>/dev/null | grep -c '^plant_' || true)
  case "${split}" in
    train) want=105000 ;;
    val)   want=22500  ;;
  esac
  if [ "${n}" -lt "${want}" ]; then
    echo "REFUSING TO START: ${split} has ${n} view folders, expected ${want}."
    echo "The Globus transfer is still running. Re-submit when it completes, or"
    echo "pass --allow_partial_split if you deliberately want a subset run."
    exit 3
  fi
  echo "split ${split}: ${n}/${want} ✅"
done

NPROC=2

# Auto-resume: highest-epoch checkpoint that ACTUALLY OPENS. A preempted job can
# be killed mid-write, leaving a truncated .pth that is the highest-numbered one
# and therefore exactly the file a naive pick would choose; resuming from it
# fails at torch.load after the queue wait. Same walk-down as the E3/E4 launcher.
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
        print(f"skipping unreadable {os.path.basename(f)}: {type(e).__name__}",
              file=sys.stderr)
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
