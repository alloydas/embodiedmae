#!/bin/bash
#SBATCH --job-name=maize4m1k
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

# MAIZE 4M — continuation of outputs/maize_4m from epoch 600 to epoch 1000.
#
#   sbatch --dependency=afterok:<600-run jobid> slurm/train_maize_1000.sbatch
#
# THIS IS A WARM RESTART, NOT A SMOOTH CONTINUATION. lr_lambda is a cosine over
# args.epochs, and LambdaLR.state_dict() stores None for a plain-function lambda,
# so resuming with --epochs 1000 rebuilds the schedule over 1000 and restores only
# last_epoch=600. Measured effect:
#
#     600-epoch cosine, ep 599 : lr = 1.18e-09   (effectively frozen)
#     resumed @1000,    ep 600 : lr = 5.56e-05   <- jumps 47,101x
#
# then decays to 0 by epoch 1000. That is a second cosine cycle (SGDR-style), and
# it is a deliberate choice, not an accident. Describe the result as "600 epochs
# + 400 with a cosine restart" -- NOT as a 1000-epoch cosine, which is a different
# run (see the "Separate clean 1000-epoch run" option that was declined).
#
# WHY EPOCH 600 STILL MATTERS: 329 steps/epoch x 600 = 197,400 optimizer steps,
# which is E2/E3/E4's budget exactly. checkpoint_epoch_600.pth is therefore THE
# comparable point for every maize-vs-sorghum claim, and this script backs up the
# 600-run's artefacts before touching them. The epoch-1000 model is a
# longer-training result, not an E2-matched one, and the two must not be swapped.

set -euo pipefail

cd /work/mech-ai-scratch/alloy/embodiedmae
mkdir -p logs outputs

source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh

# Same runtime env selection as slurm/train_maize.sbatch: `det` is CUDA 12.4 and
# dies on sm_120 at the first kernel launch, not at import.
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
case "${GPU_NAME}" in
  *"RTX PRO 6000"*|*Blackwell*) CONDA_ENV=/work/mech-ai-scratch/alloy/.conda/envs/det_cu128 ;;
  *)                            CONDA_ENV=/work/mech-ai/alloy/.conda/envs/det ;;
esac
conda activate "${CONDA_ENV}"

CONFIG=configs/config_maize.yaml
OUT=./outputs/maize_4m
CKPT_DIR="${OUT}/checkpoints"
CKPT600="${CKPT_DIR}/checkpoint_epoch_600.pth"

echo "JobID : ${SLURM_JOB_ID:-none}"
echo "Node  : $(hostname)"
echo "GPU   : ${GPU_NAME}"
echo "Env   : ${CONDA_ENV}"
echo "Date  : $(date)"

# The 600-epoch run must have actually reached 600. Without this the job would
# silently resume from whatever checkpoint happens to be newest and report it as
# a 1000-epoch result.
if [ ! -f "${CKPT600}" ]; then
    echo "REFUSING TO START: ${CKPT600} does not exist."
    echo "The 600-epoch run has not completed. This script continues that run;"
    echo "it does not start one."
    exit 3
fi

# Preserve the 600-epoch artefacts BEFORE the continuation overwrites them.
# best_model.pth is rewritten whenever val improves (best_val_loss is restored
# from the checkpoint, so it will improve), and config.json is rewritten with
# epochs=1000. training_history.json is *extended* rather than replaced -- the
# trainer restores history from the checkpoint -- but a snapshot at 600 makes the
# E2-matched segment readable without slicing.
#
# Guarded by -f on purpose: a preempted job is requeued and re-runs this block,
# and an unguarded copy would overwrite the 600 backup with 1000-run state.
for f in best_model.pth training_history.json config.json; do
    src="${OUT}/${f}"
    dst="${OUT}/${f%.*}_600ep.${f##*.}"
    if [ -f "${src}" ] && [ ! -f "${dst}" ]; then
        cp -p "${src}" "${dst}"
        echo "backed up ${f} -> $(basename "${dst}")"
    elif [ -f "${dst}" ]; then
        echo "backup $(basename "${dst}") already exists — left untouched"
    fi
done

# Auto-resume: highest-epoch checkpoint that ACTUALLY OPENS. On the first run of
# this script that is checkpoint_epoch_600.pth; after a preemption at, say, epoch
# 700 it is checkpoint_epoch_700.pth, so the continuation resumes rather than
# restarting from 600. Same walk-down as the other launchers -- a job killed
# mid-write leaves a truncated .pth that is the highest-numbered one and hence
# exactly what a naive pick would choose.
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
if [ -z "${LATEST:-}" ]; then
    echo "REFUSING TO START: no readable checkpoint in ${CKPT_DIR}."
    exit 4
fi
echo "Resuming from ${LATEST}"

NPROC=2

srun --kill-on-bad-exit=1 \
  torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC} \
    train_maize_4m.py --config "${CONFIG}" --batch_size 16 --num_workers 38 \
    --resume "${LATEST}" --epochs 1000 "$@"

echo "Done at $(date)"
