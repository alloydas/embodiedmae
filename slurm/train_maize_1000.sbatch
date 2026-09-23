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

# MAIZE 4M — epochs 601..1000, continuing the 600-epoch run.
#
#   sbatch --dependency=afterok:<600-run jobid> slurm/train_maize_1000.sbatch
#
# IT READS outputs/maize_4m AND WRITES outputs/maize_4m_1000ep. That separation is
# the whole design, and it is not cosmetic:
#
#   outputs/maize_4m          FROZEN at epoch 600 = 197,400 optimizer steps.
#                             This is E3's full-data point, E4's base-model point,
#                             and the only maize artefact comparable to sorghum's
#                             e2_pcrgbdt / e3_* / e4_* arms.
#   outputs/maize_4m_1000ep   The continuation. 329,000 steps with a cosine
#                             restart in it. NOT comparable to any of the above.
#
# An earlier version of this script resumed in place and merely backed the
# 600-epoch artefacts up to *_600ep.*. That left outputs/maize_4m/best_model.pth
# and config.json describing a 1000-epoch run, so anyone plotting the E4 curve
# would have reached for the directory the configs name and compared two
# 197,400-step arms against a 329,000-step middle point -- an inverted-looking
# model-scaling curve with no error message anywhere. Writing elsewhere removes
# the trap instead of documenting it.
#
# THIS IS A WARM RESTART, NOT A SMOOTH CONTINUATION. lr_lambda is a cosine over
# args.epochs, and LambdaLR.state_dict() stores None for a plain-function lambda,
# so resuming with --epochs 1000 rebuilds the schedule over 1000 and restores only
# last_epoch=600. Measured:
#
#     600-epoch cosine, ep 599 : lr = 1.18e-09   (effectively frozen)
#     resumed @1000,    ep 600 : lr = 5.56e-05   <- jumps 47,101x
#
# then decays to 0 by epoch 1000. That is a second cosine cycle (SGDR-style) and
# a deliberate choice over a clean 1000-epoch run (~24 h vs ~9.6 h). Report it as
# "600 epochs + 400 with a cosine restart", never as a 1000-epoch cosine.
#
# W&B: checkpoint_epoch_600.pth carries wandb_run_id (verified: yer9kdrv), and
# train_maize_4m.py resumes that id, so epochs 601..1000 continue the SAME W&B
# run rather than starting a new one -- one continuous curve across both phases.

set -euo pipefail

cd /work/mech-ai-scratch/alloy/embodiedmae
mkdir -p logs outputs

source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh

# Env from the GPU we land on: `det` is CUDA 12.4 and dies on sm_120 at the first
# kernel launch, not at import.
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
case "${GPU_NAME}" in
  *"RTX PRO 6000"*|*Blackwell*) CONDA_ENV=/work/mech-ai-scratch/alloy/.conda/envs/det_cu128 ;;
  *)                            CONDA_ENV=/work/mech-ai/alloy/.conda/envs/det ;;
esac
conda activate "${CONDA_ENV}"

CONFIG=configs/config_maize.yaml
SRC=./outputs/maize_4m                 # read-only here
OUT=./outputs/maize_4m_1000ep          # everything this job writes
CKPT600="${SRC}/checkpoints/checkpoint_epoch_600.pth"

echo "JobID : ${SLURM_JOB_ID:-none}"
echo "Node  : $(hostname)"
echo "GPU   : ${GPU_NAME}"
echo "Env   : ${CONDA_ENV}"
echo "Src   : ${SRC} (frozen, read-only)"
echo "Out   : ${OUT}"
echo "Date  : $(date)"

# The 600-epoch run must actually have reached 600. Without this the job would
# resume from whatever checkpoint is newest and report it as a 1000-epoch result.
if [ ! -f "${CKPT600}" ]; then
    echo "REFUSING TO START: ${CKPT600} does not exist."
    echo "The 600-epoch run has not completed. This script continues that run;"
    echo "it does not start one."
    exit 3
fi

# Resume source. First launch: the 600-epoch checkpoint from the frozen run.
# After a preemption: this job's own latest checkpoint, so it resumes at (say)
# 700 rather than restarting at 600. Walk down to one that actually OPENS -- a
# job killed mid-write leaves a truncated .pth that is the highest-numbered one
# and therefore exactly what a naive pick would choose.
RESUME="${CKPT600}"
if [ -d "${OUT}/checkpoints" ]; then
  LATEST=$(python - "${OUT}/checkpoints" <<'PYEOF'
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
  [ -n "${LATEST:-}" ] && RESUME="${LATEST}"
fi
echo "Resuming from ${RESUME}"

# WALLTIME: --requeue does NOT cover a TIMEOUT. Slurm requeues on preemption,
# node failure or an admin requeue only; a job that hits 2-00:00:00 ends TIMEOUT
# and nothing resubmits it. 400 epochs at ~85 s/epoch is ~9.5 h, so that is not
# expected here -- but if it does happen, re-run this same sbatch line by hand
# and the resume block above picks up from the last checkpoint.

NPROC=2

srun --kill-on-bad-exit=1 \
  torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC} \
    train_maize_4m.py --config "${CONFIG}" --batch_size 16 --num_workers 38 \
    --resume "${RESUME}" --epochs 1000 --output_dir "${OUT}" "$@"

echo "Done at $(date)"
