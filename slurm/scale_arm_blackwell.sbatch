#!/bin/bash
#SBATCH --job-name=scalearm
#SBATCH --partition=scavenger
#SBATCH --account=mech-ai-scavenger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_pro_6000:2
#SBATCH --cpus-per-task=80
#SBATCH --mem=320G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=logs/scale_%x_%j.out
#SBATCH --error=logs/scale_%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alloydas@iastate.edu

# E3 (data scaling) and E4 (model scaling) on the RTX PRO 6000 nodes.
#
# Generalised from e2_arm_blackwell.sbatch: that one hardcoded the e2_ prefix in
# both the config path and the output dir, so it could only launch E2 arms. This
# takes the run slug directly and derives configs/config_<slug>.yaml and
# outputs/<slug>, which is why E3 and E4 share one launcher.
#
#   sbatch --job-name=e3_1k     slurm/scale_arm_blackwell.sbatch e3_1k
#   sbatch --job-name=e4_large  slurm/scale_arm_blackwell.sbatch e4_large
#   sbatch --job-name=probe     slurm/scale_arm_blackwell.sbatch e3_1k --epochs 3 --no_wandb
#
# Everything below -- resources, env, global batch, auto-resume -- is identical
# to e2_arm_blackwell.sbatch ON PURPOSE. e2_pcrgbdt is the shared reference point
# for both experiments (E3's full-data arm and E4's base-model arm), so a
# resource difference between that run and these would confound both curves.
#
# GLOBAL BATCH IS 32: --batch_size 16 x 2 GPUs. The configs carry 8 (the a100:4
# value); this launcher overrides it, exactly as the E2 pair does. Verify after
# launch with: jq '.batch_size, .world_size' outputs/<slug>/config.json
#
# EQUAL STEPS, NOT EQUAL EPOCHS. The E3 configs carry very different `epochs`
# (6169 / 2100 / 631) precisely so that every arm runs ~197,400 optimizer steps.
# Do NOT "fix" the epoch counts to match each other -- that would unfix the thing
# the experiment controls. See the header of any configs/config_e3_*.yaml.
#
# On e4_large: 1024-dim, 24-layer encoder, ~3x base activations. It fits in the
# 96G card at batch 16, but if it ever OOMs the fix is the a100:4 launcher at
# --batch_size 8, NOT a smaller batch here -- global batch must stay 32.
#
# On e3_1k: 6169 epochs over a 1000-item epoch. The dataloader respawns its
# workers every epoch (persistent_workers is deliberately off -- persistent
# workers would hold a stale copy of the dataset and freeze set_epoch(), silently
# pinning view sampling to epoch 0), so per-epoch overhead is amortised over only
# ~8s of compute here. Watch the first minutes of the log: if the wall clock per
# epoch is far above the ~8s the item count implies, that overhead is the cause.
#
# sm_120 needs det_cu128 (torch 2.11+cu128); det is cu124 and has no sm_120
# kernels. scavenger is preemptible; --requeue plus the auto-resume below caps a
# preemption at save_freq epochs.

set -euo pipefail

SLUG="${1:?usage: scale_arm_blackwell.sbatch <e3_1k|e3_3k|e3_10k|e4_small|e4_large> [extra train args...]}"
shift || true

# Reciprocal of the guard in slurm/scale_arm_maize.sbatch. Without this,
# `sbatch slurm/scale_arm_blackwell.sbatch maize_e3_1k` finds the real file
# configs/config_maize_e3_1k.yaml, passes the -f check below, and runs
# train_sorghum_4m.py on maize data -- a 9-param/24-leaf ParamEmbed globbing
# '*_nc_cam.ply' against a tree that has none, which dies with "No valid samples
# found" only AFTER the scavenger queue wait and a scan of 105,000 folders.
case "${SLUG}" in
  maize_*) echo "REFUSING: '${SLUG}' is a maize arm. This launcher runs"
           echo "train_sorghum_4m.py; use slurm/scale_arm_maize.sbatch instead."
           exit 2 ;;
esac

cd /work/mech-ai-scratch/alloy/embodiedmae
mkdir -p logs

source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
conda activate /work/mech-ai-scratch/alloy/.conda/envs/det_cu128

CONFIG="configs/config_${SLUG}.yaml"
[ -f "${CONFIG}" ] || { echo "no such config: ${CONFIG}"; exit 2; }
CKPT_DIR="./outputs/${SLUG}/checkpoints"

echo "JobID   : ${SLURM_JOB_ID}"
echo "Slug    : ${SLUG}"
echo "Config  : ${CONFIG}"
echo "Node    : $(hostname)"
echo "Date    : $(date)"
nvidia-smi -L || true

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
[ "${NPROC}" -gt 0 ] 2>/dev/null || NPROC=4
echo "torchrun with ${NPROC} processes"

# Auto-resume: highest-epoch checkpoint by its NUMBER, not by sorting the path
# (the output dir name also contains '_', which shifts the field) -- and it must
# actually OPEN.
#
# A preempted job can be killed mid-write, leaving a truncated .pth that is the
# highest-numbered one and therefore exactly the file this picks. Resuming from
# it fails at torch.load with "failed finding central directory", after the
# queue wait and with no checkpoint written, so the run makes no progress and
# fails the same way every retry. Walk down until one loads.
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
    train_sorghum_4m.py --config "${CONFIG}" --batch_size 16 --num_workers 38 ${RESUME_ARG} "$@"

echo "Done at $(date)"
