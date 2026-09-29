#!/bin/bash
# Continue finished 30k-step runs to a longer optimizer-step budget.
#
# The 30k-step selection runs stop before the jump in PC recall that the long
# occ_strong_10k run showed between ~140k and ~165k steps (recall@0.03
# 0.57 -> 0.74; 0.86 by 350k), so their rankings are pre-transition. This
# resumes each run from its newest readable checkpoint with --max_steps raised;
# the cosine schedule restarts over the new length (warm restart), identically
# for every run, so comparisons between continued runs stay like-for-like.
#
#   sbatch --array=0-5%1 --export=ALL,RUNS=params_v1_seed1:recipe_b_ls020_noise1x:... \
#          train_4m_continue.sbatch
#   TARGET_STEPS=400000 to change the budget (default 300000).
#
# Re-submitting the same run after a timeout resumes from its newest checkpoint.
#SBATCH --account=mech-ai
#SBATCH --job-name=embmae4m_cont
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=228G
#SBATCH --time=6-00:00:00
#SBATCH --output=logs/embmae4m_cont_%A_%a.out
#SBATCH --error=logs/embmae4m_cont_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yongyun@iastate.edu

set -euo pipefail

cd /work/mech-ai-scratch/yongyun/embodiedmae4m_whole_to_whole
mkdir -p logs

# Colon-separated: sbatch --export splits its argument on commas, so a comma
# list silently arrives as its first item only (every other array task then
# fails on an unbound index). Commas are still accepted for direct runs.
IFS=':,' read -r -a RUN_LIST <<< "${RUNS:?set RUNS=slug1:slug2:...}"
SLUG="${RUN_LIST[${SLURM_ARRAY_TASK_ID:-0}]}"
CONFIG="config_4m_${SLUG}.yaml"
TARGET_STEPS="${TARGET_STEPS:-300000}"
[ -f "${CONFIG}" ] || { echo "ERROR: missing ${CONFIG}"; exit 1; }

module load micromamba
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/.micromamba}"
set +u
eval "$(micromamba shell hook --shell=bash)"
micromamba activate /work/mech-ai-scratch/yongyun/envs/myenv
set -u

PYTHON_BIN="/work/mech-ai-scratch/yongyun/envs/myenv/bin/python"
if [ ! -x "${PYTHON_BIN}" ]; then
    echo "ERROR: Python executable does not exist: ${PYTHON_BIN}"
    exit 1
fi

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN
# wandb service start-up can exceed the 30 s default on a busy node (killed
# recipe A, job 13278813_0, before its first step).
export WANDB__SERVICE_WAIT=300
export WANDB_INIT_TIMEOUT=300
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/xdg_${SLURM_JOB_ID}"
mkdir -p "${XDG_CACHE_HOME}"

echo "JobID         : ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${SLURM_ARRAY_TASK_ID:-0}"
echo "Account       : ${SLURM_JOB_ACCOUNT:-<unset>}"
echo "Node          : $(hostname)"
echo "Date          : $(date)"
echo "Config        : ${CONFIG}"
echo "CUDA devices  : ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L || true

# Fail fast on a bad occlusion / structured_mask block rather than after the
# multi-minute dataset scan on every rank.
"${PYTHON_BIN}" -c "
import yaml
from plant_occlusion import ProceduralOcclusion, StructuredMaskConfig
cfg = yaml.safe_load(open('${CONFIG}'))
occ = cfg.get('occlusion') or {}
print('structured_mask:', StructuredMaskConfig.from_dict(cfg['model'].get('structured_mask')))
print('occlusion      :', ProceduralOcclusion.from_config(occ))
if occ.get('eval_occlusion'):
    print('eval occlusion :', ProceduralOcclusion.from_config({**occ, **occ['eval_occlusion']}))
"

# Newest checkpoint that actually loads -- a job killed mid-save leaves a
# truncated newest file, and resuming from it would crash every retry.
RESUME=$("${PYTHON_BIN}" - "${SLUG}" <<'PYEOF'
import re, sys, torch
from pathlib import Path
d = Path('outputs') / sys.argv[1] / 'checkpoints'
cks = sorted(d.glob('checkpoint_epoch_*.pth'),
             key=lambda p: int(re.search(r'(\d+)', p.stem).group(1)), reverse=True)
for p in cks:
    try:
        torch.load(p, map_location='cpu', weights_only=False)['epoch']
        print(p); break
    except Exception as e:
        print(f'skip unreadable {p}: {e}', file=sys.stderr)
PYEOF
)
[ -n "${RESUME}" ] || { echo "ERROR: no readable checkpoint for ${SLUG}"; exit 1; }
echo "Run           : ${SLUG}"
echo "Resume from   : ${RESUME}"
echo "Target steps  : ${TARGET_STEPS}"
# Checkpoints are 1.3 GB each and /work/mech-ai-scratch runs near full, so the
# continuation saves every 100 epochs by default (4 per run, not 13).
echo "Save every    : ${SAVE_FREQ:-100} epochs"

NPROC=$("${PYTHON_BIN}" -c "import torch; print(torch.cuda.device_count())")
if [ "${NPROC}" -lt 1 ]; then
    echo "ERROR: PyTorch cannot see any CUDA GPUs."
    exit 1
fi
echo "Launching training with ${NPROC} GPU processes"

srun --export=ALL --kill-on-bad-exit=1 \
    "${PYTHON_BIN}" -m torch.distributed.run \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="${NPROC}" \
        train_sorghum_4m.py \
        --config "${CONFIG}" \
        --resume "${RESUME}" \
        --max_steps "${TARGET_STEPS}" \
        --save_freq "${SAVE_FREQ:-100}"

echo "Done at $(date)"
