#!/bin/bash
# Does structured masking merely train slower, or top out lower? The 600-epoch
# pair left maskoff still climbing (+0.06 F1@0.03 over its last 150 epochs)
# while every structured arm had flattened. These four runs take the two ends of
# that comparison to 1 000 epochs (94 000 steps), single cosine from scratch,
# two seeds each: maskoff should settle around 0.84-0.86, maskboth should stay
# flat near 0.61 -- which would make it a ceiling, not a pace.
#
#   0-1  maskoff  s1, s2   structured masking off (uniform token masking)
#   2-3  maskboth s1, s2   blobs + edge weighting (the old default)
#
# Two at a time = 8 GPUs. ~16 h per run, so the four take ~1.5 days.
#   sbatch train_4m_paper1000.sbatch
#SBATCH --account=mech-ai
#SBATCH --job-name=embmae4m_p1000
#SBATCH --array=0-3%2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=228G
#SBATCH --time=1-06:00:00
#SBATCH --output=logs/embmae4m_p1000_%A_%a.out
#SBATCH --error=logs/embmae4m_p1000_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yongyun@iastate.edu

set -euo pipefail

cd /work/mech-ai-scratch/yongyun/embodiedmae4m_whole_to_whole
mkdir -p logs

CONFIGS=(
    config_4m_paper1000_maskoff_s1.yaml
    config_4m_paper1000_maskoff_s2.yaml
    config_4m_paper1000_maskboth_s1.yaml
    config_4m_paper1000_maskboth_s2.yaml
)


# CONFIG= overrides the array entry, so a single odd run can reuse this script.
CONFIG="${CONFIG:-${CONFIGS[${SLURM_ARRAY_TASK_ID:-0}]}}"

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
        --config "${CONFIG}"

echo "Done at $(date)"
