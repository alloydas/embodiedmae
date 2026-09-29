#!/bin/bash
# EmbodiedMAE-4M recipe-selection sweep: four candidates that differ only in
# structured_mask.length_scale and the sensor-noise level, trained on the same
# 3,000 plants for the same 30,000 optimiser steps.
#
#   A  config_4m_recipe_a_ls035_noise1x.yaml    blobs 0.35, noise 1x
#   B  config_4m_recipe_b_ls020_noise1x.yaml    blobs 0.20, noise 1x
#   C  config_4m_recipe_c_ls020_noise0p5x.yaml  blobs 0.20, noise 0.5x
#   D  config_4m_recipe_d_ls020_noise2x.yaml    blobs 0.20, noise 2x
#
# All four:        sbatch train_4m_recipe.sbatch      (2 at a time: mech-ai has a
#                  17-GPU account cap shared with the rest of the lab)
# One of them:     sbatch --array=1 train_4m_recipe.sbatch
# Another config:  sbatch --array=0 --export=ALL,CONFIG=my_config.yaml train_4m_recipe.sbatch
#
# ~94 steps/epoch at 4 GPUs x batch 8, ~0.9 s/step -> ~8 h per candidate.
#SBATCH --account=mech-ai
#SBATCH --job-name=embmae4m_recipe
#SBATCH --array=0-3%2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=228G
#SBATCH --time=16:00:00
#SBATCH --output=logs/embmae4m_recipe_%A_%a.out
#SBATCH --error=logs/embmae4m_recipe_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yongyun@iastate.edu

set -euo pipefail

cd /work/mech-ai-scratch/yongyun/embodiedmae4m_whole_to_whole
mkdir -p logs

CONFIGS=(
    config_4m_recipe_a_ls035_noise1x.yaml
    config_4m_recipe_b_ls020_noise1x.yaml
    config_4m_recipe_c_ls020_noise0p5x.yaml
    config_4m_recipe_d_ls020_noise2x.yaml
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
