#!/bin/bash
# EmbodiedMAE-4M point-cloud LOSS STUDY: QAL vs Chamfer vs Sinkhorn, each at
# several settings, on recipe B / seed 1 / 3,000 plants / 30,000 steps. The
# QAL t=0.01 alpha=100 arm is the existing params_v1_seed1 run.
# Two at a time (%2): the mech-ai GPU cap is shared with the rest of the lab.
#
#   0-3   QAL      threshold 0.005 / 0.02, alpha 30 / 300
#   4-6   Chamfer  pc_loss_weight 3 / 10 / 30
#   7-10  Sinkhorn only: blur 0.01 / 0.02 / 0.05 at weight 1, blur 0.02 at weight 3
#
# All:   sbatch train_4m_loss.sbatch
# Table: python compare_recipes.py --loss-study
#SBATCH --account=mech-ai
#SBATCH --job-name=embmae4m_loss
#SBATCH --array=0-10%2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=228G
#SBATCH --time=16:00:00
#SBATCH --output=logs/embmae4m_loss_%A_%a.out
#SBATCH --error=logs/embmae4m_loss_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yongyun@iastate.edu

set -euo pipefail

cd /work/mech-ai-scratch/yongyun/embodiedmae4m_whole_to_whole
mkdir -p logs

CONFIGS=(
    config_4m_loss_q_t005.yaml
    config_4m_loss_q_t020.yaml
    config_4m_loss_q_a030.yaml
    config_4m_loss_q_a300.yaml
    config_4m_loss_c_w03.yaml
    config_4m_loss_c_w10.yaml
    config_4m_loss_c_w30.yaml
    config_4m_loss_s_b010_w1.yaml
    config_4m_loss_s_b020_w1.yaml
    config_4m_loss_s_b050_w1.yaml
    config_4m_loss_s_b020_w3.yaml
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
