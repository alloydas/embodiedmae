#!/bin/bash
# EmbodiedMAE-4M PAPER RUNS: structured masking on/off, PC loss family, params
# encoding -- all at the programme's budget so the rows can join the E1-E4 table.
#
# 56 400 optimizer steps = 600 epochs (94 steps/epoch, 3 000 plants, one view
# per plant per epoch), one cosine from scratch, global batch 32 (8 x 4 GPUs).
# PC F1@0.03 jumps from 0.60 to 0.73 between epochs 200 and 250 and flattens at
# 0.83-0.84 by 550-600, so arms are compared after that transition. This is a
# SHORT schedule next to the programme's 197 400 steps -- label it as such.
# Claim-carrying arms run at seeds 1 and 2 so a gap can be read against the
# run-to-run spread rather than assumed.
#
#   0-1  maskoff       structured masking off (uniform token masking)
#   2-3  maskedge      edge weighting only (length_scale 0.02 = white-noise field)
#   4-5  maskblob      blobs only (length_scale 0.20, center_bias 0)
#   6-7  maskboth      blobs + edge weighting -- also the QAL loss baseline
#   8    losschamfer   Chamfer at pc_loss_weight 10
#   9    losssinkhorn  Sinkhorn only, blur 0.02, 2 048 points
#   10-11 paramsv2     v2 params + pose/scale token, spline_loss_weight 0.6
#
# Two at a time = 8 GPUs, the cap this project keeps inside the shared mech-ai
# account. ~10 h per run, so the twelve take ~2.5 days at that concurrency.
#
# All twelve:   sbatch train_4m_paper.sbatch
# A subset:     sbatch --array=0-7%2 train_4m_paper.sbatch      (mask arms only)
# One arm:      sbatch --array=8 train_4m_paper.sbatch
# Timed out?    resubmit that index; the run restarts from scratch, so raise
#               --time instead if a run does not finish within 20 h.
#SBATCH --account=mech-ai
#SBATCH --job-name=embmae4m_paper
#SBATCH --array=0-11%2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=228G
#SBATCH --time=20:00:00
#SBATCH --output=logs/embmae4m_paper_%A_%a.out
#SBATCH --error=logs/embmae4m_paper_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yongyun@iastate.edu

set -euo pipefail

cd /work/mech-ai-scratch/yongyun/embodiedmae4m_whole_to_whole
mkdir -p logs

CONFIGS=(
    config_4m_paper_maskoff_s1.yaml
    config_4m_paper_maskoff_s2.yaml
    config_4m_paper_maskedge_s1.yaml
    config_4m_paper_maskedge_s2.yaml
    config_4m_paper_maskblob_s1.yaml
    config_4m_paper_maskblob_s2.yaml
    config_4m_paper_maskboth_s1.yaml
    config_4m_paper_maskboth_s2.yaml
    config_4m_paper_losschamfer_s1.yaml
    config_4m_paper_losssinkhorn_s1.yaml
    config_4m_paper_paramsv2_s1.yaml
    config_4m_paper_paramsv2_s2.yaml
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
