#!/bin/bash
#SBATCH --job-name=scene_ckpts
#SBATCH --partition=scavenger
#SBATCH --account=mech-ai-scavenger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|l40s|h200|rtxpro6000
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=logs/scene_ckpts_%j.out
#SBATCH --error=logs/scene_ckpts_%j.out

# Clean + occluded val, with PC F1 / precision / recall at 0.01/0.02/0.03, for
# every saved checkpoint of the occluded-scene arms (eval/eval_scene_ckpts.py).
# Incremental and requeue-safe: rows already in the JSON are skipped.
#   sbatch slurm/scene_ckpts.sbatch                       # score only
#   sbatch slurm/scene_ckpts.sbatch --wandb-runs sm_scene_s1 sm_scene_noparam_s1
# --wandb-runs only for runs that have FINISHED (W&B cannot resume a live run).
# GPU constraint: myenv is CUDA 13, no kernels for the V100s.

set -euo pipefail
REPO=/work/mech-ai-scratch/yongyun/embodiedmae4m_distilation
PYTHON_BIN=/work/mech-ai-scratch/yongyun/envs/myenv/bin/python
cd "${REPO}"
mkdir -p logs reports
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export WANDB__SERVICE_WAIT=600 WANDB_INIT_TIMEOUT=600
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "JobID ${SLURM_JOB_ID} (restart ${SLURM_RESTART_COUNT:-0})  $(hostname)  $(date)"
nvidia-smi -L || true

"${PYTHON_BIN}" eval/eval_scene_ckpts.py \
  --runs sm_scene_s1 sm_scene_off_s1 sm_scene_noparam_s1 \
  --out reports/scene_ckpts.json --workers 8 "$@"
echo "Done at $(date)"
