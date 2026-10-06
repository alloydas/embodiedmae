#!/bin/bash
#SBATCH --job-name=scene_oracle
#SBATCH --partition=scavenger
#SBATCH --account=mech-ai-scavenger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|l40s|h200|rtxpro6000
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=logs/scene_oracle_%j.out
#SBATCH --error=logs/scene_oracle_%j.out

# Occluded val under uniform AND oracle (neighbour-first) test-time masking for
# the occluded-scene arms' final checkpoints (eval/eval_scene_oracle.py).
# Incremental and requeue-safe: runs already in the JSON are skipped, so it can
# be re-submitted with more --runs as arms finish.
#   sbatch slurm/scene_oracle.sbatch                       # default arms below
#   sbatch slurm/scene_oracle.sbatch --runs sm_scene_mix_s1 maize_scene_mix_s1
# GPU constraint: myenv is CUDA 13, no kernels for the V100s.

set -euo pipefail
REPO=/work/mech-ai-scratch/yongyun/embodiedmae4m_distilation
PYTHON_BIN=/work/mech-ai-scratch/yongyun/envs/myenv/bin/python
cd "${REPO}"
mkdir -p logs reports
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "JobID ${SLURM_JOB_ID} (restart ${SLURM_RESTART_COUNT:-0})  $(hostname)  $(date)"
nvidia-smi -L || true

RUNS=(sm_scene_s1 sm_scene_struct_s1 sm_scene_off_s1 maize_scene_s1 maize_scene_struct_s1)
if [ "${1:-}" = "--runs" ]; then shift; RUNS=(); while [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; do RUNS+=("$1"); shift; done; fi
"${PYTHON_BIN}" eval/eval_scene_oracle.py --runs "${RUNS[@]}" \
  --out reports/scene_oracle.json --workers 8 "$@"
echo "Done at $(date)"
