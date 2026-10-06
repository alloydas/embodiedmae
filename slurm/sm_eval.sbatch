#!/bin/bash
#SBATCH --job-name=sm_eval
#SBATCH --partition=scavenger
#SBATCH --account=mech-ai-scavenger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|l40s|h200|rtxpro6000
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=logs/sm_eval_%j.out
#SBATCH --error=logs/sm_eval_%j.out

# Score structured-masking runs on neighbour-occluded TEST inputs
# (eval/eval_occlusion.py). Every run sees identical inputs and masks.
#   RUNS="outputs/sm_off_s1:outputs/sm_nbr_s1" OUT=reports/sm_occlusion.json \
#     sbatch slurm/sm_eval.sbatch [extra eval args...]
# RUNS is colon-separated: sbatch --export splits on commas.

set -euo pipefail
cd /work/mech-ai-scratch/yongyun/embodiedmae4m_distilation
mkdir -p logs reports
IFS=':' read -r -a RUN_LIST <<< "${RUNS:?set RUNS=dir1:dir2}"
echo "Node $(hostname)  $(date)  runs: ${RUN_LIST[*]}"
nvidia-smi -L || true
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
/work/mech-ai-scratch/yongyun/envs/myenv/bin/python -u eval/eval_occlusion.py \
    --runs "${RUN_LIST[@]}" --ckpt "${CKPT:-last}" --num-plants "${NPLANTS:-1000}" \
    --num-workers 14 --per-sample --out "${OUT:?set OUT=reports/....json}" "$@"
echo "Done at $(date)"
