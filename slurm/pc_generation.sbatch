#!/bin/bash
# Score an any-to-any run on occluded TEST scenes: no card / predicted card fed back /
# true card / card only (eval/pc_generation.py). Arguments pass straight through:
#   sbatch slurm/pc_generation.sbatch --species sorghum --run sm_scene_anyparam_pre_s1 --out reports/pc_generation_sorghum.json
#SBATCH --job-name=pc_gen
#SBATCH --partition=scavenger
#SBATCH --account=mech-ai-scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|l40s|h200|rtxpro6000
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=logs/pc_gen_%j.out
cd /work/mech-ai-scratch/yongyun/embodiedmae4m_distilation
export PYTHONUNBUFFERED=1
nvidia-smi -L || true
/work/mech-ai-scratch/yongyun/envs/myenv/bin/python eval/pc_generation.py "$@"
