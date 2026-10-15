#!/bin/bash
# Depth reliance + stem-origin sensitivity of a scene run on occluded TEST (eval/depth_reliance.py).
#   sbatch slurm/depth_reliance.sbatch --species maize --run maize_scene_anyparam_stem_s1 --epoch 200 --out reports/depth_reliance_maize_stem.json
#SBATCH --job-name=depth_rel
#SBATCH --partition=scavenger
#SBATCH --account=mech-ai-scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|l40s|h200|rtxpro6000
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=logs/depth_rel_%j.out
cd /work/mech-ai-scratch/yongyun/embodiedmae4m_distilation
export PYTHONUNBUFFERED=1
nvidia-smi -L || true
/work/mech-ai-scratch/yongyun/envs/myenv/bin/python eval/depth_reliance.py "$@"
