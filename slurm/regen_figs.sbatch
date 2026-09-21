#!/bin/bash
#SBATCH --job-name=regenfigs
#SBATCH --partition=scavenger
#SBATCH --account=mech-ai-scavenger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:40:00
#SBATCH --output=logs/regenfigs_%j.out
#SBATCH --error=logs/regenfigs_%j.err
set -uo pipefail
cd /work/mech-ai-scratch/alloy/embodiedmae
source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU" in *"PRO 6000"*) conda activate /work/mech-ai-scratch/alloy/.conda/envs/det_cu128 ;;
                *) conda activate det ;; esac
export PYTHONUNBUFFERED=1
echo "node $(hostname) | GPU ${GPU}"
python figures/regen_figures.py
