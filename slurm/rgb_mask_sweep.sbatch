#!/bin/bash
#SBATCH --job-name=rgbmask
#SBATCH --partition=scavenger
#SBATCH --account=mech-ai-scavenger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=120G
#SBATCH --time=00:40:00
#SBATCH --output=logs/rgbmask_%j.out
#SBATCH --error=logs/rgbmask_%j.err
set -uo pipefail
cd /work/mech-ai-scratch/alloy/embodiedmae
source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU" in *"PRO 6000"*) conda activate /work/mech-ai-scratch/alloy/.conda/envs/det_cu128 ;;
                *) conda activate det ;; esac
export PYTHONUNBUFFERED=1
echo "node $(hostname) | GPU ${GPU}"
python rgb_mask_sweep.py
