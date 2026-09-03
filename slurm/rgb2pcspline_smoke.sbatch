#!/bin/bash
#SBATCH --job-name=r2pssmoke
#SBATCH --partition=scavenger
#SBATCH --account=mech-ai-scavenger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=120G
#SBATCH --time=00:40:00
#SBATCH --output=logs/r2pssmoke_%j.out
#SBATCH --error=logs/r2pssmoke_%j.err
set -uo pipefail
cd /work/mech-ai-scratch/alloy/embodiedmae
source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU" in *"PRO 6000"*) conda activate /work/mech-ai-scratch/alloy/.conda/envs/det_cu128 ;;
                *) conda activate det ;; esac
export PYTHONUNBUFFERED=1
echo "node $(hostname) | GPU ${GPU}"
python train_sorghum_4m_distill.py \
  --config configs/config_4m_distill_rgb2pcspline_smoke.yaml --world_size 1 \
  --batch_size 8 --num_workers 16 --max_steps 20 --epochs 1 \
  --val_max_batches 3 --output_dir ./outputs/_r2ps_smoke_${SLURM_JOB_ID} --no_wandb
echo "SMOKE_EXIT=$?"
