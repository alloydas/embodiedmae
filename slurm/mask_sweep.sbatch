#!/bin/bash
#SBATCH --job-name=masksweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/masksweep_%j.out
#SBATCH --error=logs/masksweep_%j.err

set -euo pipefail

cd /work/mech-ai-scratch/alloy/embodiedmae
mkdir -p logs

source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
conda activate det

echo "JobID         : ${SLURM_JOB_ID}"
echo "Node          : $(hostname)"
echo "Date          : $(date)"
echo "GPUs visible  : ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L || true

python mask_ratio_sweep.py \
    --checkpoint outputs/4m_run_v3/checkpoints/checkpoint_epoch_2400.pth \
    --config configs/config_4m.yaml \
    --output_dir vis_mask_sweep \
    --num_samples 8 \
    --seed 1234

echo "Done at $(date)"
