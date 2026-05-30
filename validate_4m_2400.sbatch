#!/bin/bash
#SBATCH --job-name=val4m2400
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/val4m2400_%j.out
#SBATCH --error=logs/val4m2400_%j.err

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

python validate_4m.py \
    --checkpoint outputs/4m_run_v3/checkpoints/checkpoint_epoch_2400.pth \
    --output_dir slide_dump_v3_2400 \
    --config config_4m.yaml \
    --mask_ratio 0.4 \
    --num_samples 15 \
    --batch_size 1 \
    --num_workers 4 \
    --seed 42

echo "Done at $(date)"
