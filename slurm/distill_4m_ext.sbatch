#!/bin/bash
#SBATCH --job-name=4m_distill_ext
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/distill_ext_%j.out
#SBATCH --error=logs/distill_ext_%j.err

set -euo pipefail

cd /work/mech-ai-scratch/alloy/embodiedmae
mkdir -p logs

source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
conda activate det

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "JobID         : ${SLURM_JOB_ID}"
echo "Node          : $(hostname)"
echo "Date          : $(date)"
echo "GPUs visible  : ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L || true

# Extension run: resume 4m_distill_v1 from epoch 100, warm-restart cosine to epoch 200.
python train_sorghum_4m_distill.py \
    --config configs/config_4m_distill_long2.yaml \
    --world_size 1

echo "Done at $(date)"
