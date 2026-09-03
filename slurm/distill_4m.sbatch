#!/bin/bash
#SBATCH --job-name=4m_distill
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/distill_%j.out
#SBATCH --error=logs/distill_%j.err

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

# Single-GPU cross-modal distillation (config world_size: 1).
# For multi-GPU, request more GPUs above and launch via torchrun instead:
#   torchrun --standalone --nproc_per_node=4 train_sorghum_4m_distill.py --config configs/config_4m_distill.yaml
python train_sorghum_4m_distill.py \
    --config configs/config_4m_distill.yaml \
    --world_size 1

echo "Done at $(date)"
