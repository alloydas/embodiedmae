#!/bin/bash
# Single-source cross-modal distillation. Usage:
#   sbatch --gres=gpu:a40:1 distill_src.sbatch config_4m_distill_src_pc.yaml
#SBATCH --job-name=distill_src
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/distill_src_%j.out
#SBATCH --error=logs/distill_src_%j.err

set -euo pipefail
cd /work/mech-ai-scratch/alloy/embodiedmae
mkdir -p logs

source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
conda activate det

CONFIG="${1:?usage: sbatch distill_src.sbatch <config.yaml>}"
echo "JobID  : ${SLURM_JOB_ID}"
echo "Node   : $(hostname)"
echo "Config : ${CONFIG}"
echo "GPU    : ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L || true

python train_sorghum_4m_distill.py --config "${CONFIG}" --world_size 1

echo "Done at $(date)"
