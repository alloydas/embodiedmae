#!/bin/bash
# RGB -> point-cloud specialist distillation, 500 epochs, single A40.
# Usage:  sbatch slurm/distill_rgb2pc.sbatch
#SBATCH --job-name=distill_rgb2pc
#SBATCH --partition=nova
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=6-00:00:00
#SBATCH --output=logs/distill_rgb2pc_%j.out
#SBATCH --error=logs/distill_rgb2pc_%j.err

set -euo pipefail
cd /work/mech-ai-scratch/alloy/embodiedmae
mkdir -p logs

source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
conda activate det

CONFIG=configs/config_4m_distill_rgb2pc.yaml
echo "JobID  : ${SLURM_JOB_ID}"
echo "Node   : $(hostname)"
echo "Config : ${CONFIG}"
echo "GPU    : ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L || true

python train_sorghum_4m_distill.py --config "${CONFIG}" --world_size 1

echo "Done at $(date)"
