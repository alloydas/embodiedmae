#!/bin/bash
#SBATCH --job-name=cmpfigs
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/cmpfigs_%j.out
#SBATCH --error=logs/cmpfigs_%j.err

set -euo pipefail

cd /work/mech-ai-scratch/alloy/embodiedmae
mkdir -p logs

source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
conda activate det

echo "JobID : ${SLURM_JOB_ID}"
echo "Node  : $(hostname)"
nvidia-smi -L || true

# Pin the checkpoint by copying it: the live training run overwrites
# best_model.pth whenever val improves, and the figures must all come from
# one known epoch.
PIN=./outputs/4m_pretrain_15k/best_model_pinned.pth
if [ ! -f "${PIN}" ]; then
  cp ./outputs/4m_pretrain_15k/best_model.pth "${PIN}"
  echo "Pinned checkpoint → ${PIN}"
fi

python -u make_comparison_figs.py \
  --checkpoint "${PIN}" \
  --out_dir ./figures_compare \
  --num_recon 6 \
  --n_param_samples 400 \
  --batch_size 4 \
  --num_workers 8

echo "Done at $(date)"
