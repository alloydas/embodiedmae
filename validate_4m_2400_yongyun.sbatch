#!/bin/bash
#SBATCH --job-name=val4m_cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/val4m_cpu_%j.out
#SBATCH --error=logs/val4m_cpu_%j.err

set -euo pipefail

cd /work/mech-ai-scratch/yongyun/embodiedmae4m
mkdir -p logs

PYTHON_BIN="/work/mech-ai-scratch/yongyun/envs/myenv/bin/python"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONFAULTHANDLER=1

echo "JobID : ${SLURM_JOB_ID}"
echo "Node  : $(hostname)"
echo "Date  : $(date)"

srun "${PYTHON_BIN}" validate_4m.py \
    --checkpoint outputs/4m_run_v5_qal_fixed_mask_sampling_threshold_0.01_normalize_first_then_sample/best_model.pth \
    --output_dir outputs/4m_run_v5_qal_fixed_mask_sampling_threshold_0.01_normalize_first_then_sample/test_cpu_best_mask075 \
    --config config_4m.yaml \
    --split test \
    --mask_ratio 0.75 \
    --num_samples 12 \
    --batch_size 1 \
    --num_workers 4 \
    --device cpu \
    --seed 42

echo "Done at $(date)"
