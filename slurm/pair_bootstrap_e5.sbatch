#!/bin/bash
#SBATCH --job-name=pairboot_e5
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# E5 view regime: paired bootstraps over plants of the three view-band arms against the
# all-view reference, from the feature caches the E5 probe wrote. No GPU.
#   sbatch --dependency=afterok:<probe job> slurm/pair_bootstrap_e5.sbatch
set -euo pipefail
cd /work/mech-ai-scratch/alloy/embodiedmae
source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
conda activate /work/mech-ai/alloy/.conda/envs/det
export PYTHONUNBUFFERED=1 LOKY_MAX_CPU_COUNT=8 OMP_NUM_THREADS=1
J="${SLURM_JOB_ID:-manual}"
python eval/probe_pair_bootstrap.py --species sorghum \
  --a e5_topdown_d2 e5_horizon_d2 e5_single_d2 --b e2_pcrgbdt_d2 --out reports/e5_delta_d2_${J}.csv
echo "Done at $(date) -> reports/e5_delta_d2_${J}.csv"
