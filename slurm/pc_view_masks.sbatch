#!/bin/bash
#SBATCH --job-name=pcview_masks
#SBATCH --partition=scavenger
#SBATCH --account=mech-ai-scavenger
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Visibility masks for the 1-view vs 3-view point-cloud evaluation: which points
# of every sorghum cloud cameras 00, 01 and 02 see (eval/pc_view_masks.py).
# CPU only. It ray-casts each plant's source mesh, about 4 s per plant per
# worker, so ~30 min for all 15,000 plants on 32 CPUs.
#
#   sbatch slurm/pc_view_masks.sbatch
#   sbatch --dependency=afterok:<this job> slurm/pc_view_eval.sbatch
#
# No GPU, and a small --mem: a CPU job that asks for a node's whole memory
# blocks every GPU on that node from scavenger (see CLAUDE.md).

set -euo pipefail

cd /work/mech-ai-scratch/alloy/embodiedmae
mkdir -p logs

source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
conda activate /work/mech-ai/alloy/.conda/envs/det
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1

echo "JobID : ${SLURM_JOB_ID:-none}"
echo "Node  : $(hostname)"
echo "Date  : $(date)"

python eval/pc_view_masks.py --workers "${SLURM_CPUS_PER_TASK:-32}" --validate 20 "$@"

echo "Done at $(date)"
