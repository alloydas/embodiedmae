#!/bin/bash
#SBATCH --job-name=embmae4m
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100-pcie:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=428G
#SBATCH --time=8-00:00:00
#SBATCH --output=logs/embmae4m_%j.out
#SBATCH --error=logs/embmae4m_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alloydas@iastate.edu

set -euo pipefail

# ── Workdir ───────────────────────────────────────────────────────────────────
cd /work/mech-ai-scratch/alloy/embodiedmae
mkdir -p logs

# ── Conda env (det) ───────────────────────────────────────────────────────────
# Source whichever conda init you have on Nova
# source /work/mech-ai/alloy/.conda/etc/profile.d/conda.sh 2>/dev/null \
#   || source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
#   || { echo "Could not find conda.sh"; exit 1; }

source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh

conda activate det

# ── Diagnostics ───────────────────────────────────────────────────────────────
echo "JobID         : ${SLURM_JOB_ID}"
echo "Node          : $(hostname)"
echo "Date          : $(date)"
echo "GPUs visible  : ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count())"

# ── Distributed env ───────────────────────────────────────────────────────────
# Pick a free-ish port to avoid collisions with other jobs on the node
export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$((10000 + RANDOM % 50000))
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN
# Uncomment if Nova's IB/Ethernet config gives NCCL trouble:
# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=^docker,lo

# ── Launch ────────────────────────────────────────────────────────────────────
# Number of GPUs requested via --gres above:
NPROC=$(echo "${SLURM_GPUS_ON_NODE:-${CUDA_VISIBLE_DEVICES//,/ }}" | wc -w)
[ "${NPROC}" -gt 0 ] || NPROC=8
echo "Launching torchrun with ${NPROC} processes"

srun --kill-on-bad-exit=1 \
  torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=${NPROC} \
    train_sorghum_4m.py \
    --config config_4m.yaml

echo "Done at $(date)"
