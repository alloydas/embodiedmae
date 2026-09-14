#!/bin/bash
# EPOCH-0 BASELINE for the two ->params arms, on CPU.
#
# The GPU route is blocked in both directions: nova rejects a GPU ask with
# AssocGrpGRES while the lab sits at its 18-GPU cap, and every scavenger GPU node
# is draining, which put its estimated start a day out. This asks for no GPU at
# all, so neither limit applies and it schedules against the idle CPU pool.
#
# One model load covers both arms: the warm start is the same teacher_final.pth
# for rgb and pc, and the two configs agree on everything the measurement depends
# on (batch_size 16, val_max_batches 100, targets [text], same data_root), so
# --sources rgb,pc scores both against one checkpoint.
#
# Runtime is the open question -- roughly 20 s/sample at batch 1 in the earlier
# CPU example dumps, better batched, so expect ~8-15 h for 2 x 1600 samples.
# Hence the long walltime, the sharding, and the partial JSON written after every
# shard: a run that is cut short still yields a usable number over the samples it
# reached, and the shard spread says how stable that number is.
#SBATCH --job-name=ws_cpu
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/ws_cpu_%j.out
#SBATCH --error=logs/ws_cpu_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alloydas@iastate.edu

set -euo pipefail

cd /work/mech-ai-scratch/alloy/embodiedmae
mkdir -p logs results

source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
conda activate /work/mech-ai/alloy/.conda/envs/det

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32

echo "JobID : ${SLURM_JOB_ID}"
echo "Node  : $(hostname)"
echo "CPUs  : ${SLURM_CPUS_PER_TASK}"
echo "Date  : $(date)"

python eval_warmstart_cpu.py \
  --config configs/config_4m_distill_15k_rgb2text.yaml \
  --sources rgb,pc \
  --shards 10 \
  --threads 32 \
  --out results/warmstart_text_epoch0.json

echo "Done at $(date)"
