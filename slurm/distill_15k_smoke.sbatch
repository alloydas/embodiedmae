#!/bin/bash
# Short smoke test for the 15k distillation configs: 1 GPU, ~10 min, with GPU
# utilisation/memory sampling so we can size batch_size and pick a GPU type.
# Submit with:  sbatch --export=ALL,CFG=<config> [--gres=...] slurm/distill_15k_smoke.sbatch
#SBATCH --job-name=distsmoke
#SBATCH --partition=scavenger
#SBATCH --account=mech-ai-scavenger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=logs/distsmoke_%j.out
#SBATCH --error=logs/distsmoke_%j.err

set -uo pipefail
cd /work/mech-ai-scratch/alloy/embodiedmae
mkdir -p logs
source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
conda activate det

CFG=${CFG:-configs/config_4m_distill_15k_all.yaml}
BS=${BS:-16}
OUT=/tmp/distsmoke_${SLURM_JOB_ID}
echo "Node   : $(hostname)"
echo "Config : ${CFG}   batch_size/GPU: ${BS}"
nvidia-smi -L || true
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv || true

# Sample utilisation + memory every 5s for the life of the run.
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used \
           --format=csv,noheader -l 5 > logs/gpuutil_${SLURM_JOB_ID}.csv &
SMI_PID=$!

timeout 600 python train_sorghum_4m_distill.py \
  --config "${CFG}" --world_size 1 --batch_size "${BS}" --output_dir "${OUT}" \
  --num_workers 16 --val_max_batches 5 --no_wandb
RC=$?
kill ${SMI_PID} 2>/dev/null || true
echo "EXIT=${RC}  (124 == ran clean until the timeout killed it)"

echo "=== GPU utilisation summary ==="
awk -F', ' 'NR>2{gsub(/ %/,"",$2); gsub(/ MiB/,"",$4);
             u+=$2; n++; if($2>mu)mu=$2; if($4+0>mm)mm=$4+0}
            END{if(n)printf "samples=%d  mean_util=%.1f%%  peak_util=%d%%  peak_mem=%d MiB\n",n,u/n,mu,mm}' \
    logs/gpuutil_${SLURM_JOB_ID}.csv
