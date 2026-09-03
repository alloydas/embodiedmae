#!/bin/bash
# Decisive pre-launch check: bs16 vs bs24 under the REAL 8-GPU DDP topology,
# including the val (4 sources) and viz legs where memory spikes. Single-GPU
# sweeps miss NCCL buffers, and torch-reserved misses cuDNN workspace, so this
# samples nvidia-smi device-wide while the real script runs.
#SBATCH --job-name=distconfirm
#SBATCH --partition=scavenger
#SBATCH --account=mech-ai-scavenger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_pro_6000:8
#SBATCH --cpus-per-task=360
#SBATCH --mem=1200G
#SBATCH --time=01:00:00
#SBATCH --output=logs/distconfirm_%j.out
#SBATCH --error=logs/distconfirm_%j.err

set -uo pipefail
cd /work/mech-ai-scratch/alloy/embodiedmae
mkdir -p logs
source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
conda activate /work/mech-ai-scratch/alloy/.conda/envs/det_cu128
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 NCCL_DEBUG=WARN
export MASTER_ADDR=$(hostname -s) MASTER_PORT=$((10000 + RANDOM % 50000))

CFG=configs/config_4m_distill_15k_all_smoke.yaml
echo "Node: $(hostname)"

for BS in 16 24; do
  echo ""
  echo "########## 8-GPU DDP, bs/GPU=${BS} (global $((BS*8))) ##########"
  OUT=./outputs/_distill_confirm_${SLURM_JOB_ID}_bs${BS}
  CSV=logs/distconfirm_util_${SLURM_JOB_ID}_bs${BS}.csv
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -l 1 > "${CSV}" &
  SMI=$!
  torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    train_sorghum_4m_distill.py --config "${CFG}" \
    --batch_size ${BS} --num_workers 44 --max_steps 60 --epochs 1 \
    --val_max_batches 3 --output_dir "${OUT}" --no_wandb \
    > logs/distconfirm_${SLURM_JOB_ID}_bs${BS}.log 2>&1
  RC=$?
  kill ${SMI} 2>/dev/null; sleep 1
  PEAK=$(awk -F', ' '{gsub(/ MiB/,"",$2); if($2+0>m)m=$2+0} END{print m+0}' "${CSV}")
  grep -h "^PEAKMEM" logs/distconfirm_${SLURM_JOB_ID}_bs${BS}.log | head -1
  if grep -qi "out of memory" logs/distconfirm_${SLURM_JOB_ID}_bs${BS}.log; then ST=OOM
  elif [ ${RC} -ne 0 ]; then ST="ERR(${RC})"; else ST=ok; fi
  echo "RESULT bs=${BS} peak_device=${PEAK}MiB ($(awk -v p=${PEAK} 'BEGIN{printf "%.0f", 100*p/97280}')% of card) status=${ST}"
  rm -rf "${OUT}"
done
echo ""
echo "CONFIRM_DONE"
