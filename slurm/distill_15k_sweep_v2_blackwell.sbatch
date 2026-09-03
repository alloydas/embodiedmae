#!/bin/bash
# Corrected distill batch sweep. v1 read nvidia-smi memory.used at 2s over a
# 12-step run: too few samples, and the allocator's reserved pool had not reached
# steady state, so bs32 came out LOWER than bs24. This reads
# torch.cuda.max_memory_{allocated,reserved} from inside the process and times
# steady state with the first 10 steps discarded.
#SBATCH --job-name=distsweep2
#SBATCH --partition=scavenger
#SBATCH --account=mech-ai-scavenger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G
#SBATCH --time=01:30:00
#SBATCH --output=logs/distsweep2_%j.out
#SBATCH --error=logs/distsweep2_%j.err

set -uo pipefail
cd /work/mech-ai-scratch/alloy/embodiedmae
mkdir -p logs
source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
conda activate /work/mech-ai-scratch/alloy/.conda/envs/det_cu128
export PYTHONUNBUFFERED=1

CFG=configs/config_4m_distill_15k_all_smoke.yaml
CARD_MIB=97280
echo "Node : $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

printf "\n%7s %11s %11s %10s %11s %10s\n" \
       "bs/GPU" "alloc MiB" "resv MiB" "resv/card" "steady it/s" "status"
for BS in 8 16 24 32 40 48; do
  OUT=./outputs/_distill_sweep2_${SLURM_JOB_ID}_bs${BS}
  LOG=logs/distsweep2_${SLURM_JOB_ID}_bs${BS}.log
  python train_sorghum_4m_distill.py --config "${CFG}" --world_size 1 \
    --batch_size ${BS} --num_workers 16 --max_steps 50 --epochs 1 \
    --val_max_batches 1 --output_dir "${OUT}" --no_wandb > "${LOG}" 2>&1
  RC=$?
  LINE=$(grep -m1 "^PEAKMEM" "${LOG}")
  if [ -n "${LINE}" ]; then
    A=$(sed -E 's/.*alloc=([0-9]+)MiB.*/\1/'    <<<"${LINE}")
    R=$(sed -E 's/.*reserved=([0-9]+)MiB.*/\1/' <<<"${LINE}")
    S=$(sed -E 's/.*steady_it_s=([0-9.]+).*/\1/'<<<"${LINE}")
    ST=ok
  else
    A=-; R=-; S=-
    if grep -qi "out of memory" "${LOG}"; then ST=OOM; else ST="ERR(${RC})"; fi
  fi
  PCT=$([ "${R}" = "-" ] && echo "-" || awk -v r=${R} -v c=${CARD_MIB} 'BEGIN{printf "%.0f%%", 100*r/c}')
  printf "%7s %11s %11s %10s %11s %10s\n" "${BS}" "${A}" "${R}" "${PCT}" "${S}" "${ST}"
  rm -rf "${OUT}"
  [ "${ST}" = "OOM" ] && break
done
echo ""
echo "SWEEP2_DONE"
