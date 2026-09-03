#!/bin/bash
# Distillation smoke + batch-size sweep on ONE RTX PRO 6000 Blackwell.
# Phase 1 validates the full distill path end-to-end (teacher load -> XM train ->
# per-source val -> viz -> checkpoint) using --max_steps so we don't walk all
# 105k folders. Phase 2 sizes batch_size against the REAL script, since the
# Aug-19 sweep measured the pretrain model with no frozen teacher.
#SBATCH --job-name=distsweep
#SBATCH --partition=scavenger
#SBATCH --account=mech-ai-scavenger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G
#SBATCH --time=01:00:00
#SBATCH --output=logs/distsweep_%j.out
#SBATCH --error=logs/distsweep_%j.err

set -uo pipefail
cd /work/mech-ai-scratch/alloy/embodiedmae
mkdir -p logs
source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
conda activate /work/mech-ai-scratch/alloy/.conda/envs/det_cu128
export PYTHONUNBUFFERED=1

CFG=configs/config_4m_distill_15k_all_smoke.yaml
echo "Node   : $(hostname)"
echo "Config : ${CFG}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
python -c "import torch;print('torch',torch.__version__,'cc',torch.cuda.get_device_capability(0))"

########## Phase 1 — end-to-end smoke ##########
echo ""
echo "################ PHASE 1: end-to-end smoke (bs=16, 30 steps) ################"
OUT=./outputs/_distill_smoke_${SLURM_JOB_ID}
python train_sorghum_4m_distill.py --config "${CFG}" --world_size 1 \
  --batch_size 16 --num_workers 16 --max_steps 30 --epochs 1 \
  --val_max_batches 2 --output_dir "${OUT}" --no_wandb
echo "SMOKE_EXIT=$?"
echo "--- artifacts written ---"
find "${OUT}" -type f | head -20

########## Phase 2 — batch-size sweep ##########
echo ""
echo "################ PHASE 2: batch-size sweep (real script, 12 steps each) ################"
printf "%7s %10s %12s %10s\n" "bs/GPU" "peak MiB" "peak/95GiB" "status"
for BS in 8 16 24 32 48; do
  SWOUT=./outputs/_distill_sweep_${SLURM_JOB_ID}_bs${BS}
  CSV=logs/distsweep_util_${SLURM_JOB_ID}_bs${BS}.csv
  nvidia-smi --query-gpu=memory.used --format=csv,noheader -l 2 > "${CSV}" &
  SMI=$!
  python train_sorghum_4m_distill.py --config "${CFG}" --world_size 1 \
    --batch_size ${BS} --num_workers 16 --max_steps 12 --epochs 1 \
    --val_max_batches 1 --output_dir "${SWOUT}" --no_wandb \
    > logs/distsweep_${SLURM_JOB_ID}_bs${BS}.log 2>&1
  RC=$?
  kill ${SMI} 2>/dev/null; sleep 1
  PEAK=$(awk '{gsub(/ MiB/,"");if($1+0>m)m=$1+0}END{print m+0}' "${CSV}")
  if [ ${RC} -ne 0 ]; then
    if grep -qi "out of memory" logs/distsweep_${SLURM_JOB_ID}_bs${BS}.log; then ST=OOM; else ST="ERR(${RC})"; fi
  else ST=ok; fi
  printf "%7s %10s %11s%% %10s\n" "${BS}" "${PEAK}" "$(awk -v p=${PEAK} 'BEGIN{printf "%.0f", 100*p/97280}')" "${ST}"
  rm -rf "${SWOUT}"
  [ "${ST}" = "OOM" ] && break
done
echo ""
echo "SWEEP_DONE"
