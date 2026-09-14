#!/bin/bash
# RGB -> SPLINE PARAMETERS specialist distillation on the FULL Sorghum_15K.
# Config: configs/config_4m_distill_15k_rgb2text.yaml   (target: [text] only, source_mask_ratio 0.5)
#
# Recovers the plant's procedural generation parameters from a single RGB render alone.
# `mean_gen` in this run's history IS param_mae_masked and nothing else, so the
# headline number is the metric with no Chamfer mixed in.
#
# 8x A100-80GB instead of 8x RTX PRO 6000. This is the SAME recipe as the sibling
# arms -- batch_size 16 x 8 GPUs x accum 1 = global batch 128, no accumulation --
# just on different silicon, because the two rtx_pro_6000 nodes are the most
# contested resource on the cluster while A100s are plentiful.
#
# Verified before writing this, not assumed:
#   * SLURM gres 'a100' = A100 80GB PCIe, 81920 MiB, cc 8.0 (job 12248615 ran
#     nvidia-smi on nova22-gpu-2). gres 'a100-pcie' is the 40 GB part -- do NOT
#     use it, bs16 will not fit.
#   * Device peak is ~2325 + 3260 x batch_size MiB, so bs16 = ~54.5 GiB = 67% of
#     an 80 GB card. L40S (46068 MiB) cannot hold bs16 and was ruled out.
#   * The code is pure FP32 -- no autocast, no bfloat16, no GradScaler, no
#     torch.compile, no flash-attn, no custom CUDA extensions -- so sm_80 needs no
#     code change, only the older 'det' env instead of the Blackwell det_cu128.
#
# 96 CPUs / 450G: the 8x A100 nodes cap at 96 CPUs and 500000M, and asking >=100
# CPUs or >=490G silently pushes placement onto the scarce 1TB 'amp' nodes and
# costs ~2 days. num_workers drops 14 -> 10 to stay inside 96 (8 x 10 + 8 = 88).
#
# Loss balance was checked before submitting (job 12247376, 20 steps each):
# recon 0.047 against feat 0.040-0.049, i.e. the parameter objective sits level
# with the feature-distillation term rather than being drowned by it, so
# spline_loss_weight stays at the teacher's own 5.0.
#SBATCH --job-name=rgb2txtA100
#SBATCH --partition=scavenger
#SBATCH --account=mech-ai-scavenger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=450G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
# Append on requeue: SLURM truncates by default, which erased the epoch 1-3
# history when this job was first preempted.
#SBATCH --open-mode=append
#SBATCH --output=logs/rgb2textA100_%j.out
#SBATCH --error=logs/rgb2textA100_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=alloydas@iastate.edu

set -euo pipefail

cd /work/mech-ai-scratch/alloy/embodiedmae
mkdir -p logs

source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU" in
  *"PRO 6000"*) conda activate /work/mech-ai-scratch/alloy/.conda/envs/det_cu128 ;;
  *)            conda activate /work/mech-ai/alloy/.conda/envs/det ;;
esac
echo "GPU: ${GPU}"

CONFIG=configs/config_4m_distill_15k_rgb2text.yaml
CKPT_DIR=./outputs/4m_distill_15k_rgb2text/checkpoints

echo "JobID         : ${SLURM_JOB_ID}"
echo "Node          : $(hostname)"
echo "Date          : $(date)"
echo "GPUs visible  : ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count(), 'cc', torch.cuda.get_device_capability(0))"

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$((10000 + RANDOM % 50000))
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN
# Unbuffered: a preempted job otherwise logs nothing at all.
export PYTHONUNBUFFERED=1

# SLURM_GPUS_ON_NODE is a COUNT (e.g. "8"), not a list — use it directly.
if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
  NPROC=${SLURM_GPUS_ON_NODE}
else
  NPROC=$(echo "${CUDA_VISIBLE_DEVICES//,/ }" | wc -w)
fi
[ "${NPROC}" -gt 0 ] 2>/dev/null || NPROC=8
echo "Launching torchrun with ${NPROC} processes"

# Auto-resume from the highest-epoch checkpoint (survives requeue/preemption).
RESUME_ARG=""
if [ -d "${CKPT_DIR}" ]; then
  LATEST=$(for f in "${CKPT_DIR}"/checkpoint_epoch_*.pth; do
             [ -e "$f" ] || continue
             b=$(basename "$f" .pth); echo "${b##*_} $f"
           done | sort -n | tail -1 | cut -d' ' -f2-)
  if [ -n "${LATEST:-}" ]; then
    echo "Resuming from latest checkpoint: ${LATEST}"
    RESUME_ARG="--resume ${LATEST}"
  fi
fi

srun --kill-on-bad-exit=1 \
  torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC} \
    train_sorghum_4m_distill.py --config "${CONFIG}" --num_workers 10 ${RESUME_ARG}

echo "Done at $(date)"
