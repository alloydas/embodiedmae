#!/bin/bash
#SBATCH --job-name=maize_row_ft
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alloydas@iastate.edu

# Maize ROW training (Regime III): train_maize_row_4m.py on the rover row renders.
#
#   sbatch slurm/train_maize_row.sbatch                                   # configs/config_maize_row_ft.yaml
#   sbatch --job-name=maize_row_iso slurm/train_maize_row.sbatch configs/config_maize_row_iso.yaml
#   sbatch --time=00:30:00 slurm/train_maize_row.sbatch configs/config_maize_row_ft.yaml --smoke 8
#
# Auto-resumes from the newest checkpoint of the config's output_dir (so a requeue or a
# resubmission continues). Env is picked from the GPU granted (det_cu128 on Blackwell).

set -euo pipefail
CONFIG="${1:-configs/config_maize_row_ft.yaml}"
shift || true
cd /work/mech-ai-scratch/alloy/embodiedmae
mkdir -p logs outputs
source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
case "${GPU_NAME}" in
  *"RTX PRO 6000"*|*Blackwell*) CONDA_ENV=/work/mech-ai-scratch/alloy/.conda/envs/det_cu128 ;;
  *)                            CONDA_ENV=/work/mech-ai/alloy/.conda/envs/det ;;
esac
conda activate "${CONDA_ENV}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 NCCL_DEBUG=WARN
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-600}"
export MASTER_ADDR=$(hostname -s) MASTER_PORT=$((10000 + RANDOM % 50000))

OUT=$(python -c "import yaml,sys; print(yaml.safe_load(open('${CONFIG}'))['checkpointing']['output_dir'])")
RESUME=""
if [ -d "${OUT}/checkpoints" ]; then
  LATEST=$(ls -1 "${OUT}"/checkpoints/checkpoint_epoch_*.pth 2>/dev/null | sort -t_ -k3 -n | tail -1 || true)
  [ -n "${LATEST}" ] && RESUME="--resume ${LATEST}"
fi
NPROC="${SLURM_GPUS_ON_NODE:-$(echo "${CUDA_VISIBLE_DEVICES//,/ }" | wc -w)}"
echo "JobID ${SLURM_JOB_ID:-none}  node $(hostname)  GPU ${GPU_NAME} x${NPROC}  env ${CONDA_ENV}"
echo "Config ${CONFIG}  out ${OUT}  ${RESUME}  extra: $*"
srun --kill-on-bad-exit=1 torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
  train_maize_row_4m.py --config "${CONFIG}" --num_workers "$(( SLURM_CPUS_PER_TASK / NPROC ))" ${RESUME} "$@"
echo "Done at $(date)"
