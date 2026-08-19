#!/bin/bash
#SBATCH --job-name=latent4m
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/latent4m_%j.out
#SBATCH --error=logs/latent4m_%j.err

set -euo pipefail

cd /work/mech-ai-scratch/yongyun/embodiedmae4m

LATENT_PYTHON="/work/mech-ai-scratch/yongyun/envs/myenv/bin/python"
LATENT_CHECKPOINT="${LATENT_CHECKPOINT:-outputs/4m_run_v5_qal_fixed_mask_sampling_threshold_0.01_normalize_first_then_sample/best_model.pth}"
LATENT_SPLIT="${LATENT_SPLIT:-val}"
LATENT_NUM_SAMPLES="${LATENT_NUM_SAMPLES:-1000}"
LATENT_BATCH_SIZE="${LATENT_BATCH_SIZE:-8}"
LATENT_NUM_WORKERS="${LATENT_NUM_WORKERS:-8}"
LATENT_POOL="${LATENT_POOL:-mean}"
LATENT_PROJECTION="${LATENT_PROJECTION:-pca}"
LATENT_RECALL_K="${LATENT_RECALL_K:-1 5}"
LATENT_CONFIG="${LATENT_CONFIG:-}"
LATENT_ONE_PER_GROUP="${LATENT_ONE_PER_GROUP:-0}"
LATENT_AMP="${LATENT_AMP:-1}"

LATENT_CHECKPOINT_PARENT="$(dirname "${LATENT_CHECKPOINT}")"
if [ "$(basename "${LATENT_CHECKPOINT_PARENT}")" = "checkpoints" ]; then
    LATENT_RUN_DIR="$(dirname "${LATENT_CHECKPOINT_PARENT}")"
else
    LATENT_RUN_DIR="${LATENT_CHECKPOINT_PARENT}"
fi
LATENT_CHECKPOINT_STEM="$(basename "${LATENT_CHECKPOINT}" .pth)"
LATENT_OUTPUT_DIR="${LATENT_OUTPUT_DIR:-${LATENT_RUN_DIR}/modality_alignment_${LATENT_CHECKPOINT_STEM}}"

if [ ! -x "${LATENT_PYTHON}" ]; then
    echo "ERROR: Python executable not found: ${LATENT_PYTHON}"
    exit 1
fi
if [ ! -f "${LATENT_CHECKPOINT}" ]; then
    echo "ERROR: Checkpoint not found: ${LATENT_CHECKPOINT}"
    exit 1
fi
if [ -n "${LATENT_CONFIG}" ] && [ ! -f "${LATENT_CONFIG}" ]; then
    echo "ERROR: Config not found: ${LATENT_CONFIG}"
    exit 1
fi

export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib_${SLURM_JOB_ID:-local}"
mkdir -p "${MPLCONFIGDIR}"

echo "Checkpoint : ${LATENT_CHECKPOINT}"
echo "Config     : ${LATENT_CONFIG:-checkpoint-saved config}"
echo "Split      : ${LATENT_SPLIT}"
echo "Samples    : ${LATENT_NUM_SAMPLES}"
echo "Pool       : ${LATENT_POOL}"
echo "Recall K   : ${LATENT_RECALL_K}"
echo "Output     : ${LATENT_OUTPUT_DIR}"
nvidia-smi -L || true

read -r -a LATENT_RECALL_VALUES <<< "${LATENT_RECALL_K}"
LATENT_ARGS=(
    analyze_modality_alignment_4m.py
    --checkpoint "${LATENT_CHECKPOINT}"
    --split "${LATENT_SPLIT}"
    --num_samples "${LATENT_NUM_SAMPLES}"
    --output_dir "${LATENT_OUTPUT_DIR}"
    --batch_size "${LATENT_BATCH_SIZE}"
    --num_workers "${LATENT_NUM_WORKERS}"
    --device cuda
    --pool "${LATENT_POOL}"
    --projection "${LATENT_PROJECTION}"
    --recall_k "${LATENT_RECALL_VALUES[@]}"
)
if [ -n "${LATENT_CONFIG}" ]; then
    LATENT_ARGS+=(--config "${LATENT_CONFIG}")
fi
if [ "${LATENT_ONE_PER_GROUP}" = "1" ]; then
    LATENT_ARGS+=(--one_per_group)
fi
if [ "${LATENT_AMP}" = "1" ]; then
    LATENT_ARGS+=(--amp)
fi

srun --export=ALL --kill-on-bad-exit=1 \
    "${LATENT_PYTHON}" "${LATENT_ARGS[@]}"
