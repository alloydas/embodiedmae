#!/bin/bash
#SBATCH --job-name=latent4m
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=latent4m_%j.out
#SBATCH --error=latent4m_%j.err

set -euo pipefail

cd /work/mech-ai-scratch/yongyun/embodiedmae4m

LATENT_PYTHON="/work/mech-ai-scratch/yongyun/envs/myenv/bin/python"
LATENT_CHECKPOINT="${LATENT_CHECKPOINT:-outputs/4m_run_v4_qal_sinkhorn_no_duplicate_sk_loss_1.0/checkpoints/checkpoint_epoch_800.pth}"
LATENT_OUTPUT_DIR="${LATENT_OUTPUT_DIR:-outputs/4m_run_v4_qal_sinkhorn_no_duplicate_sk_loss_1.0/latent_alignment_ep800}"
LATENT_SPLIT="${LATENT_SPLIT:-val}"
LATENT_NUM_SAMPLES="${LATENT_NUM_SAMPLES:-1000}"

if [ ! -x "${LATENT_PYTHON}" ]; then
    echo "ERROR: Python executable not found: ${LATENT_PYTHON}"
    exit 1
fi
if [ ! -f "${LATENT_CHECKPOINT}" ]; then
    echo "ERROR: Checkpoint not found: ${LATENT_CHECKPOINT}"
    exit 1
fi

export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib_${SLURM_JOB_ID:-local}"
mkdir -p "${MPLCONFIGDIR}"

echo "Checkpoint : ${LATENT_CHECKPOINT}"
echo "Split      : ${LATENT_SPLIT}"
echo "Samples    : ${LATENT_NUM_SAMPLES}"
echo "Output     : ${LATENT_OUTPUT_DIR}"
nvidia-smi -L

srun --export=ALL --kill-on-bad-exit=1 \
    "${LATENT_PYTHON}" analyze_modality_alignment_4m.py \
        --checkpoint "${LATENT_CHECKPOINT}" \
        --split "${LATENT_SPLIT}" \
        --num_samples "${LATENT_NUM_SAMPLES}" \
        --output_dir "${LATENT_OUTPUT_DIR}" \
        --batch_size 8 \
        --num_workers 8 \
        --device cuda \
        --projection pca \
        --amp
