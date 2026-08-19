#!/bin/bash

# Submit the 4M training job and a one-GPU modality-alignment job that runs
# after training leaves the queue.  `afterany` is intentional: long training
# jobs commonly end at the Slurm time limit, but their best checkpoint remains
# valid for analysis.

set -euo pipefail

cd /work/mech-ai-scratch/yongyun/embodiedmae4m

PIPELINE_PYTHON="/work/mech-ai-scratch/yongyun/envs/myenv/bin/python"
PIPELINE_CONFIG="config_4m.yaml"
PIPELINE_TRAIN_SCRIPT="train_4m.sbatch"
PIPELINE_ALIGNMENT_SCRIPT="analyze_latent_4m.sbatch"

if [ ! -x "${PIPELINE_PYTHON}" ]; then
    echo "ERROR: Python executable not found: ${PIPELINE_PYTHON}"
    exit 1
fi
for PIPELINE_FILE in \
    "${PIPELINE_CONFIG}" \
    "${PIPELINE_TRAIN_SCRIPT}" \
    "${PIPELINE_ALIGNMENT_SCRIPT}"
do
    if [ ! -f "${PIPELINE_FILE}" ]; then
        echo "ERROR: Required file not found: ${PIPELINE_FILE}"
        exit 1
    fi
done

PIPELINE_RUN_DIR="$("${PIPELINE_PYTHON}" -c '
from pathlib import Path
import yaml

with Path("config_4m.yaml").open() as handle:
    config = yaml.safe_load(handle)
print(config["checkpointing"]["output_dir"])
')"

PIPELINE_CHECKPOINT="${LATENT_CHECKPOINT:-${PIPELINE_RUN_DIR}/best_model.pth}"
PIPELINE_ALIGNMENT_OUTPUT="${LATENT_OUTPUT_DIR:-${PIPELINE_RUN_DIR}/modality_alignment_best_model}"

mkdir -p logs

PIPELINE_TRAIN_JOB_RAW="$(sbatch --parsable "${PIPELINE_TRAIN_SCRIPT}")"
PIPELINE_TRAIN_JOB_ID="${PIPELINE_TRAIN_JOB_RAW%%;*}"

PIPELINE_ALIGNMENT_JOB_RAW="$(
    sbatch \
        --parsable \
        --dependency="afterany:${PIPELINE_TRAIN_JOB_ID}" \
        --export="ALL,LATENT_CHECKPOINT=${PIPELINE_CHECKPOINT},LATENT_OUTPUT_DIR=${PIPELINE_ALIGNMENT_OUTPUT}" \
        "${PIPELINE_ALIGNMENT_SCRIPT}"
)"
PIPELINE_ALIGNMENT_JOB_ID="${PIPELINE_ALIGNMENT_JOB_RAW%%;*}"

echo "Training job          : ${PIPELINE_TRAIN_JOB_ID}"
echo "Alignment job         : ${PIPELINE_ALIGNMENT_JOB_ID}"
echo "Alignment dependency  : afterany:${PIPELINE_TRAIN_JOB_ID}"
echo "Alignment checkpoint  : ${PIPELINE_CHECKPOINT}"
echo "Alignment output      : ${PIPELINE_ALIGNMENT_OUTPUT}"
