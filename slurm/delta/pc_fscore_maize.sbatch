#!/bin/bash
# Point-cloud precision / recall / F1 for maize runs (eval/pc_fscore_maize.py),
# val + test, then the Markdown report. One A40 is plenty: one forward pass per
# plant and an 8192 x 8192 distance matrix per sample.
#   sbatch --account=bepk-delta-gpu --partition=gpuA40x4 --job-name=fs_maize \
#          slurm/delta/pc_fscore_maize.sbatch maize_e2_pc_d2 [more runs...]
# A run may carry an explicit checkpoint as run:checkpoints/checkpoint_epoch_N.pth;
# the default is its highest-numbered checkpoint.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/delta_%x_%j.out
#SBATCH --error=logs/delta_%x_%j.err

set -euo pipefail
[ "$#" -ge 1 ] || { echo "usage: pc_fscore_maize.sbatch <run>[:ckpt] [more...]" >&2; exit 2; }

ENV_FILE="${DELTA_ENV_FILE:-${SLURM_SUBMIT_DIR:-.}/slurm/delta/delta_env.sh}"
# shellcheck source=delta_env.example.sh
source "${ENV_FILE}"
: "${REPO_DIR:?}" "${MAIZE_ROOT:?}" "${CONDA_SH:?}" "${CONDA_ENV:?}"
cd "${REPO_DIR}"
mkdir -p logs reports

# Refuse early on a run that has not finished: scoring a mid-schedule checkpoint
# would put a partially trained point in the comparison table.
for spec in "$@"; do
  run="${spec%%:*}"
  cfg="outputs/${run}/config.json"
  [ -f "${cfg}" ] || { echo "REFUSING: no ${cfg}"; exit 2; }
  want=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["epochs"])' "${cfg}")
  have=$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1])).get("train_loss", [])))' \
         "outputs/${run}/training_history.json" 2>/dev/null || echo 0)
  if [ "${have}" -lt "${want}" ] && [ -z "${ALLOW_UNFINISHED:-}" ]; then
    echo "REFUSING: ${run} has ${have}/${want} train epochs. Set ALLOW_UNFINISHED=1 to score anyway."
    exit 3
  fi
done

set +u; source "${CONDA_SH}"; conda activate "${CONDA_ENV}"; set -u
export PYTHONUNBUFFERED=1
echo "JobID ${SLURM_JOB_ID:-none}  node $(hostname)  $(date)  runs: $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-node=1 \
  python eval/pc_fscore_maize.py --runs "$@" --splits val test \
    --data-root "${MAIZE_ROOT}" --batch-size 16 --num-workers $((SLURM_CPUS_PER_TASK - 1))
python eval/pc_fscore_report.py
echo "Done at $(date)"
