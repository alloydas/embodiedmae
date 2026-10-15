#!/bin/bash
# ── Delta resources: one GPU slice of a gpuA40x4 / gpuA100x4 node ───────────
# Account, partition and job name go on the sbatch command line, as for the
# other slurm/delta launchers:
#   sbatch --account=bepk-delta-gpu --partition=gpuA40x4 --job-name=pcview_maize_d2 \
#          --export=ALL,DELTA_ENV_FILE=$PWD/slurm/delta/delta_env.sh \
#          slurm/delta/pc_view_maize.sbatch maize_e2_pc_d2 maize_e2_pcrgb_d2 maize_e2_pcrgbd_d2 maize_4m_d2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/delta_%x_%j.out
#SBATCH --error=logs/delta_%x_%j.err

# 1 camera vs 3 vs 10 fused point clouds, MAIZE (eval/pc_view_eval_maize.py), then
# the paired bootstrap over plants (eval/pc_view_bootstrap.py --species maize).
# No masks to build first: a maize pointcloud_cam.ply is what its own camera sees, so
# the conditions are unions of the per-view clouds moved into camera 00's frame.
# Every run in one job must read the same data version (all _d2, or all v1).
# One pass per split serves every run; the cost is data loading (10 PLY + 10 poses
# per plant, 15,000 plants). Features cache per (run, ckpt, split) and are written
# atomically, so a requeued job resumes at the split it lost.
#
#   <run> or <run>:<ckpt-relpath>; default checkpoints/checkpoint_epoch_600.pth.

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: pc_view_maize.sbatch <run>[:<ckpt>] [more runs...]" >&2
  exit 2
fi

ENV_FILE="${DELTA_ENV_FILE:-${SLURM_SUBMIT_DIR:-.}/slurm/delta/delta_env.sh}"
if [ ! -f "${ENV_FILE}" ]; then
  echo "REFUSING: no ${ENV_FILE}."; exit 2
fi
# shellcheck source-path=SCRIPTDIR source=delta_env.example.sh
source "${ENV_FILE}"
: "${REPO_DIR:?REPO_DIR unset in ${ENV_FILE}}" "${MAIZE_ROOT:?}" "${CONDA_SH:?}" "${CONDA_ENV:?}"
cd "${REPO_DIR}"
mkdir -p logs reports

missing=()
for spec in "$@"; do
  run="${spec%%:*}"; ckpt="${spec#*:}"; [ "${ckpt}" != "${spec}" ] || ckpt=checkpoints/checkpoint_epoch_600.pth
  for f in "outputs/${run}/config.json" "outputs/${run}/${ckpt}"; do
    [ -f "${f}" ] || missing+=("${f}")
  done
done
[ -f "${MAIZE_ROOT}/plant_scores.csv" ] || missing+=("${MAIZE_ROOT}/plant_scores.csv")
[ -f "${MAIZE_ROOT}/val/plant_0000_00/camera_pose.json" ] || missing+=("${MAIZE_ROOT}/val/plant_0000_00/camera_pose.json (poses)")
if [ "${#missing[@]}" -gt 0 ]; then
  printf 'REFUSING: missing %s\n' "${missing[@]}"; exit 2
fi

for split in train val test; do
  case "${split}" in train) want=105000 ;; *) want=22500 ;; esac
  # shellcheck disable=SC2010
  n=$(ls -f "${MAIZE_ROOT}/${split}" 2>/dev/null | grep -c '^plant_' || true)
  if [ "${n}" -ne "${want}" ]; then
    echo "REFUSING: ${MAIZE_ROOT}/${split} has ${n} plant_* folders, expected ${want}."; exit 3
  fi
done

set +u
# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
set -u
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1

python slurm/delta/check_split_index.py maize "${MAIZE_ROOT}"

echo "JobID : ${SLURM_JOB_ID:-none}"
echo "Node  : $(hostname)"
echo "Runs  : $*"
echo "Date  : $(date)"
echo "Git   : $(git rev-parse HEAD 2>/dev/null || echo 'not a git checkout')"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
python -c '
import sys, torch
if not torch.cuda.is_available():
    sys.exit("CUDA not available: wrong env, or no GPU granted")
torch.ones(8, device="cuda").mul_(2).sum().item()
print("  cuda ok, torch", torch.__version__)
'

TAG="${SLURM_JOB_NAME:-pcview_maize}_${SLURM_JOB_ID:-manual}"
CPUS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
NW=$((CPUS > 1 ? CPUS - 1 : 1))

python eval/pc_view_eval_maize.py --runs "$@" --data-root "${MAIZE_ROOT}" \
  --num-workers "${NW}" --batch-size 32 --out-prefix "reports/${TAG}"

# The bootstrap refits the ridge 11 targets x 3 conditions x runs; keep its workers bounded.
LOKY_MAX_CPU_COUNT="${CPUS}" python eval/pc_view_bootstrap.py --species maize --runs "$@" \
  --data-root "${MAIZE_ROOT}" --out "reports/${TAG}_bootstrap.csv"

echo "Done at $(date)  ->  reports/${TAG}_{coverage,recon,probe,bootstrap}.csv"
