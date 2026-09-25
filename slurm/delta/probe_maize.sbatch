#!/bin/bash
# ── Delta resources ─────────────────────────────────────────────────────────
# From NCSA's Delta docs as best known -- gpuA40x4 = 4x A40-48GB, 64 cores,
# ~256 GB; this takes a quarter-node slice. CONFIRM on Delta before trusting:
#   sinfo -p gpuA40x4 -o "%P %c %m %G %l"
# Account, partition and job name come from submit.sh, not from here.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
# No --requeue here, but Slurm still requeues on node failure (JobRequeue=1 by
# default) and would truncate the log: append instead.
#SBATCH --open-mode=append
#SBATCH --output=logs/delta_%x_%j.out
#SBATCH --error=logs/delta_%x_%j.err

# MAIZE linear probe then E9 latent analysis, at ONE checkpoint -- the shape of
# probe.sbatch with the maize scripts, root and cache:
#   bash slurm/delta/submit.sh probe_maize checkpoints/checkpoint_epoch_600.pth \
#        maize_4m maize_e4_small maize_e4_large
#   bash slurm/delta/submit.sh probe_maize checkpoints/checkpoint_epoch_6168.pth maize_e3_1k
# maize_4m is probed at 600 -- outputs/maize_4m is frozen there and is the only
# maize artefact on the 197,400-step budget. maize_4m_1000ep has a cosine
# restart in it and is comparable to nothing; submit.sh refuses it.
#
# Cache names match between the two maize scripts:
# maize_<run>__<Path(ckpt).stem>__<split>__cls__seed0__rep1.npz under
# outputs/_probe_cache_maize, and --split-set train,val,test extracts all three,
# so E9 on --device cpu is a cache hit for every split.

set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: probe_maize.sbatch <ckpt-relpath> <run-slug> [more slugs...]" >&2
  exit 2
fi
CKPT="$1"; shift

ENV_FILE="${DELTA_ENV_FILE:-${SLURM_SUBMIT_DIR:-.}/slurm/delta/delta_env.sh}"
if [ ! -f "${ENV_FILE}" ]; then
  echo "REFUSING: no ${ENV_FILE}."
  echo "Submit with slurm/delta/submit.sh, after copying delta_env.example.sh to delta_env.sh."
  exit 2
fi
# shellcheck source-path=SCRIPTDIR source=delta_env.example.sh
source "${ENV_FILE}"
: "${REPO_DIR:?REPO_DIR unset in ${ENV_FILE}}" "${MAIZE_ROOT:?}" "${CONDA_SH:?}" "${CONDA_ENV:?}"
cd "${REPO_DIR}"
mkdir -p logs reports

# plant_scores.csv is the maize target table (there is no features.csv or
# assignment.csv on this side); without it the probe dies after model load.
missing=()
for run in "$@"; do
  for f in "outputs/${run}/config.json" "outputs/${run}/${CKPT}"; do
    [ -f "${f}" ] || missing+=("${f}")
  done
done
[ -f "${MAIZE_ROOT}/plant_scores.csv" ] || missing+=("${MAIZE_ROOT}/plant_scores.csv")
if [ "${#missing[@]}" -gt 0 ]; then
  printf 'REFUSING: missing %s\n' "${missing[@]}"
  exit 2
fi

# Split by plant, 10 views each: 105,000 / 22,500 / 22,500 plant_* folders. A
# partial copy would score on a silent subset. Each split root also holds a
# _params.json, which the ^plant_ match leaves out.
for split in train val test; do
  case "${split}" in train) want=105000 ;; *) want=22500 ;; esac
  # shellcheck disable=SC2010  # unsorted ls -f beats a 105k-entry glob on Lustre; names are fixed-form
  n=$(ls -f "${MAIZE_ROOT}/${split}" 2>/dev/null | grep -c '^plant_' || true)
  if [ "${n}" -ne "${want}" ]; then
    echo "REFUSING: ${MAIZE_ROOT}/${split} has ${n} plant_* folders, expected ${want}."
    exit 3
  fi
  echo "split ${split}: ${n}/${want} ok"
done

set +u
# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
set -u
export PYTHONUNBUFFERED=1

# The folder count above cannot see a folder that exists but is still empty; the
# loader skips it and caches the short index keyed on the split dir's mtime,
# which later files never change. Check the index the probe will read.
python slurm/delta/check_split_index.py maize "${MAIZE_ROOT}"

echo "JobID : ${SLURM_JOB_ID:-none}"
echo "Node  : $(hostname)"
echo "Ckpt  : ${CKPT}"
echo "Runs  : $*"
echo "Date  : $(date)"
echo "Git   : $(git rev-parse HEAD 2>/dev/null || echo 'not a git checkout')"
sha256sum eval/linear_probe_maize.py eval/latent_analysis_maize.py | sed 's/^/  sha256: /'
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# linear_probe_maize.py falls back to CPU with a warning when CUDA is missing,
# which runs into the 4 h wall. Fail here instead, with a real kernel launch.
python -c '
import sys, torch
if not torch.cuda.is_available():
    sys.exit("CUDA not available: wrong env, or no GPU granted")
p = torch.cuda.get_device_properties(0)
torch.ones(8, device="cuda").mul_(2).sum().item()
print(f"  cuda:0 {p.name} sm_{p.major}{p.minor}, torch {torch.__version__}  kernel ok")
'

TAG="${SLURM_JOB_NAME:-probe_maize}_${SLURM_JOB_ID:-manual}"
CPUS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
NW=$((CPUS > 1 ? CPUS - 1 : 1))
CACHE="${PROBE_CACHE_DIR_MAIZE:-outputs/_probe_cache_maize}"
COMMON=(--ckpt "${CKPT}" --data-root "${MAIZE_ROOT}" --cache-dir "${CACHE}"
        --feature cls --seed 0 --repeats 1)

# train FIRST: the maize probe fits on the first split of --split-set.
python eval/linear_probe_maize.py --runs "$@" "${COMMON[@]}" \
  --split-set train,val,test --num-workers "${NW}" --batch-size 32 \
  --out "reports/probe_${TAG}.csv"

# One directory per job: E9's CSVs are named by run and split, not checkpoint.
ck="$(basename "${CKPT}")"; ck="${ck%.*}"          # == Path(ckpt).stem
E9_FAILED=()
for split in ${E9_SPLITS:-val train test}; do
  cold=()
  for run in "$@"; do
    c="${CACHE}/maize_${run}__${ck}__${split}__cls__seed0__rep1.npz"
    [ -f "${c}" ] || cold+=("${c}")
  done
  if [ "${#cold[@]}" -gt 0 ]; then
    printf 'E9 %s skipped, cache missing:\n' "${split}"
    printf '  %s\n' "${cold[@]}"
    E9_FAILED+=("${split}"); continue
  fi
  python eval/latent_analysis_maize.py --runs "$@" --split "${split}" "${COMMON[@]}" \
    --device cpu --out-dir "reports/e9_${TAG}" --fig-dir "reports/e9_${TAG}/figures" \
    || E9_FAILED+=("${split}")
done

echo "Done at $(date)  ->  reports/probe_${TAG}.csv, reports/e9_${TAG}/"
if [ "${#E9_FAILED[@]}" -gt 0 ]; then
  echo "E9 FAILED for split(s): ${E9_FAILED[*]} (the probe CSV above is complete)"
  exit 1
fi
