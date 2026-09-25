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

# Sorghum linear probe (decision 6.4) then E9 latent analysis, at ONE checkpoint:
#   bash slurm/delta/submit.sh probe checkpoints/checkpoint_epoch_600.pth \
#        e2_pc e2_pcrgb e2_pcrgbd e2_pcrgbdt e2_pcrgbdt_tg e4_small e4_large
#   bash slurm/delta/submit.sh probe checkpoints/checkpoint_epoch_6168.pth e3_1k
# submit.sh refuses a run at anything but its matched cut; e3 arms end at
# different epochs, so each is its own job.
#
# One GPU is enough: one forward pass per plant, no backward. No --requeue:
# features cache per (run, ckpt, split), so a rerun re-reads what it wrote.
#
# E9 runs on --device cpu against the cache the probe just wrote. The two
# scripts build the SAME cache name, <run>__<Path(ckpt).stem>__<split>__cls
# __seed0__rep1.npz, and the probe extracts train, val AND test, so every E9
# split is a cache hit. They are given one shared --ckpt/--cache-dir/--feature/
# --seed/--repeats so the names cannot drift apart through a default.

set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: probe.sbatch <ckpt-relpath> <run-slug> [more slugs...]" >&2
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
: "${REPO_DIR:?REPO_DIR unset in ${ENV_FILE}}" "${SORGHUM_ROOT:?}" "${CONDA_SH:?}" "${CONDA_ENV:?}"
cd "${REPO_DIR}"
mkdir -p logs reports

# Every missing input at once, not one per queue wait.
missing=()
for run in "$@"; do
  for f in "outputs/${run}/config.json" "outputs/${run}/${CKPT}"; do
    [ -f "${f}" ] || missing+=("${f}")
  done
done
for f in features.csv assignment.csv; do
  [ -f "${SORGHUM_ROOT}/${f}" ] || missing+=("${SORGHUM_ROOT}/${f}")
done
if [ "${#missing[@]}" -gt 0 ]; then
  printf 'REFUSING: missing %s\n' "${missing[@]}"
  exit 2
fi

# The probe scores every arm on the same held-out plants. A partially copied
# split would score on a silent subset and move R2 with nothing in the log to
# say so -- the same exact-count guard as train_arm.sbatch.
for split in train val test; do
  case "${split}" in train) want=105000 ;; *) want=22500 ;; esac
  # shellcheck disable=SC2010  # unsorted ls -f beats a 105k-entry glob on Lustre; names are fixed-form
  n=$(ls -f "${SORGHUM_ROOT}/${split}" 2>/dev/null | grep -c '^Sorghum_' || true)
  if [ "${n}" -ne "${want}" ]; then
    echo "REFUSING: ${SORGHUM_ROOT}/${split} has ${n} Sorghum_* folders, expected ${want}."
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
python slurm/delta/check_split_index.py sorghum "${SORGHUM_ROOT}"

echo "JobID : ${SLURM_JOB_ID:-none}"
echo "Node  : $(hostname)"
echo "Ckpt  : ${CKPT}"
echo "Runs  : $*"
echo "Date  : $(date)"
echo "Git   : $(git rev-parse HEAD 2>/dev/null || echo 'not a git checkout')"
sha256sum eval/linear_probe.py eval/latent_analysis.py | sed 's/^/  sha256: /'
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# linear_probe.py falls back to CPU with a one-line warning when CUDA is
# missing, and a CPU extraction of 15,000 plants x N runs runs straight into
# the 4 h wall. Fail here instead, with a real kernel launch.
python -c '
import sys, torch
if not torch.cuda.is_available():
    sys.exit("CUDA not available: wrong env, or no GPU granted")
p = torch.cuda.get_device_properties(0)
torch.ones(8, device="cuda").mul_(2).sum().item()
print(f"  cuda:0 {p.name} sm_{p.major}{p.minor}, torch {torch.__version__}  kernel ok")
'

TAG="${SLURM_JOB_NAME:-probe}_${SLURM_JOB_ID:-manual}"
CPUS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
NW=$((CPUS > 1 ? CPUS - 1 : 1))
COMMON=(--ckpt "${CKPT}" --data-root "${SORGHUM_ROOT}"
        --cache-dir "${PROBE_CACHE_DIR:-outputs/_probe_cache}"
        --feature cls --seed 0 --repeats 1)

python eval/linear_probe.py --runs "$@" "${COMMON[@]}" \
  --num-workers "${NW}" --batch-size 32 \
  --out "reports/probe_${TAG}.csv"

# E9 per split. Its summary CSV is named by split alone and the per-run CSVs
# by run and split, not checkpoint, so two probe jobs sharing reports/ would
# overwrite each other; a directory per job keeps them apart.
ck="$(basename "${CKPT}")"; ck="${ck%.*}"          # == Path(ckpt).stem
E9_FAILED=()
for split in ${E9_SPLITS:-val train test}; do
  # A missing cache would make E9 re-extract on CPU for hours. After a
  # successful probe none is missing; if one is, the keys have drifted.
  cold=()
  for run in "$@"; do
    c="${PROBE_CACHE_DIR:-outputs/_probe_cache}/${run}__${ck}__${split}__cls__seed0__rep1.npz"
    [ -f "${c}" ] || cold+=("${c}")
  done
  if [ "${#cold[@]}" -gt 0 ]; then
    printf 'E9 %s skipped, cache missing:\n' "${split}"
    printf '  %s\n' "${cold[@]}"
    E9_FAILED+=("${split}"); continue
  fi
  python eval/latent_analysis.py --runs "$@" --split "${split}" "${COMMON[@]}" \
    --device cpu --out-dir "reports/e9_${TAG}" --fig-dir "reports/e9_${TAG}/figures" \
    || E9_FAILED+=("${split}")
done

echo "Done at $(date)  ->  reports/probe_${TAG}.csv, reports/e9_${TAG}/"
if [ "${#E9_FAILED[@]}" -gt 0 ]; then
  echo "E9 FAILED for split(s): ${E9_FAILED[*]} (the probe CSV above is complete)"
  exit 1
fi
