#!/bin/bash
# The one command for running this repo's remaining jobs on NCSA Delta.
#
#   bash slurm/delta/submit.sh train       <slug> [extra train_sorghum_4m.py args]
#   bash slurm/delta/submit.sh probe       <ckpt-relpath> <run> [run...]
#   bash slurm/delta/submit.sh probe_maize <ckpt-relpath> <run> [run...]
#   DRY_RUN=1 bash slurm/delta/submit.sh ...      # print the sbatch line, submit nothing
#   TRAIN_GPUS=4 bash slurm/delta/submit.sh train ...   # override train_arm's 2 GPUs (8 x 4)
#
# Account, partition and job name go on the sbatch COMMAND LINE from
# delta_env.sh, so the .sbatch headers carry no allocation and the only file a
# new user edits is delta_env.sh.
#
# It cds to REPO_DIR before submitting because a batch script cannot locate
# itself: Slurm runs a spooled copy, so BASH_SOURCE points into the spool. The
# batch scripts find delta_env.sh through SLURM_SUBMIT_DIR (and the explicit
# DELTA_ENV_FILE exported below), which is REPO_DIR only because of that cd.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ENV_FILE="${DELTA_ENV_FILE:-${HERE}/delta_env.sh}"

usage() {
  cat >&2 <<'EOF'
usage:
  bash slurm/delta/submit.sh train       <slug> [extra train_sorghum_4m.py args]
  bash slurm/delta/submit.sh probe       <ckpt-relpath> <run> [run...]
  bash slurm/delta/submit.sh probe_maize <ckpt-relpath> <run> [run...]
  DRY_RUN=1 bash slurm/delta/submit.sh ...     # print, do not submit

examples:
  bash slurm/delta/submit.sh train e2_pcrgbdt_tg --epochs 1 --no_wandb \
       --output_dir outputs/_smoke_tg                                      # smoke test
  bash slurm/delta/submit.sh train e2_pcrgbdt_tg
  bash slurm/delta/submit.sh probe checkpoints/checkpoint_epoch_600.pth \
       e2_pc e2_pcrgb e2_pcrgbd e2_pcrgbdt e2_pcrgbdt_tg e4_small e4_large
  bash slurm/delta/submit.sh probe checkpoints/checkpoint_epoch_6168.pth e3_1k
  bash slurm/delta/submit.sh probe_maize checkpoints/checkpoint_epoch_600.pth \
       maize_4m maize_e4_small maize_e4_large

One --ckpt per job, so each run is probed at its matched cut (see matched_epoch
below); E3 arms end at different epochs and therefore go in separate jobs.
EOF
  exit "${1:-2}"
}

die() { echo "submit.sh: $*" >&2; exit 2; }

[ "$#" -ge 1 ] || usage 2
CMD="$1"; shift
case "${CMD}" in
  train|probe|probe_maize) ;;
  -h|--help|help) usage 0 ;;
  *) echo "submit.sh: unknown subcommand '${CMD}'" >&2; usage 2 ;;
esac

[ -f "${ENV_FILE}" ] || die "no ${ENV_FILE}.
  cp ${HERE}/delta_env.example.sh ${HERE}/delta_env.sh   and fill it in."
# shellcheck source-path=SCRIPTDIR source=delta_env.example.sh
source "${ENV_FILE}"
TRAIN_PARTITION="${TRAIN_PARTITION:-gpuA100x4}"
EVAL_PARTITION="${EVAL_PARTITION:-gpuA40x4}"

# Set, and not still the template's placeholder. Which data root is required
# depends on the subcommand, so a user who has not copied maize yet can still
# train and probe sorghum.
need() {
  local v bad=0
  for v in "$@"; do
    if [ -z "${!v:-}" ]; then
      echo "  ${v} is not set in ${ENV_FILE}" >&2; bad=1
    elif [[ "${!v}" == *CHANGEME* ]]; then
      echo "  ${v} is still the template placeholder: ${!v}" >&2; bad=1
    fi
  done
  return "${bad}"
}
REQUIRED=(DELTA_ACCOUNT REPO_DIR CONDA_SH CONDA_ENV TRAIN_PARTITION EVAL_PARTITION)
case "${CMD}" in
  train|probe) REQUIRED+=(SORGHUM_ROOT); DATA_ROOT_VAR=SORGHUM_ROOT ;;
  probe_maize) REQUIRED+=(MAIZE_ROOT);   DATA_ROOT_VAR=MAIZE_ROOT ;;
esac
need "${REQUIRED[@]}" || die "fix ${ENV_FILE} first."
EXTRA_SBATCH=()

[ -f "${REPO_DIR}/train_sorghum_4m.py" ] || die "REPO_DIR=${REPO_DIR} is not a checkout of this repo."
[ -f "${CONDA_SH}" ] || die "CONDA_SH=${CONDA_SH} does not exist."
[ -d "${!DATA_ROOT_VAR}" ] || die "${DATA_ROOT_VAR}=${!DATA_ROOT_VAR} does not exist."

# The job runs REPO_DIR's scripts, not this file's neighbours. If they differ,
# an edit made here would silently not be what runs.
if [ "$(cd "${HERE}/../.." && pwd -P)" != "$(cd "${REPO_DIR}" && pwd -P)" ]; then
  echo "WARNING: this submit.sh lives in $(cd "${HERE}/../.." && pwd -P)," >&2
  echo "         but REPO_DIR is ${REPO_DIR}; the job will run REPO_DIR's copy." >&2
fi

# The matched checkpoint cut per run (CLAUDE.md). best_model.pth is selected on
# total val loss, which lands at 42 % of schedule for e3_1k and 100 % for
# e3_10k, so probing it bills training length to whatever the arms vary. Every
# arm is compared at its last SAVED checkpoint instead -- which for E3 is not the
# final epoch, because save_freq was scaled per arm: e3_1k@6168 = 197,376 steps,
# e3_3k@2024 = 190,256 (96.4 %), e3_10k@624 = 195,312 (98.9 %), the rest @600 =
# 197,400. Quote the steps next to the R^2. e2_pcrgbdt@600 is also E3's
# full-data point and E4's base point. maize_4m_1000ep has a cosine restart in
# it and is comparable to nothing, hence "none".
matched_epoch() {
  case "$1" in
    e2_pc|e2_pcrgb|e2_pcrgbd|e2_pcrgbdt|e2_pcrgbdt_tg|e4_small|e4_large) echo 600 ;;
    e3_1k|maize_e3_1k)   echo 6168 ;;
    e3_3k|maize_e3_3k)   echo 2024 ;;
    e3_10k|maize_e3_10k) echo 624 ;;
    maize_4m|maize_e4_small|maize_e4_large) echo 600 ;;
    maize_4m_1000ep) echo none ;;
    *) echo "" ;;
  esac
}

case "${CMD}" in
  train)
    [ "$#" -ge 1 ] || usage 2
    SLUG="$1"; shift
    case "${SLUG}" in
      maize_*) die "'${SLUG}' is a maize arm; train_arm.sbatch runs train_sorghum_4m.py,
  which would build a 9-param/24-leaf model and find no *_nc_cam.ply in the maize tree." ;;
    esac
    [ -f "${REPO_DIR}/configs/config_${SLUG}.yaml" ] || die "no configs/config_${SLUG}.yaml in ${REPO_DIR}."
    PARTITION="${TRAIN_PARTITION}"
    JOB_NAME="tr_${SLUG}"
    SCRIPT=slurm/delta/train_arm.sbatch
    ARGS=("${SLUG}" "$@")
    # train_arm.sbatch asks for 2 GPUs (the reference arms' 16 x 2). A command-line
    # --gpus-per-node overrides the #SBATCH line; the launcher derives the rest.
    [ -z "${TRAIN_GPUS:-}" ] || EXTRA_SBATCH+=(--gpus-per-node "${TRAIN_GPUS}")

    # A second trainer on the same slug auto-resumes into the same output dir:
    # both write checkpoint_epoch_N.pth, training_history.json and best_model.pth
    # non-atomically over each other and log duplicate epochs into one W&B run.
    # The usual way in: re-running this line after what looked like a TIMEOUT but
    # was a job still running, or requeued to PENDING after a node failure.
    if command -v squeue >/dev/null 2>&1; then
      live="$(squeue -h -u "${USER}" -n "${JOB_NAME}" -o '%i %T' 2>/dev/null || true)"
      if [ -n "${live}" ] && [ "${ALLOW_DUPLICATE:-0}" != 1 ]; then
        printf '%s\n' "${live}" | sed 's/^/  already queued or running: /' >&2
        die "refusing a second ${JOB_NAME} (ALLOW_DUPLICATE=1 overrides; an --output_dir smoke test is the one legitimate case)."
      fi
    else
      echo "WARNING: no squeue on PATH; cannot check for a live ${JOB_NAME}." >&2
    fi
    ;;

  probe|probe_maize)
    [ "$#" -ge 2 ] || usage 2
    CKPT="$1"; shift
    b="$(basename "${CKPT}" .pth)"
    case "${b}" in checkpoint_epoch_*) EP="${b##*_}" ;; *) EP="${b}" ;; esac

    wrong_species=0; unmatched=0
    for run in "$@"; do
      if [ "${CMD}" = probe ] && [[ "${run}" == maize_* ]]; then
        echo "  ${run}: a maize run -- use probe_maize" >&2; wrong_species=1; continue
      fi
      if [ "${CMD}" = probe_maize ] && [[ "${run}" != maize_* ]]; then
        echo "  ${run}: not a maize run -- use probe" >&2; wrong_species=1; continue
      fi
      want="$(matched_epoch "${run}")"
      if [ -z "${want}" ]; then
        echo "  note: no matched cut on record for ${run}; probing ${CKPT} as asked" >&2
      elif [ "${CKPT}" != "checkpoints/checkpoint_epoch_${want}.pth" ] \
           && [ "${ALLOW_UNMATCHED_CKPT:-0}" != 1 ]; then
        if [ "${want}" = none ]; then
          echo "  ${run}: not comparable to any other run at any checkpoint" >&2
        else
          echo "  ${run}: matched cut is checkpoints/checkpoint_epoch_${want}.pth, not ${CKPT}" >&2
        fi
        unmatched=1
      fi
    done
    [ "${wrong_species}" = 0 ] || die "refusing: the probe scripts are species-specific."
    [ "${unmatched}" = 0 ] || die "refusing (ALLOW_UNMATCHED_CKPT=1 overrides the matched-cut check)."

    # The batch script repeats this preflight; doing it here as well means a
    # typo costs nothing instead of a queue wait.
    missing=()
    for run in "$@"; do
      for f in "outputs/${run}/config.json" "outputs/${run}/${CKPT}"; do
        [ -f "${REPO_DIR}/${f}" ] || missing+=("${f}")
      done
    done
    if [ "${CMD}" = probe ]; then
      for f in features.csv assignment.csv; do
        [ -f "${SORGHUM_ROOT}/${f}" ] || missing+=("\${SORGHUM_ROOT}/${f}")
      done
    else
      [ -f "${MAIZE_ROOT}/plant_scores.csv" ] || missing+=("\${MAIZE_ROOT}/plant_scores.csv")
    fi
    if [ "${#missing[@]}" -gt 0 ]; then
      printf '  missing: %s\n' "${missing[@]}" >&2
      die "nothing submitted."
    fi

    PARTITION="${EVAL_PARTITION}"
    if [ "${CMD}" = probe ]; then
      JOB_NAME="pr_${EP}"; SCRIPT=slurm/delta/probe.sbatch
    else
      JOB_NAME="pm_${EP}"; SCRIPT=slurm/delta/probe_maize.sbatch
    fi
    ARGS=("${CKPT}" "$@")
    ;;
esac

ENV_ABS="$(cd "$(dirname "${ENV_FILE}")" && pwd -P)/$(basename "${ENV_FILE}")"
SBATCH=(sbatch
        --account "${DELTA_ACCOUNT}"
        --partition "${PARTITION}"
        --job-name "${JOB_NAME}"
        --export "ALL,DELTA_ENV_FILE=${ENV_ABS}"
        ${EXTRA_SBATCH[@]+"${EXTRA_SBATCH[@]}"}
        "${SCRIPT}" "${ARGS[@]}")

cd "${REPO_DIR}"
printf '+ cd %q\n+' "${REPO_DIR}"
printf ' %q' "${SBATCH[@]}"
printf '\n'

if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "DRY_RUN=1: not submitted."
  exit 0
fi

# Slurm opens --output=logs/... before the script runs; without the directory
# the job fails at once and leaves no log to say why.
mkdir -p logs
"${SBATCH[@]}"
