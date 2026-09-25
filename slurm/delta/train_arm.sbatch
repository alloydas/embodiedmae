#!/bin/bash
# ── Delta resources ─────────────────────────────────────────────────────────
# From NCSA's Delta docs as best known -- gpuA100x4 = 4x A100-SXM4-40GB,
# 64 cores, ~256 GB, 48 h max walltime. CONFIRM on Delta before trusting them:
#   sinfo -p gpuA100x4 -o "%P %c %m %G %l"      # cpus, MB, gres, time limit
# These are the only hardware numbers in this file; per-GPU batch and worker
# count below are derived from what Slurm actually granted. TRAIN_GPUS=4 on the
# submit.sh line overrides the GPU count without editing this file.
# Account, partition and job name are NOT here -- submit.sh passes them on the
# sbatch command line from delta_env.sh.
#
# TWO GPUs, not four: every reference arm (all nine E2/E3/E4 config.json) ran
# 16/GPU x 2 ranks. Global batch and batch composition are identical at 8 x 4,
# but PointCloudEmbed's BatchNorm1d is per-rank (no SyncBN) and each rank draws
# its own Dirichlet split, so 8 x 4 changes both for the one arm meant to split a
# ~0.015 R^2 gap. Batch 16 peaks at 22.4 GiB (logs/bsweep_12044382.out, base 4M,
# all four streams), inside an A100's 40. The run is dataloader-bound, so it keeps
# all 64 cores: 2 ranks x 31 workers load as fast as 4 x 15.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=48:00:00
#SBATCH --requeue
# Append on requeue: Slurm truncates the log by default, and the per-epoch val
# lines in it are what recovered e3_1k's history when its JSON was cut mid-write.
#SBATCH --open-mode=append
#SBATCH --output=logs/delta_%x_%j.out
#SBATCH --error=logs/delta_%x_%j.err

# Any sorghum 4M arm, by slug, on one Delta node:
#   bash slurm/delta/submit.sh train e2_pcrgbdt_tg
#   bash slurm/delta/submit.sh train e2_pcrgbdt_tg --epochs 1 --no_wandb --output_dir outputs/_smoke_tg
# Config is configs/config_<slug>.yaml; the output dir is that YAML's
# checkpointing.output_dir unless --output_dir is among the extra args, which go
# straight to train_sorghum_4m.py. --epochs requires --output_dir, so a smoke
# test can never leave checkpoints where the real run would resume from them.
#
# GLOBAL BATCH IS 32 whatever the GPU count: --batch_size is 32/NPROC. Check
# after launch:  jq '.batch_size, .world_size' outputs/<slug>/config.json
#
# --requeue covers preemption and node failure, NOT a walltime TIMEOUT. Slurm
# resubmits nothing when --time runs out, and from outside a TIMEOUT looks like
# a job still running (`sacct -j <id> -o State` tells them apart). Re-run the
# same submit.sh line by hand; the auto-resume below then caps the loss at
# save_freq epochs. An explicit --resume is honoured on the first start only:
# Slurm replays the same argv on a requeue, so from then on the run's own newest
# checkpoint wins.

set -euo pipefail

SLUG="${1:?usage: train_arm.sbatch <slug> [extra train_sorghum_4m.py args...]}"
shift

# Spooled by Slurm, so BASH_SOURCE is useless here: find the env through the
# submit dir (submit.sh cds to REPO_DIR first) or the path submit.sh exported.
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
mkdir -p logs

# Same guard as slurm/scale_arm_blackwell.sbatch: configs/config_maize_*.yaml
# exist, so the -f check below would pass and train_sorghum_4m.py would build a
# 9-param/24-leaf model against a tree with no *_nc_cam.ply -- dying only after
# the queue wait and a 105,000-folder scan.
case "${SLUG}" in
  maize_*) echo "REFUSING: '${SLUG}' is a maize arm; this launcher runs train_sorghum_4m.py."
           exit 2 ;;
esac

CONFIG="configs/config_${SLUG}.yaml"
[ -f "${CONFIG}" ] || { echo "no such config: ${CONFIG}"; exit 2; }

# Flags this launcher owns or reads. argparse is last-wins AND resolves
# abbreviations (allow_abbrev), so `--batch 16` is --batch_size to the trainer
# and would quietly break the global batch of 32 while this log still printed 8.
# Resolve every extra flag against the trainer's own option list, the way
# argparse will; an ambiguous or unknown one is left for argparse to reject.
mapfile -t TRAINER_OPTS < <(grep -oE "add_argument\('--[a-z_]+'" train_sorghum_4m.py \
                            | sed -E "s/.*'(--[a-z_]+)'/\1/")
canon() {
  local o hits=()
  for o in "${TRAINER_OPTS[@]}"; do
    [ "${o}" = "$1" ] && { echo "${o}"; return; }
    [[ "${o}" == "$1"* ]] && hits+=("${o}")
  done
  [ "${#hits[@]}" -eq 1 ] && echo "${hits[0]}"
  return 0
}
USER_OUT=""; USER_EPOCHS=""; USER_RESUME=""; pending=""
for a in "$@"; do
  if [ -n "${pending}" ]; then
    case "${pending}" in
      --output_dir) USER_OUT="${a}" ;;
      --epochs)     USER_EPOCHS="${a}" ;;
      --resume)     USER_RESUME="${a}" ;;
    esac
    pending=""; continue
  fi
  case "${a}" in --*) ;; *) continue ;; esac
  c="$(canon "${a%%=*}")"
  case "${c}" in
    --batch_size)  echo "REFUSING: ${a}: the per-GPU batch is 32 / GPUs granted, not an argument."; exit 2 ;;
    --world_size)  echo "REFUSING: ${a}: world size comes from the GPUs granted."; exit 2 ;;
    --data_root)   echo "REFUSING: ${a}: set SORGHUM_ROOT in delta_env.sh."; exit 2 ;;
    --num_workers) echo "REFUSING: ${a}: set NUM_WORKERS (the memory check reads it)."; exit 2 ;;
    --config)      echo "REFUSING: ${a}: the slug picks the config, and this launcher reads it for"
                   echo "          the output dir, epochs and text_mask_ratio -- a second one would"
                   echo "          train a YAML nobody checked."; exit 2 ;;
    --output_dir|--epochs|--resume)
      if [[ "${a}" == *=* ]]; then
        case "${c}" in
          --output_dir) USER_OUT="${a#*=}" ;;
          --epochs)     USER_EPOCHS="${a#*=}" ;;
          --resume)     USER_RESUME="${a#*=}" ;;
        esac
      else
        pending="${c}"
      fi ;;
  esac
done

# A shortened run in the arm's real output dir leaves a config.json describing
# it, and -- at >= save_freq epochs -- a checkpoint the real run would resume from.
if [ -n "${USER_EPOCHS}" ] && [ -z "${USER_OUT}" ]; then
  echo "REFUSING: --epochs ${USER_EPOCHS} without --output_dir would write into the arm's"
  echo "real output dir. Add e.g. --output_dir outputs/_smoke_${SLUG}."
  exit 2
fi

# conda's activate hooks reference unset variables, which set -u turns fatal.
set +u
# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
set -u

export WANDB_MODE="${WANDB_MODE:-online}"
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1

# Output dir, epochs and text_mask_ratio, read the way the trainer reads them.
mapfile -t _cfg < <(python -c '
import sys, yaml
c = yaml.safe_load(open(sys.argv[1])) or {}
print((c.get("checkpointing") or {}).get("output_dir") or "")
print((c.get("training") or {}).get("epochs") or "")
print((c.get("model") or {}).get("text_mask_ratio"))
' "${CONFIG}")
OUT_DIR="${USER_OUT:-${_cfg[0]:-}}"
EPOCHS="${USER_EPOCHS:-${_cfg[1]:-}}"
CFG_TMR="${_cfg[2]:-None}"
[ -n "${OUT_DIR}" ] || { echo "REFUSING: ${CONFIG} has no checkpointing.output_dir, so auto-resume cannot find the run."; exit 2; }
[[ "${EPOCHS}" =~ ^[0-9]+$ ]] || { echo "REFUSING: cannot read training.epochs from ${CONFIG} (got '${EPOCHS}')."; exit 2; }

# A trainer that predates text_mask_ratio ignores the YAML key without a word,
# and the control arm would silently re-run e2_pcrgbdt. Refuse on a stale checkout.
if { [ "${CFG_TMR}" != None ] || [[ " $* " == *" --text_mask_ratio"* ]]; } \
   && ! grep -q text_mask_ratio train_sorghum_4m.py; then
  echo "REFUSING: ${CONFIG} sets text_mask_ratio (${CFG_TMR}) but this checkout's"
  echo "train_sorghum_4m.py does not read it -- sync the trainer from Nova first."
  exit 2
fi

echo "JobID   : ${SLURM_JOB_ID:-none}  (restart ${SLURM_RESTART_COUNT:-0})"
echo "Slug    : ${SLUG}"
echo "Config  : ${CONFIG}  (text_mask_ratio ${CFG_TMR}, epochs ${EPOCHS})"
echo "Output  : ${OUT_DIR}"
echo "Data    : ${SORGHUM_ROOT}"
echo "Node    : $(hostname)"
echo "Date    : $(date)"
echo "Env     : ${CONDA_ENV}  (WANDB_MODE=${WANDB_MODE})"
echo "Git     : $(git rev-parse HEAD 2>/dev/null || echo 'not a git checkout')"
# The control arm's trainer change is uncommitted, and a `transfer_from_nova.sh
# code` copy has no .git at all, so hash what actually runs. Tracked files only
# in the dirty list: untracked would walk outputs/.
git --no-optional-locks status --porcelain --untracked-files=no 2>/dev/null | sed 's/^/  dirty: /' || true
sha256sum train_sorghum_4m.py embodied_mae_4m.py embodied_mae.py \
          sorghum_dataset.py sorghum_dataset_4m.py "${CONFIG}" | sed 's/^/  sha256: /'
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# A half-transferred split is invisible to the trainer. Under view_sampling an
# epoch is one view per PLANT, so a train split short of folders trains on a
# silent subset at fewer steps per epoch, and the 197,400-step budget every arm
# is matched on no longer holds. For E3 it is worse: the nested max_plants
# subset is drawn from the plant list, so a partial split can change WHICH
# plants e3_1k sees. Exact counts, both directions. `ls -f` skips the sort,
# which matters over 105,000 entries on Lustre.
for split in train val test; do
  case "${split}" in train) want=105000 ;; *) want=22500 ;; esac
  # shellcheck disable=SC2010  # unsorted ls -f beats a 105k-entry glob on Lustre; names are fixed-form
  n=$(ls -f "${SORGHUM_ROOT}/${split}" 2>/dev/null | grep -c '^Sorghum_' || true)
  if [ "${n}" -ne "${want}" ]; then
    echo "REFUSING TO START: ${SORGHUM_ROOT}/${split} has ${n} Sorghum_* folders, expected ${want}."
    echo "The transfer is incomplete (or pointed at the wrong root). Resubmit when it matches."
    exit 3
  fi
  echo "split ${split}: ${n}/${want} ok"
done
# ...but a folder exists before its files do, and the loader's cached index would
# keep skipping a folder that was empty when it was first scanned. Check the
# index the trainer will read (one process builds it; the ranks then hit it).
python slurm/delta/check_split_index.py sorghum "${SORGHUM_ROOT}"

if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
  NPROC="${SLURM_GPUS_ON_NODE}"
else
  NPROC=$(echo "${CUDA_VISIBLE_DEVICES:-}" | tr ',' ' ' | wc -w)
fi
[ "${NPROC}" -gt 0 ] 2>/dev/null || { echo "REFUSING: no GPUs in this allocation."; exit 2; }

GLOBAL_BATCH="${GLOBAL_BATCH:-32}"
if [ "${GLOBAL_BATCH}" != 32 ] && [ "${ALLOW_GLOBAL_BATCH:-0}" != 1 ]; then
  echo "REFUSING: GLOBAL_BATCH=${GLOBAL_BATCH}. Every 197,400-step arm is at 32; an arm at"
  echo "another value is comparable to nothing (ALLOW_GLOBAL_BATCH=1 if that is the point)."
  exit 2
fi
if [ $((GLOBAL_BATCH % NPROC)) -ne 0 ]; then
  echo "REFUSING: global batch ${GLOBAL_BATCH} does not split over ${NPROC} GPUs."
  exit 2
fi
PER_GPU=$((GLOBAL_BATCH / NPROC))
if [ "${NPROC}" != 2 ]; then
  echo "WARNING: ${PER_GPU}/GPU x ${NPROC} is not the reference arms' 16 x 2. Global batch is"
  echo "         matched, per-rank BatchNorm stats and Dirichlet draws are not -- say so next"
  echo "         to any result from this run."
fi

# The pipeline is dataloader-bound (~1.6 items/s per worker), so workers set
# throughput; leave one core per rank for the rank itself. Memory scales with
# them: ~2.3 GB RSS per worker plus ~10 GB for model and CUDA context, measured
# on Nova.
CPUS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
NW="${NUM_WORKERS:-$((CPUS / NPROC - 1))}"
[ "${NW}" -ge 1 ] || NW=1
MEM_NEED=$(awk -v r="${NPROC}" -v w="${NW}" 'BEGIN { printf "%.0f", r * w * 2.3 + 10 }')
MEM_HAVE=$(( ${SLURM_MEM_PER_NODE:-0} / 1024 ))
echo "Batch   : ${PER_GPU}/GPU x ${NPROC} GPUs = ${GLOBAL_BATCH} global"
echo "Workers : ${NW}/rank x ${NPROC} = $((NW * NPROC)) -> ~$(awk -v n=$((NW * NPROC)) 'BEGIN { printf "%.0f", n * 1.6 }') items/s"
echo "Memory  : ~${MEM_NEED} GB projected (${NPROC} x ${NW} x 2.3 + 10), ${MEM_HAVE} GB granted"
if [ "${MEM_HAVE}" -gt 0 ] && [ "${MEM_NEED}" -gt "${MEM_HAVE}" ]; then
  echo "WARNING: projected RSS exceeds --mem; expect an oom-kill. Lower NUM_WORKERS or raise --mem."
fi

# srun does not reliably inherit --cpus-per-task from the allocation (Slurm
# 22.05 stopped reading it; 23.11 brought it back as SRUN_CPUS_PER_TASK). Left
# to default, the step can be pinned to ONE core and all ${NW}x${NPROC} workers
# share it -- a run that is merely slow, never an error. Pass it both ways.
# --gpu-bind=none: one task drives every GPU, so do not bind it to a subset.
export SRUN_CPUS_PER_TASK="${CPUS}"
SRUN=(srun --kill-on-bad-exit=1 --ntasks=1 --cpus-per-task="${CPUS}" --gpu-bind=none)

# CUDA sanity inside the SAME kind of step torchrun will get. A wrong-generation
# build fails at the first kernel launch, not at import (CLAUDE.md), so launch
# one on every device, and check the step really has the GPUs and cores.
"${SRUN[@]}" python -c '
import os, sys, torch
gpus, cpus_asked = int(sys.argv[1]), int(sys.argv[2])
if not torch.cuda.is_available():
    sys.exit("CUDA not available in the job step: wrong env, or no GPU granted")
n = torch.cuda.device_count()
print(f"  torch {torch.__version__}, CUDA {torch.version.cuda}, built for {torch.cuda.get_arch_list()}")
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    torch.ones(8, device=f"cuda:{i}").mul_(2).sum().item()
    print(f"  cuda:{i} {p.name} sm_{p.major}{p.minor} {p.total_memory / 2**30:.0f} GiB  kernel ok")
if n != gpus:
    sys.exit(f"step sees {n} GPUs, launcher expects {gpus}")
cpus = len(os.sched_getaffinity(0))
print(f"  step CPU affinity: {cpus} of {cpus_asked} cores asked")
# Halving tolerates SMT/core accounting differences; the failure being caught
# is a step pinned to 1-2 cores.
if cpus < cpus_asked // 2:
    sys.exit(f"step got {cpus} cores of {cpus_asked}: srun did not honour --cpus-per-task")
if cpus < cpus_asked:
    print(f"  WARNING: step has fewer cores than asked", file=sys.stderr)
' "${NPROC}" "${CPUS}"

# Auto-resume: the highest-numbered checkpoint by its NUMBER (e2_arm.sbatch's
# sort -- a path sort puts 75 after 100), walking down past any that are not a
# complete zip. A kill mid-write leaves a truncated .pth as the highest number,
# and resuming from it dies at torch.load with "failed finding central
# directory" on every retry. Reading the zip directory catches exactly that
# without loading 1.3-4 GB of weights.
CKPT_DIR="${OUT_DIR}/checkpoints"
AUTO=""
if [ -d "${CKPT_DIR}" ]; then
  while read -r _ f; do
    if python -c '
import sys, zipfile
try:
    with zipfile.ZipFile(sys.argv[1]) as z:
        sys.exit(0 if any(n.endswith("data.pkl") for n in z.namelist()) else 1)
except Exception:
    sys.exit(1)
' "${f}"; then
      AUTO="${f}"
      break
    fi
    echo "Resume  : skipping unreadable $(basename "${f}") (no zip directory -- killed mid-write?)"
  done < <(for f in "${CKPT_DIR}"/checkpoint_epoch_*.pth; do
             [ -e "${f}" ] || continue
             b=$(basename "${f}" .pth); echo "${b##*_} ${f}"
           done | sort -rn)
fi
# Placed AFTER "$@" below, so argparse's last-wins makes it override a replayed
# explicit --resume on a requeue.
RESUME_ARGS=()
if [ -n "${USER_RESUME}" ] && { [ "${SLURM_RESTART_COUNT:-0}" = 0 ] || [ -z "${AUTO}" ]; }; then
  RESUME="${USER_RESUME}"
  echo "Resume  : ${RESUME}  (explicit)"
elif [ -n "${AUTO}" ]; then
  RESUME="${AUTO}"; RESUME_ARGS=(--resume "${AUTO}")
  [ -n "${USER_RESUME}" ] && echo "Resume  : requeue ${SLURM_RESTART_COUNT} -- the run's own checkpoint overrides the explicit ${USER_RESUME}"
  echo "Resume  : ${RESUME}"
else
  RESUME=""
  echo "Resume  : none, starting from scratch"
fi

# A finished run -- or one whose checkpoints came over from Nova -- resumes at
# epoch N+1 > epochs, trains nothing, and train_worker still overwrites its
# config.json with this launch's values first. Refuse instead.
rb="$(basename "${RESUME:-none}" .pth)"
if [[ "${rb}" == checkpoint_epoch_* ]] && [[ "${rb##*_}" =~ ^[0-9]+$ ]] \
   && [ "${rb##*_}" -ge "${EPOCHS}" ]; then
  echo "REFUSING: ${RESUME} is epoch ${rb##*_} and this run ends at ${EPOCHS}: training would"
  echo "start at $(( ${rb##*_} + 1 )), do nothing, and overwrite config.json on the way. If the run"
  echo "is finished (or its checkpoints came over from Nova), a replicate needs its own config"
  echo "and output_dir, e.g. configs/config_${SLUG}_delta.yaml -> ./outputs/${SLUG}_delta."
  exit 2
fi

"${SRUN[@]}" torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
  train_sorghum_4m.py --config "${CONFIG}" --data_root "${SORGHUM_ROOT}" \
  --batch_size "${PER_GPU}" --num_workers "${NW}" \
  "$@" ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}

echo "Done at $(date)"
