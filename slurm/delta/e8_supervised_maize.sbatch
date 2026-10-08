#!/bin/bash
# E8 supervised-from-scratch baseline, MAIZE, on Delta -- the maize twin of the
# Nova slurm/e8_supervised.sbatch + e8_supervised_score.sbatch, in the shape of
# slurm/delta/train_maize_arm.sbatch. Account, partition and job name go on the
# sbatch command line (delta_env.sh holds the rest):
#
#   # train (2 GPUs x 16 = global batch 32; the trainer refuses any other shape)
#   sbatch --account=bepk-delta-gpu --partition=gpuA100x4-preempt --job-name=tr_maize_e8_supervised_d2 \
#          --cpus-per-task=32 --mem=100G \
#          --export=ALL,DELTA_ENV_FILE=/work/hdd/bepk/adas14/embodiedmae/slurm/delta/delta_env.sh \
#          slurm/delta/e8_supervised_maize.sbatch
#   # score (after epoch 600): its own head, then the cls and the mean probes, on
#   # ONE GPU with the probe launcher's loader shape (16 CPUs -> 15 workers, batch 32)
#   SCORE=1 sbatch --account=bepk-delta-gpu --partition=gpuA40x4 --job-name=e8_score_maize \
#          --gpus-per-node=1 --cpus-per-task=16 --mem=64G --time=04:00:00 \
#          --export=ALL,SCORE=1,DELTA_ENV_FILE=... slurm/delta/e8_supervised_maize.sbatch
#   # smoke: never into the real output dir
#   ... slurm/delta/e8_supervised_maize.sbatch --smoke_plants 64 --epochs 2 --no_wandb \
#          --output_dir outputs/_smoke_maize_e8_supervised
#
# Config: E8_CONFIG (default configs/config_maize_e8_supervised_d2.yaml). Extra
# args go to train_supervised_maize_4m.py. --requeue covers preemption and node
# failure, NOT a walltime TIMEOUT; re-run the same sbatch line and the
# auto-resume below caps the loss at save_freq epochs.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/delta_%x_%j.out
#SBATCH --error=logs/delta_%x_%j.err

set -euo pipefail

ENV_FILE="${DELTA_ENV_FILE:-${SLURM_SUBMIT_DIR:-.}/slurm/delta/delta_env.sh}"
if [ ! -f "${ENV_FILE}" ]; then
  echo "REFUSING: no ${ENV_FILE}. Copy delta_env.example.sh to delta_env.sh first."
  exit 2
fi
# shellcheck source-path=SCRIPTDIR source=delta_env.example.sh
source "${ENV_FILE}"
: "${REPO_DIR:?REPO_DIR unset in ${ENV_FILE}}" "${MAIZE_ROOT:?}" "${CONDA_SH:?}" "${CONDA_ENV:?}"
cd "${REPO_DIR}"
mkdir -p logs reports

CONFIG="${E8_CONFIG:-configs/config_maize_e8_supervised_d2.yaml}"
[ -f "${CONFIG}" ] || { echo "no such config: ${CONFIG}"; exit 2; }
TRAINER=train_supervised_maize_4m.py

# Flags this launcher owns or reads. argparse is last-wins AND resolves
# abbreviations, so resolve every extra flag against the trainer's own list.
mapfile -t TRAINER_OPTS < <(grep -oE "add_argument\('--[a-z_]+'" "${TRAINER}" \
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
USER_OUT=""; USER_EPOCHS=""; USER_RESUME=""; SMOKE=0; pending=""
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
    --data_root)   echo "REFUSING: ${a}: set MAIZE_ROOT in delta_env.sh."; exit 2 ;;
    --num_workers) echo "REFUSING: ${a}: set NUM_WORKERS (the memory check reads it)."; exit 2 ;;
    --config)      echo "REFUSING: ${a}: set E8_CONFIG instead."; exit 2 ;;
    --score_head)  echo "REFUSING: ${a}: use SCORE=1 (it also runs the two probes)."; exit 2 ;;
    --smoke_plants) SMOKE=1 ;;
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
if { [ -n "${USER_EPOCHS}" ] || [ "${SMOKE}" = 1 ]; } && [ -z "${USER_OUT}" ]; then
  echo "REFUSING: --epochs / --smoke_plants without --output_dir would write into the real run dir."
  echo "Add e.g. --output_dir outputs/_smoke_maize_e8_supervised."
  exit 2
fi

set +u
# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
set -u

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-600}"
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1

mapfile -t _cfg < <(python -c '
import sys, yaml
c = yaml.safe_load(open(sys.argv[1])) or {}
print((c.get("checkpointing") or {}).get("output_dir") or "")
print((c.get("training") or {}).get("epochs") or "")
print(c.get("entry_point") or "")
print((c.get("data") or {}).get("data_root") or "")
' "${CONFIG}")
OUT_DIR="${USER_OUT:-${_cfg[0]:-}}"
EPOCHS="${USER_EPOCHS:-${_cfg[1]:-}}"
ENTRY="${_cfg[2]:-}"
CFG_ROOT="${_cfg[3]:-}"
[ -n "${OUT_DIR}" ] || { echo "REFUSING: ${CONFIG} has no checkpointing.output_dir."; exit 2; }
[[ "${EPOCHS}" =~ ^[0-9]+$ ]] || { echo "REFUSING: cannot read training.epochs from ${CONFIG}."; exit 2; }
if [ "${ENTRY}" != "${TRAINER}" ]; then
  echo "REFUSING: ${CONFIG} has entry_point '${ENTRY}', this launcher runs ${TRAINER}."
  exit 2
fi
RUN="$(basename "${OUT_DIR}")"

echo "JobID   : ${SLURM_JOB_ID:-none}  (restart ${SLURM_RESTART_COUNT:-0})"
echo "Mode    : $([ "${SCORE:-0}" = 1 ] && echo score || echo train)"
echo "Config  : ${CONFIG}  (epochs ${EPOCHS})"
echo "Output  : ${OUT_DIR}"
echo "Data    : ${MAIZE_ROOT}"
echo "Node    : $(hostname)"
echo "Date    : $(date)"
echo "Git     : $(git rev-parse HEAD 2>/dev/null || echo 'not a git checkout')"
git --no-optional-locks status --porcelain --untracked-files=no 2>/dev/null | sed 's/^/  dirty: /' || true
sha256sum "${TRAINER}" eval/linear_probe_maize.py embodied_mae_4m_maize.py embodied_mae_4m.py \
          embodied_mae.py maize_dataset_4m.py "${CONFIG}" | sed 's/^/  sha256: /'
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

for split in train val test; do
  case "${split}" in train) want=105000 ;; *) want=22500 ;; esac
  # shellcheck disable=SC2010
  n=$(ls -f "${MAIZE_ROOT}/${split}" 2>/dev/null | grep -c '^plant_' || true)
  if [ "${n}" -ne "${want}" ]; then
    echo "REFUSING TO START: ${MAIZE_ROOT}/${split} has ${n} plant_* folders, expected ${want}."
    exit 3
  fi
  echo "split ${split}: ${n}/${want} ok"
done
python slurm/delta/check_split_index.py maize "${MAIZE_ROOT}"

if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
  NPROC="${SLURM_GPUS_ON_NODE}"
else
  NPROC=$(echo "${CUDA_VISIBLE_DEVICES:-}" | tr ',' ' ' | wc -w)
fi
[ "${NPROC}" -gt 0 ] 2>/dev/null || { echo "REFUSING: no GPUs in this allocation."; exit 2; }
CPUS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
export SRUN_CPUS_PER_TASK="${CPUS}"
# NOT --gpu-bind=none: on Delta it reduces the step to ONE GPU (job 22376656).
SRUN=(srun --kill-on-bad-exit=1 --ntasks=1 --cpus-per-task="${CPUS}" --gpus-per-node="${NPROC}")

"${SRUN[@]}" python -c '
import os, sys, torch
gpus, cpus_asked = int(sys.argv[1]), int(sys.argv[2])
if not torch.cuda.is_available():
    sys.exit("CUDA not available in the job step: wrong env, or no GPU granted")
n = torch.cuda.device_count()
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    torch.ones(8, device=f"cuda:{i}").mul_(2).sum().item()
    print(f"  cuda:{i} {p.name} sm_{p.major}{p.minor} {p.total_memory / 2**30:.0f} GiB  kernel ok")
if n != gpus:
    sys.exit(f"step sees {n} GPUs, launcher expects {gpus}")
cpus = len(os.sched_getaffinity(0))
print(f"  step CPU affinity: {cpus} of {cpus_asked} cores asked")
if cpus < cpus_asked // 2:
    sys.exit(f"step got {cpus} cores of {cpus_asked}: srun did not honour --cpus-per-task")
' "${NPROC}" "${CPUS}"

# ── SCORE mode: head on the probe's features, then the cls and mean probes ──
if [ "${SCORE:-0}" = 1 ]; then
  CKPT="checkpoints/checkpoint_epoch_${EPOCHS}.pth"
  if [ ! -f "${OUT_DIR}/${CKPT}" ]; then
    echo "${OUT_DIR}/${CKPT} does not exist: training has not reached epoch ${EPOCHS}." >&2
    echo "Resubmit with SCORE=1 once it has." >&2
    exit 3
  fi
  # The probe launcher's loader shape (slurm/delta/probe_maize.sbatch: 16 CPUs
  # -> 15 workers, batch 32, seed 0). The point subsets depend on it, so the head
  # and the probes must share it to share features; run this mode at 16 CPUs.
  PNW="${PROBE_WORKERS:-$((CPUS > 1 ? CPUS - 1 : 1))}"
  CACHE="${PROBE_CACHE_DIR_MAIZE:-outputs/_probe_cache_maize}"
  TAG="${SLURM_JOB_NAME:-e8_score_maize}_${SLURM_JOB_ID:-manual}"
  echo "Score   : ${OUT_DIR}/${CKPT}  workers ${PNW}  cache ${CACHE}"
  "${SRUN[@]}" python "${TRAINER}" --config "${CONFIG}" --data_root "${MAIZE_ROOT}" \
    --score_head --score_ckpt "${CKPT}" --score_out "reports/probe_${TAG}_head.csv" \
    --probe_cache_dir "${CACHE}" --probe_batch_size 32 --probe_num_workers "${PNW}" "$@"
  "${SRUN[@]}" python eval/linear_probe_maize.py --runs "${RUN}" --ckpt "${CKPT}" \
    --data-root "${MAIZE_ROOT}" --cache-dir "${CACHE}" --feature cls --seed 0 --repeats 1 \
    --split-set train,val,test --batch-size 32 --num-workers "${PNW}" \
    --out "reports/probe_${TAG}_cls.csv"
  "${SRUN[@]}" python eval/linear_probe_maize.py --runs "${RUN}" --ckpt "${CKPT}" \
    --data-root "${MAIZE_ROOT}" --cache-dir "${CACHE}" --feature mean --seed 0 --repeats 1 \
    --split-set train,val,test --batch-size 32 --num-workers "${PNW}" \
    --out "reports/probe_${TAG}_mean.csv"
  echo "Done at $(date)  ->  reports/probe_${TAG}_{head,cls,mean}.csv"
  exit 0
fi

# ── TRAIN mode ──────────────────────────────────────────────────────────────
GLOBAL_BATCH=32
if [ $((GLOBAL_BATCH % NPROC)) -ne 0 ]; then
  echo "REFUSING: global batch ${GLOBAL_BATCH} does not split over ${NPROC} GPUs."; exit 2
fi
PER_GPU=$((GLOBAL_BATCH / NPROC))
if [ "${NPROC}" != 2 ] && [ "${SMOKE}" != 1 ]; then
  echo "WARNING: ${PER_GPU}/GPU x ${NPROC} is not the E2 shape 16 x 2; the trainer will refuse"
  echo "         it unless --allow_unmatched is among the extra args."
fi
NW="${NUM_WORKERS:-$((CPUS / NPROC - 1))}"
[ "${NW}" -ge 1 ] || NW=1
MEM_NEED=$(awk -v r="${NPROC}" -v w="${NW}" 'BEGIN { printf "%.0f", r * w * 2.3 + 10 }')
MEM_HAVE=$(( ${SLURM_MEM_PER_NODE:-0} / 1024 ))
echo "Batch   : ${PER_GPU}/GPU x ${NPROC} GPUs = ${GLOBAL_BATCH} global"
echo "Workers : ${NW}/rank x ${NPROC}"
echo "Memory  : ~${MEM_NEED} GB projected, ${MEM_HAVE} GB granted"
if [ "${MEM_HAVE}" -gt 0 ] && [ "${MEM_NEED}" -gt "${MEM_HAVE}" ]; then
  echo "WARNING: projected RSS exceeds --mem; expect an oom-kill. Lower NUM_WORKERS or raise --mem."
fi

# Auto-resume from the highest-numbered checkpoint whose zip directory reads.
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
' "${f}"; then AUTO="${f}"; break; fi
    echo "Resume  : skipping unreadable $(basename "${f}") (no zip directory -- killed mid-write?)"
  done < <(for f in "${CKPT_DIR}"/checkpoint_epoch_*.pth; do
             [ -e "${f}" ] || continue
             b=$(basename "${f}" .pth); echo "${b##*_} ${f}"
           done | sort -rn)
fi
RESUME_ARGS=()
if [ -n "${USER_RESUME}" ] && { [ "${SLURM_RESTART_COUNT:-0}" = 0 ] || [ -z "${AUTO}" ]; }; then
  RESUME="${USER_RESUME}"; echo "Resume  : ${RESUME}  (explicit)"
elif [ -n "${AUTO}" ]; then
  RESUME="${AUTO}"; RESUME_ARGS=(--resume "${AUTO}"); echo "Resume  : ${RESUME}"
else
  RESUME=""; echo "Resume  : none, starting from scratch"
fi
rb="$(basename "${RESUME:-none}" .pth)"
if [[ "${rb}" == checkpoint_epoch_* ]] && [[ "${rb##*_}" =~ ^[0-9]+$ ]] \
   && [ "${rb##*_}" -ge "${EPOCHS}" ]; then
  echo "DONE: ${RESUME} is epoch ${rb##*_} and this run ends at ${EPOCHS}; nothing to train."
  echo "Score it: SCORE=1 sbatch ... --gpus-per-node=1 --cpus-per-task=16 slurm/delta/e8_supervised_maize.sbatch"
  exit 0
fi

"${SRUN[@]}" torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
  "${TRAINER}" --config "${CONFIG}" --data_root "${MAIZE_ROOT}" \
  --batch_size "${PER_GPU}" --num_workers "${NW}" \
  "$@" ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}

echo "Done at $(date)"
