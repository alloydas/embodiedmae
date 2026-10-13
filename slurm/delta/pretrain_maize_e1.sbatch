#!/bin/bash
# MAIZE E1 (the headline 1000-epoch, all-views, global-batch-256 pretrain) on one
# Delta gpuA100x4 node: configs/config_maize_4m_pretrain_15k_d2.yaml with
# train_maize_4m.py. The maize twin of slurm/pretrain_15k_d2.sbatch (Nova).
#
#   sbatch --account=$DELTA_ACCOUNT --partition=gpuA100x4-preempt \
#          --job-name=tr_maize_e1_d2 --export=ALL,DELTA_ENV_FILE=$REPO/slurm/delta/delta_env.sh \
#          slurm/delta/pretrain_maize_e1.sbatch
#   ... --gpus-per-node=1 --cpus-per-task=16 --mem=60G --time=00:40:00 \
#          slurm/delta/pretrain_maize_e1.sbatch --epochs 1 --max_plants 64 --view_sampling \
#          --no_wandb --output_dir outputs/_smoke_maize_e1_d2            # smoke
#
# GLOBAL BATCH 256 ON ANY GPU COUNT: 16/GPU (the 40 GB cards' limit for the base
# model) x GPUs x accum, accum = 256 / (16 x GPUs). 4 GPUs -> accum 4; 2 -> 8.
#
# ~10 DAYS OF WALL: 1000 epochs x 105,000 views at ~100-130 items/s (loader-bound,
# 60 workers). Longer than the 48 h cap, so the job REQUEUES ITSELF 10 min before
# the wall (--signal=B:USR1@600 + the trap; same job id, same log) and every
# segment resumes from the newest checkpoint that opens (save_freq 5, pruned to
# the newest 3 + every 100th). --requeue also covers preemption. A requeue that
# fails ends the job TIMEOUT: re-run the same sbatch line by hand.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@600
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
mkdir -p logs

CONFIG="${E1_CONFIG:-configs/config_maize_4m_pretrain_15k_d2.yaml}"
[ -f "${CONFIG}" ] || { echo "no such config: ${CONFIG}"; exit 2; }
grep -q accum_steps train_maize_4m.py || { echo "REFUSING: this train_maize_4m.py has no accum_steps."; exit 2; }

# Flags this launcher owns or reads (argparse is last-wins and resolves prefixes).
USER_OUT=""; USER_EPOCHS=""; SMOKE=0; pending=""
for a in "$@"; do
  if [ -n "${pending}" ]; then
    case "${pending}" in --output_dir) USER_OUT="${a}" ;; --epochs) USER_EPOCHS="${a}" ;; esac
    pending=""; continue
  fi
  case "${a}" in
    --output_dir=*) USER_OUT="${a#*=}" ;;
    --epochs=*)     USER_EPOCHS="${a#*=}" ;;
    --output_dir|--epochs) pending="${a}" ;;
    --max_plants|--max_plants=*|--view_sampling) SMOKE=1 ;;
    --resume|--resume=*) echo "REFUSING: ${a}: the launcher resumes from the run's newest checkpoint itself."; exit 2 ;;
    --batch_size|--batch_size=*|--accum_steps|--accum_steps=*)
      echo "REFUSING: ${a}: 16/GPU x accum = 256 is derived from the GPUs granted."; exit 2 ;;
    --config|--config=*) echo "REFUSING: ${a}: set E1_CONFIG instead."; exit 2 ;;
    --data_root|--data_root=*) echo "REFUSING: ${a}: set MAIZE_ROOT in delta_env.sh."; exit 2 ;;
    --num_workers|--num_workers=*) echo "REFUSING: ${a}: set NUM_WORKERS."; exit 2 ;;
  esac
done
if { [ -n "${USER_EPOCHS}" ] || [ "${SMOKE}" = 1 ]; } && [ -z "${USER_OUT}" ]; then
  echo "REFUSING: --epochs / --max_plants / --view_sampling without --output_dir would write"
  echo "a shortened or subsampled run into the real E1 output dir."
  exit 2
fi
SELF_REQUEUE=1; [ -n "${USER_OUT}" ] && SELF_REQUEUE=0

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
# 40 GB cards: the val chamfer needs one 12 GiB block (see distill_maize.sbatch).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mapfile -t _cfg < <(python -c '
import sys, yaml
c = yaml.safe_load(open(sys.argv[1])) or {}
print((c.get("checkpointing") or {}).get("output_dir") or "")
print((c.get("training") or {}).get("epochs") or "")
print((c.get("data") or {}).get("data_root") or "")
print((c.get("data") or {}).get("view_sampling", False))
' "${CONFIG}")
OUT_DIR="${USER_OUT:-${_cfg[0]:-}}"
EPOCHS="${USER_EPOCHS:-${_cfg[1]:-}}"
CFG_ROOT="${_cfg[2]:-}"
[ -n "${OUT_DIR}" ] || { echo "REFUSING: ${CONFIG} has no checkpointing.output_dir."; exit 2; }
[[ "${EPOCHS}" =~ ^[0-9]+$ ]] || { echo "REFUSING: cannot read training.epochs from ${CONFIG}."; exit 2; }
if [ "${_cfg[3]:-False}" != False ]; then
  echo "REFUSING: ${CONFIG} has view_sampling on; E1 pretrains on every view of every plant."; exit 2
fi
if [ "$(readlink -f "${CFG_ROOT}")" != "$(readlink -f "${MAIZE_ROOT}")" ]; then
  echo "NOTE: ${CONFIG} data_root=${CFG_ROOT}; passing --data_root ${MAIZE_ROOT}."
fi

echo "JobID   : ${SLURM_JOB_ID:-none}  (restart ${SLURM_RESTART_COUNT:-0})"
echo "Config  : ${CONFIG}  (epochs ${EPOCHS})"
echo "Output  : ${OUT_DIR}"
echo "Data    : ${MAIZE_ROOT}"
echo "Node    : $(hostname)"
echo "Date    : $(date)"
echo "Git     : $(git rev-parse HEAD 2>/dev/null || echo 'not a git checkout')"
git --no-optional-locks status --porcelain --untracked-files=no 2>/dev/null | sed 's/^/  dirty: /' || true
sha256sum train_maize_4m.py embodied_mae_4m_maize.py embodied_mae_4m.py \
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
PER_GPU=16
GLOBAL_BATCH=256
if [ $((GLOBAL_BATCH % (PER_GPU * NPROC))) -ne 0 ]; then
  echo "REFUSING: global batch ${GLOBAL_BATCH} is not a multiple of ${PER_GPU} x ${NPROC} GPUs."
  exit 2
fi
ACCUM=$((GLOBAL_BATCH / (PER_GPU * NPROC)))
CPUS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
NW="${NUM_WORKERS:-$((CPUS / NPROC - 1))}"
[ "${NW}" -ge 1 ] || NW=1
MEM_NEED=$(awk -v r="${NPROC}" -v w="${NW}" 'BEGIN { printf "%.0f", r * w * 2.3 + 10 }')
MEM_HAVE=$(( ${SLURM_MEM_PER_NODE:-0} / 1024 ))
echo "Batch   : ${PER_GPU}/GPU x ${NPROC} GPUs x accum ${ACCUM} = ${GLOBAL_BATCH} global"
echo "Workers : ${NW}/rank x ${NPROC} = $((NW * NPROC))"
echo "Memory  : ~${MEM_NEED} GB projected, ${MEM_HAVE} GB granted"
if [ "${MEM_HAVE}" -gt 0 ] && [ "${MEM_NEED}" -gt "${MEM_HAVE}" ]; then
  echo "WARNING: projected RSS exceeds --mem; expect an oom-kill. Lower NUM_WORKERS or raise --mem."
fi

export SRUN_CPUS_PER_TASK="${CPUS}"
SRUN=(srun --kill-on-bad-exit=1 --ntasks=1 --cpus-per-task="${CPUS}" --gpus-per-node="${NPROC}")
"${SRUN[@]}" python -c '
import os, sys, torch
gpus, cpus_asked = int(sys.argv[1]), int(sys.argv[2])
if not torch.cuda.is_available(): sys.exit("CUDA not available in the job step")
n = torch.cuda.device_count()
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    torch.ones(8, device=f"cuda:{i}").mul_(2).sum().item()
    print(f"  cuda:{i} {p.name} sm_{p.major}{p.minor} {p.total_memory / 2**30:.0f} GiB  kernel ok")
if n != gpus: sys.exit(f"step sees {n} GPUs, launcher expects {gpus}")
cpus = len(os.sched_getaffinity(0))
print(f"  step CPU affinity: {cpus} of {cpus_asked} cores asked")
if cpus < cpus_asked // 2: sys.exit(f"step got {cpus} cores of {cpus_asked}")
' "${NPROC}" "${CPUS}"

# Auto-resume from the newest checkpoint whose zip directory reads.
CKPT_DIR="${OUT_DIR}/checkpoints"
RESUME=""
if [ -d "${CKPT_DIR}" ]; then
  while read -r _ f; do
    if python -c '
import sys, zipfile
try:
    with zipfile.ZipFile(sys.argv[1]) as z:
        sys.exit(0 if any(n.endswith("data.pkl") for n in z.namelist()) else 1)
except Exception:
    sys.exit(1)
' "${f}"; then RESUME="${f}"; break; fi
    echo "Resume  : skipping unreadable $(basename "${f}")"
  done < <(for f in "${CKPT_DIR}"/checkpoint_epoch_*.pth; do
             [ -e "${f}" ] || continue
             b=$(basename "${f}" .pth); echo "${b##*_} ${f}"
           done | sort -rn)
fi
RESUME_ARGS=()
if [ -n "${RESUME}" ]; then
  rb="$(basename "${RESUME}" .pth)"
  if [ "${rb##*_}" -ge "${EPOCHS}" ] 2>/dev/null; then
    echo "DONE: ${RESUME} is epoch ${rb##*_} of ${EPOCHS}; nothing to train."
    exit 0
  fi
  RESUME_ARGS=(--resume "${RESUME}"); echo "Resume  : ${RESUME}"
else
  echo "Resume  : none, starting from scratch"
fi

requeue_before_wall() {
  if [ "${SELF_REQUEUE}" != 1 ]; then
    echo "$(date) USR1: walltime in ~10 min; smoke launch, NOT requeueing"; return 0
  fi
  echo "$(date) USR1: walltime in ~10 min; requeueing job ${SLURM_JOB_ID:-?} to resume"
  if scontrol requeue "${SLURM_JOB_ID}"; then echo "$(date) requeued ${SLURM_JOB_ID}"
  else echo "$(date) scontrol requeue FAILED: the job will end TIMEOUT; re-run this sbatch line by hand"; fi
}
trap requeue_before_wall USR1

"${SRUN[@]}" torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
  train_maize_4m.py --config "${CONFIG}" --data_root "${MAIZE_ROOT}" \
  --batch_size "${PER_GPU}" --accum_steps "${ACCUM}" --num_workers "${NW}" \
  "$@" ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} &
SRUN_PID=$!
rc=0
wait "${SRUN_PID}" || rc=$?
if [ "${rc}" -gt 128 ] && kill -0 "${SRUN_PID}" 2>/dev/null; then
  rc=0; wait "${SRUN_PID}" || rc=$?
fi
echo "srun exit ${rc} at $(date)"
exit "${rc}"
