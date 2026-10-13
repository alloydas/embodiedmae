#!/bin/bash
# Maize cross-modal distillation on Delta -- the Delta twin of slurm/distill_maize.sbatch
# (Nova). Account, partition and job name go on the sbatch command line:
#
#   sbatch --account=$DELTA_ACCOUNT --partition=gpuA100x4-preempt --job-name=distill_maize_d2 \
#          --cpus-per-task=32 --mem=100G slurm/delta/distill_maize.sbatch
#   ... --gpus-per-node=1 --time=00:40:00 slurm/delta/distill_maize.sbatch \
#          --max_steps 40 --accum_steps 8 --no_wandb --output_dir outputs/_smoke_maize_distill_d2   # smoke
#
# Config: DISTILL_CONFIG (default configs/config_maize_distill_all_d2.yaml). Its
# data_root must be MAIZE_ROOT (the trainer has no --data_root flag).
#
# One run is ~2-3 days on 2x A100-40GB at 821 optimizer steps/epoch x 100 epochs,
# longer than Delta's 2-day cap, so the job requeues ITSELF 10 min before the wall
# (--signal=B:USR1@600 + the trap below; same job id, same log) and every
# segment resumes from the newest checkpoint that opens (save_freq 2 epochs).
# --requeue also covers preemption on the *-preempt partitions. A requeue that
# fails ends the job TIMEOUT: re-run the same sbatch line by hand.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
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

CONFIG="${DISTILL_CONFIG:-configs/config_maize_distill_all_d2.yaml}"
[ -f "${CONFIG}" ] || { echo "no such config: ${CONFIG}"; exit 2; }

# Flags this launcher owns or reads (argparse is last-wins and resolves prefixes).
USER_OUT=""; USER_EPOCHS=""; SELF_REQUEUE=1; pending=""
for a in "$@"; do
  if [ -n "${pending}" ]; then
    case "${pending}" in --output_dir) USER_OUT="${a}" ;; --epochs) USER_EPOCHS="${a}" ;; esac
    pending=""; continue
  fi
  case "${a}" in
    --max_steps|--max_steps=*|--eval_only) SELF_REQUEUE=0 ;;
    --output_dir=*) USER_OUT="${a#*=}" ;;
    --epochs=*)     USER_EPOCHS="${a#*=}" ;;
    --output_dir|--epochs) pending="${a}" ;;
    --resume|--resume=*) echo "REFUSING: ${a}: the launcher resumes from the run's newest checkpoint itself."; exit 2 ;;
    --batch_size|--batch_size=*) echo "REFUSING: ${a}: keep 16/GPU (per-rank BatchNorm); accum_steps sets the global batch."; exit 2 ;;
    --config|--config=*) echo "REFUSING: ${a}: set DISTILL_CONFIG instead."; exit 2 ;;
  esac
done
if [ -n "${USER_EPOCHS}" ] && [ -z "${USER_OUT}" ]; then
  echo "REFUSING: --epochs without --output_dir would write into the real run's output dir."
  exit 2
fi
if [ "${SELF_REQUEUE}" = 0 ] && [ -z "${USER_OUT}" ]; then
  echo "REFUSING: a smoke/eval launch (--max_steps / --eval_only) needs --output_dir outputs/_smoke_..."
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
# 40 GB cards: the val chamfer needs one 12 GiB block; expandable segments keep
# the training allocator from fragmenting the card so that block cannot be found.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1

mapfile -t _cfg < <(python -c '
import sys, yaml
c = yaml.safe_load(open(sys.argv[1])) or {}
print((c.get("checkpointing") or {}).get("output_dir") or "")
print((c.get("training") or {}).get("epochs") or "")
print((c.get("data") or {}).get("data_root") or "")
print((c.get("training") or {}).get("global_batch") or 128)
print((c.get("distill") or {}).get("teacher_checkpoint") or "")
' "${CONFIG}")
OUT_DIR="${USER_OUT:-${_cfg[0]:-}}"
EPOCHS="${USER_EPOCHS:-${_cfg[1]:-}}"
CFG_ROOT="${_cfg[2]:-}"
GLOBAL_BATCH="${_cfg[3]:-128}"
TEACHER="${_cfg[4]:-}"
[ -n "${OUT_DIR}" ] || { echo "REFUSING: ${CONFIG} has no checkpointing.output_dir."; exit 2; }
[[ "${EPOCHS}" =~ ^[0-9]+$ ]] || { echo "REFUSING: cannot read training.epochs from ${CONFIG}."; exit 2; }
if [ "$(readlink -f "${CFG_ROOT}")" != "$(readlink -f "${MAIZE_ROOT}")" ]; then
  echo "REFUSING: ${CONFIG} data_root=${CFG_ROOT} is not MAIZE_ROOT=${MAIZE_ROOT} (the trainer has no --data_root)."
  exit 2
fi
[ -f "${TEACHER}" ] || { echo "REFUSING: teacher checkpoint ${TEACHER} not found."; exit 2; }

echo "JobID   : ${SLURM_JOB_ID:-none}  (restart ${SLURM_RESTART_COUNT:-0})"
echo "Config  : ${CONFIG}  (epochs ${EPOCHS}, global batch ${GLOBAL_BATCH})"
echo "Output  : ${OUT_DIR}"
echo "Teacher : ${TEACHER}"
echo "Data    : ${MAIZE_ROOT}"
echo "Node    : $(hostname)"
echo "Date    : $(date)"
echo "Git     : $(git rev-parse HEAD 2>/dev/null || echo 'not a git checkout')"
git --no-optional-locks status --porcelain --untracked-files=no 2>/dev/null | sed 's/^/  dirty: /' || true
sha256sum train_maize_4m_distill.py embodied_mae_4m_maize.py embodied_mae_4m.py \
          embodied_mae.py maize_dataset_4m.py "${CONFIG}" | sed 's/^/  sha256: /'
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

for split in train val; do
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
if [ $((GLOBAL_BATCH % (PER_GPU * NPROC))) -ne 0 ]; then
  echo "REFUSING: global batch ${GLOBAL_BATCH} is not a multiple of ${PER_GPU} x ${NPROC} GPUs."
  exit 2
fi
ACCUM=$((GLOBAL_BATCH / (PER_GPU * NPROC)))
CPUS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
NW="${NUM_WORKERS:-$((CPUS / NPROC - 1))}"
[ "${NW}" -ge 1 ] || NW=1
echo "Batch   : ${PER_GPU}/GPU x ${NPROC} GPUs x accum ${ACCUM} = ${GLOBAL_BATCH} global"
echo "Workers : ${NW}/rank"

export SRUN_CPUS_PER_TASK="${CPUS}"
SRUN=(srun --kill-on-bad-exit=1 --ntasks=1 --cpus-per-task="${CPUS}" --gpus-per-node="${NPROC}")
"${SRUN[@]}" python -c '
import sys, torch
gpus = int(sys.argv[1])
if not torch.cuda.is_available(): sys.exit("CUDA not available in the job step")
n = torch.cuda.device_count()
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    torch.ones(8, device=f"cuda:{i}").mul_(2).sum().item()
    print(f"  cuda:{i} {p.name} sm_{p.major}{p.minor} {p.total_memory / 2**30:.0f} GiB  kernel ok")
if n != gpus: sys.exit(f"step sees {n} GPUs, launcher expects {gpus}")
' "${NPROC}"

# Auto-resume from the newest checkpoint whose zip directory reads (a kill
# mid-save leaves a truncated highest-numbered file; the trainer's keep_last=2
# keeps one below it).
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
  echo "Resume  : none, warm start from the config's student_init"
fi

requeue_before_wall() {
  if [ "${SELF_REQUEUE}" != 1 ]; then
    echo "$(date) USR1: walltime in ~10 min; smoke/eval launch, NOT requeueing"; return 0
  fi
  echo "$(date) USR1: walltime in ~10 min; requeueing job ${SLURM_JOB_ID:-?} to resume"
  if scontrol requeue "${SLURM_JOB_ID}"; then echo "$(date) requeued ${SLURM_JOB_ID}"
  else echo "$(date) scontrol requeue FAILED: the job will end TIMEOUT; re-run this sbatch line by hand"; fi
}
trap requeue_before_wall USR1

"${SRUN[@]}" torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
  train_maize_4m_distill.py --config "${CONFIG}" --accum_steps "${ACCUM}" --num_workers "${NW}" \
  "$@" ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} &
SRUN_PID=$!
rc=0
wait "${SRUN_PID}" || rc=$?
if [ "${rc}" -gt 128 ] && kill -0 "${SRUN_PID}" 2>/dev/null; then
  rc=0; wait "${SRUN_PID}" || rc=$?
fi
echo "srun exit ${rc} at $(date)"
exit "${rc}"
