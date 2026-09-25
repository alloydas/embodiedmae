#!/bin/bash
#SBATCH --job-name=probe
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/probe_%x_%j.out
#SBATCH --error=logs/probe_%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alloydas@iastate.edu

# The downstream linear probe of locked decision 6.4.
#
#   sbatch slurm/linear_probe.sbatch e2_pc e2_pcrgb e2_pcrgbd e2_pcrgbdt
#   sbatch --job-name=probe_e3 slurm/linear_probe.sbatch e3_1k e3_3k e3_10k e2_pcrgbdt
#
# If it sits PENDING with reason AssocGrpGRES, that is the shared mech-ai GPU
# cap (gres/gpu=17 for the whole account, saturated by other users) and not
# your own usage -- shrinking this request will not help. scavenger runs under
# a different association and bypasses the cap:
#
#   sbatch --partition=scavenger --account=mech-ai-scavenger --requeue \
#          slurm/linear_probe.sbatch e2_pc e2_pcrgb e2_pcrgbd e2_pcrgbdt
#
# Preemption there is survivable because features cache per (run, split): a
# requeued job re-reads the caches it already wrote and resumes at the split
# it lost.
#
# Queueing a copy on BOTH partitions as a hedge is fine: the cache is written
# write-then-rename (os.replace), so a reader never sees a half-written array
# and the worst case is the two jobs duplicating one split's extraction. Cancel
# the loser once one starts, and note each job writes its own
# reports/probe_<name>_<jobid>.csv, so the outputs never collide either.
#
# ONE GPU IS ENOUGH and this is deliberately not the training launcher's shape.
# The probe does a single forward pass per plant over 15,000 plants with no
# backward, so it is dataloader-bound and finishes in minutes; the 2-GPU /
# 320 GB / 80-CPU request the training arms use would sit in the queue for days
# to buy nothing. `nova` rather than `scavenger` because a preempted probe has
# no auto-resume -- it is cheap enough to simply rerun, but only if it is not
# competing with the training arms for the preemptible pool.
#
# Features are cached under outputs/_probe_cache/, so a rerun that only changes
# targets or the ridge alpha grid re-reads nothing.

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: sbatch slurm/linear_probe.sbatch <run-slug> [more slugs...]" >&2
  exit 2
fi

cd /work/mech-ai-scratch/alloy/embodiedmae
mkdir -p logs reports

source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh

# Pick the env from the GPU we actually landed on. This job asks for a bare
# `gpu:1` so it can backfill anywhere, and on this cluster "anywhere" includes
# both A100/V100 (sm_70/sm_80) and the RTX PRO 6000 Blackwell nodes (sm_120).
# `det` is a CUDA 12.4 build and dies on sm_120 with "no kernel image is
# available for execution on the device" -- not at import, but on the first
# kernel launch, which is deep inside the PC embedder's FPS. Constraining the
# gres to a100 instead would work but would queue behind the A100 demand to
# avoid a problem one line of bash solves.
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
case "${GPU_NAME}" in
  *"RTX PRO 6000"*|*Blackwell*) CONDA_ENV=/work/mech-ai-scratch/alloy/.conda/envs/det_cu128 ;;
  *)                            CONDA_ENV=/work/mech-ai/alloy/.conda/envs/det ;;
esac
conda activate "${CONDA_ENV}"

echo "JobID : ${SLURM_JOB_ID:-none}"
echo "Node  : $(hostname)"
echo "Runs  : $*"
echo "Date  : $(date)"
echo "GPU   : ${GPU_NAME}"
echo "Env   : ${CONDA_ENV}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

OUT="reports/probe_${SLURM_JOB_NAME:-probe}_${SLURM_JOB_ID:-manual}.csv"

# CKPT: checkpoint inside each run dir. Default best_model.pth is NOT a matched cut
# across arms (it lands at 42-100% of schedule) -- pass the arm's last saved epoch,
# e.g. CKPT=checkpoints/checkpoint_epoch_600.pth. SPECIES=maize switches to the maize
# probe/E9 scripts. E9_SPLITS runs eval/latent_analysis*.py afterwards on CPU from the
# same feature cache (same --ckpt -> same cache key), so it costs no extra extraction.
CKPT="${CKPT:-best_model.pth}"
case "${SPECIES:-sorghum}" in
  sorghum) PROBE=eval/linear_probe.py;       E9=eval/latent_analysis.py ;;
  maize)   PROBE=eval/linear_probe_maize.py; E9=eval/latent_analysis_maize.py ;;
  *) echo "SPECIES must be sorghum or maize" >&2; exit 2 ;;
esac
echo "Ckpt  : ${CKPT}   Species: ${SPECIES:-sorghum}"

python "${PROBE}" \
  --runs "$@" \
  --ckpt "${CKPT}" \
  --num-workers "${SLURM_CPUS_PER_TASK:-16}" \
  --batch-size 32 \
  --out "${OUT}"

E9_OUT="reports/e9_${SLURM_JOB_NAME:-probe}_${SLURM_JOB_ID:-manual}"
for split in ${E9_SPLITS:-}; do
  python "${E9}" --runs "$@" --ckpt "${CKPT}" --split "${split}" --device cpu \
    --out-dir "${E9_OUT}" --fig-dir "${E9_OUT}/figures" \
    || echo "E9 ${split} FAILED (probe results above are unaffected)"
done

echo "Done at $(date)  ->  ${OUT}${E9_SPLITS:+  and ${E9_OUT}/}"
