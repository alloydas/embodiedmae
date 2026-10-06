#!/bin/bash
#SBATCH --job-name=split_gap
#SBATCH --partition=scavenger
#SBATCH --account=mech-ai-scavenger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=logs/split_gap_%x_%j.out
#SBATCH --error=logs/split_gap_%x_%j.out

# Train vs val vs test reconstruction under the trainer's own random masking
# (eval/eval_single_source_pc.py "ref" mode: all modalities, Dirichlet mask at
# the run's mask_ratio, real params), one deterministic view per plant, the
# first N plants of each split. Answers "does it train well but fail on test?"
# with the SAME checkpoint, masking and metric on every split.
#   sbatch --job-name=gap_sorghum slurm/split_gap.sbatch sorghum e2_pcrgbdt 750
#   sbatch --job-name=gap_maize   slurm/split_gap.sbatch maize   maize_4m   750
# CPU only (--gres=NONE sidesteps mech-ai's GPU cap). Rows flush atomically, so
# a preempted, requeued job resumes where it stopped.

set -euo pipefail
SPECIES="${1:?usage: split_gap.sbatch <sorghum|maize> <run> [n_plants] [ckpt]}"
RUN="${2:?usage: split_gap.sbatch <sorghum|maize> <run> [n_plants] [ckpt]}"
N="${3:-750}"
CKPT="${4:-checkpoints/checkpoint_epoch_600.pth}"

REPO=/work/mech-ai-scratch/yongyun/embodiedmae4m_distilation
# The reference runs live in the main checkout, not this one.
OUTPUTS=/work/mech-ai-scratch/alloy/embodiedmae/outputs
PYTHON_BIN=/work/mech-ai-scratch/yongyun/envs/myenv/bin/python
cd "${REPO}"
mkdir -p logs reports/split_gap
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16

echo "JobID ${SLURM_JOB_ID} (restart ${SLURM_RESTART_COUNT:-0})  ${SPECIES} ${RUN} ${CKPT}  n=${N}  $(hostname)  $(date)"
for SPLIT in train val test; do
  OUT="reports/split_gap/${SPECIES}_${RUN}_${SPLIT}_n${N}"
  if [ -f "${OUT}.json" ]; then echo "done already: ${OUT}.json"; continue; fi
  "${PYTHON_BIN}" eval/eval_single_source_pc.py --species "${SPECIES}" --run "${RUN}" \
    --outputs-root "${OUTPUTS}" --ckpt "${CKPT}" --sources '' --split "${SPLIT}" \
    --n-plants "${N}" --threads 16 --workers 6 --out "${OUT}"
done
echo "Done at $(date)"
