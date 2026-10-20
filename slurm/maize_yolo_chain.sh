#!/bin/bash
# Submit the maize YOLO run as N chained 2-day scavenger jobs (default 7 = 2 weeks). Each starts
# when the previous one ENDS for any reason (afterany: finished, timed out or failed) and resumes
# from the run's latest checkpoint (slurm/maize_yolo.sbatch auto-resume), so a walltime TIMEOUT --
# which --requeue does not cover -- costs nothing. A job that starts after the run is complete
# exits at once (the trainer sees the final epoch). Cancel the rest with: scancel <ids printed>.
#   slurm/maize_yolo_chain.sh [N] [arm] [seed]
set -euo pipefail
N=${1:-7}; ARM=${2:-row_yolo}; SEED=${3:-1}
cd /work/mech-ai-scratch/yongyun/embodiedmae4m_distilation
dep=""; ids=()
for i in $(seq 1 "$N"); do
  j=$(sbatch --parsable ${dep:+--dependency=afterany:$dep} --job-name="maize_${ARM}_s${SEED}" \
      slurm/maize_yolo.sbatch "$ARM" "$SEED")
  ids+=("$j"); dep=$j
done
echo "chain: ${ids[*]}"
