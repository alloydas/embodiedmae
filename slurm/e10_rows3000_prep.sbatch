#!/bin/bash
#SBATCH --job-name=e10_r3000_prep
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
# E10 on the 3000-row upload (2026-11-13): index the unzipped PCD headers (eval/real_rows.build_index_3000 ->
# /work/mech-ai/alloydas/e10_rows_3000/<species>/index.csv) and write the plant split of each species
# (train_real_rows.py --split-only -> outputs/<species>_real_ft_r3000/split.csv). CPU only.
set -euo pipefail
cd /work/mech-ai-scratch/alloy/embodiedmae
source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
conda activate /work/mech-ai/alloy/.conda/envs/det
export PYTHONUNBUFFERED=1
for sp in maize sorghum; do
  python -c "import sys; sys.path[:0]=['.','eval']; import real_rows as r, csv; p=r.build_index_3000('$sp', 16); m=list(csv.DictReader(open(p))); print(p, len(m), 'stops,', sum(int(x['png_ok']) for x in m), 'with photo,', len({r.scan_id(x['plant_id']) for x in m}), 'plants')"
  python train_real_rows.py --species $sp --source rows3000 --split-only --device cpu
done
echo "Done at $(date)"
