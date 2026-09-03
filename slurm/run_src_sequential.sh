#!/bin/bash
# Run the 4 single-source distillation specializations sequentially on the
# current GPU (co-located with the main long run). Each logs separately.
set -uo pipefail
cd /work/mech-ai-scratch/alloy/embodiedmae
source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
conda activate det

for mod in pc depth rgb text; do
  echo "======== START src=${mod}  $(date) ========"
  python -u train_sorghum_4m_distill.py \
      --config "configs/config_4m_distill_src_${mod}.yaml" --world_size 1 \
      > "logs/distill_src_${mod}.log" 2>&1
  echo "======== END   src=${mod}  rc=$?  $(date) ========"
done
echo "######## ALL SOURCE RUNS DONE $(date) ########"
