#!/bin/bash
# Run when the 15k v2 pretrain has finished. Snapshots the final teacher to a
# frozen path (best_model.pth is rewritten on every val improvement, so pointing
# a multi-day distill job at it risks a torn read on each scavenger requeue),
# repoints the distill config, and prints the submit command.
set -euo pipefail
cd /work/mech-ai-scratch/alloy/embodiedmae

PRE=./outputs/4m_pretrain_15k_v2_depthfix_qal
SRC=${PRE}/best_model.pth
DST=${PRE}/teacher_final.pth
CFG=configs/config_4m_distill_15k_all.yaml

if squeue -h -u "$USER" -o "%i %j" | grep -q pre15kv2bw; then
  echo "REFUSING: the pretrain job is still running — best_model.pth is live." >&2
  squeue -h -u "$USER" -o "  %i %j %T %M" | grep pre15kv2bw >&2
  exit 1
fi

[ -f "${SRC}" ] || { echo "missing ${SRC}" >&2; exit 1; }
cp "${SRC}" "${DST}"

source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
conda activate det
python - "${DST}" "${CFG}" <<'PY'
import sys, torch, re
dst, cfg = sys.argv[1], sys.argv[2]
c = torch.load(dst, map_location='cpu', weights_only=False)
ep, bv = c.get('epoch'), c.get('best_val_loss')
n = len(c['model_state_dict'])
print(f"teacher snapshot: epoch {ep}  best_val {bv:.4f}  tensors {n}")
assert n == 302, f"unexpected tensor count {n} (expected 302) — architecture mismatch?"
s = open(cfg).read()
s = re.sub(r'(teacher_checkpoint:\s*)\S+', r'\1' + dst, s)
s = re.sub(r'(student_init:\s*)\S+',       r'\1' + dst, s)
open(cfg, 'w').write(s)
print(f"{cfg} repointed at {dst}")
PY

grep -nE "teacher_checkpoint|student_init|batch_size:" "${CFG}"
echo ""
echo "Ready. Submit with:"
echo "  sbatch slurm/distill_15k_all_blackwell.sbatch"
