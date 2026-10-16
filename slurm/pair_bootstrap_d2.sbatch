#!/bin/bash
#SBATCH --job-name=pairboot_d2
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Paired bootstraps over plants for the sorghum v2 (gain-matched) probes: every comparison the
# paper's tables cite, from the feature caches written by 16930422 / 16930423 / 16930425 /
# 16930424. No GPU. Each line writes reports/<name>_d2_<jobid>.csv.
set -euo pipefail
cd /work/mech-ai-scratch/alloy/embodiedmae
source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
conda activate /work/mech-ai/alloy/.conda/envs/det
export PYTHONUNBUFFERED=1
J="${SLURM_JOB_ID:-manual}"
P="python eval/probe_pair_bootstrap.py --species sorghum"
# E2 (CLS): every arm against PC+RGB+D, and depth = PC+RGB+D against PC+RGB
$P --a e2_pc_d2 e2_pcrgb_d2 e2_pcrgbdt_d2 --b e2_pcrgbd_d2 --out reports/e2_delta_d2_${J}.csv
$P --a e2_pcrgbd_d2 --b e2_pcrgb_d2 --out reports/e2_depth_delta_d2_${J}.csv
# controls (CLS): the gated arms against PC+RGB+D and against the as-trained four-stream arm
$P --a e2_pcrgbdt_tg_d2 e2_pcrgbdt_tg40_d2 e2_pcrgbdt_tg100_d2 --b e2_pcrgbd_d2 --out reports/control_delta_d2_${J}.csv
$P --a e2_pcrgbdt_tg_d2 e2_pcrgbdt_tg40_d2 e2_pcrgbdt_tg100_d2 --b e2_pcrgbdt_d2 --out reports/control_delta_vs_pcrgbdt_d2_${J}.csv
# E4 (CLS): small and large against base
$P --a e4_small_d2 e4_large_d2 --b e2_pcrgbdt_d2 --out reports/e4_delta_d2_${J}.csv
# (E3 arms sit at different checkpoints, 6168 / 2024 / 624, and the pair script takes one --ckpt
#  for both sides; their CIs need a per-run ckpt option first.)
# partial clouds: paired deltas vs the full cloud, from the pcview caches of 16930426
python eval/pc_view_bootstrap.py --runs e2_pc_d2 e2_pcrgb_d2 e2_pcrgbd_d2 e2_pcrgbdt_d2 --out reports/pcview_16930426_bootstrap.csv
# E8 (mean pool): ours against the strongest baseline and the supervised row
$P --feature mean --a e2_pcrgbd_d2 e2_pcrgbdt_d2 e2_pc_d2 --b base:embodiedmae --rgb-tag rgb_nobg_g1 --out reports/e8_vs_embodiedmae_delta_d2_${J}.csv
$P --feature mean --a e2_pc_d2 --b base:pointmae --rgb-tag rgb_nobg_g1 --out reports/e8_pc_vs_pointmae_delta_d2_${J}.csv
$P --feature mean cls --a e2_pcrgbd_d2 e2_pcrgbdt_d2 --b e8_supervised_d2 --out reports/e8_supervised_delta_d2_${J}.csv
echo "Done at $(date)"
