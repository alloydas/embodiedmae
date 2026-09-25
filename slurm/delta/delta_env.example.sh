# Everything machine-specific about running this repo on NCSA Delta, in one file.
#
#   cp slurm/delta/delta_env.example.sh slurm/delta/delta_env.sh    # the copy is gitignored
#   $EDITOR slurm/delta/delta_env.sh
#
# Sourced twice per job: by submit.sh on the login node, and by the batch script
# on the compute node. Slurm runs a SPOOLED copy of each .sbatch, so a batch
# script cannot find this file relative to itself -- it reads
# ${SLURM_SUBMIT_DIR}/slurm/delta/delta_env.sh, which is why submit.sh always
# submits from REPO_DIR. submit.sh refuses to run while any value below still
# contains CHANGEME.
#
# Storage paths: Delta's project and work file systems are per allocation
# (/projects/<alloc>, /work/hdd/<alloc> as best known) -- `quota` on Delta lists
# the ones you actually have. Put the data and the repo where `quota` says there
# is room: ~390 GB for the filtered sorghum copy (measured 2.55 MB/folder x 150k;
# CLAUDE.md's 265 GB is stale), ~70 GB for maize, ~24 GB of Nova checkpoints, and
# ~35 GB for the control arm's own 24 checkpoints + best_model. See MANIFEST.md.

# GPU allocation to charge. `accounts` on Delta lists yours; GPU ones end in -delta-gpu.
export DELTA_ACCOUNT="CHANGEME-delta-gpu"

# This repo's checkout on Delta. Jobs run from here and write outputs/, logs/, reports/ under it.
export REPO_DIR="/projects/CHANGEME/${USER}/embodiedmae"

# Sorghum_15K split root: train/ val/ test/ plus assignment.csv and features.csv (the probe's targets).
export SORGHUM_ROOT="/projects/CHANGEME/${USER}/data/Sorghum_15K"

# Maize split root: train/ val/ test/ plus plant_scores.csv. Only probe_maize reads it.
export MAIZE_ROOT="/projects/CHANGEME/${USER}/data/Maize"

# conda's shell hook. Build the env once, on a login node: conda env create -f environment.yml
export CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"

# Env name or absolute prefix. det = torch 2.5 + CUDA 12.4, which has A100 (sm_80) and A40 (sm_86) kernels.
# The reference arms trained under det_cu128 (torch 2.11); Nova-Blackwell needed it, Delta does not.
export CONDA_ENV="det"

# Training partition: 4x A100 per node. `sinfo -s` on Delta lists partitions; -preempt variants exist.
# train_arm.sbatch takes 2 of the 4 GPUs (the reference arms' 16 x 2) and all 64 cores.
export TRAIN_PARTITION="gpuA100x4"
# Uncomment to train on 4 GPUs (8 x 4: same global batch, per-rank BatchNorm differs; the log warns).
# export TRAIN_GPUS=4

# Probe partition: 1 GPU is plenty (one forward pass per plant, no backward).
export EVAL_PARTITION="gpuA40x4"

# online needs compute nodes to reach api.wandb.ai (and `wandb login` once); use offline + `wandb sync` if they cannot.
export WANDB_MODE="online"
