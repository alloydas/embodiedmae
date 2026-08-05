#!/bin/bash
#SBATCH --job-name=embodiedmae4m
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=228G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/embmae4m_%j.out
#SBATCH --error=logs/embmae4m_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yongyun@iastate.edu

set -euo pipefail

cd /work/mech-ai-scratch/yongyun/embodiedmae4m
mkdir -p logs

module load micromamba

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/.micromamba}"

set +u
eval "$(micromamba shell hook --shell=bash)"
micromamba activate /work/mech-ai-scratch/yongyun/envs/myenv
set -u

PYTHON_BIN="/work/mech-ai-scratch/yongyun/envs/myenv/bin/python"

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "ERROR: Python executable does not exist: ${PYTHON_BIN}"
    exit 1
fi

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN
export KEOPS_CACHE_FOLDER="${SLURM_TMPDIR:-/tmp}/keops_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/xdg_${SLURM_JOB_ID}"
mkdir -p "${KEOPS_CACHE_FOLDER}" "${XDG_CACHE_HOME}"

echo "JobID         : ${SLURM_JOB_ID}"
echo "Node          : $(hostname)"
echo "Date          : $(date)"
echo "Python        : ${PYTHON_BIN}"
echo "CUDA devices  : ${CUDA_VISIBLE_DEVICES:-<unset>}"

nvidia-smi -L || true

"${PYTHON_BIN}" -c "
import sys
import torch
import geomloss
import yaml

print('python:', sys.executable)
print('torch:', torch.__version__)
print('torch path:', torch.__file__)
print('cuda available:', torch.cuda.is_available())
print('cuda devices:', torch.cuda.device_count())
print('geomloss:', getattr(geomloss, '__version__', 'installed'))

with open('config_4m.yaml') as config_file:
    model_config = yaml.safe_load(config_file)['model']
sinkhorn_backend = model_config.get('sinkhorn_backend', 'tensorized')
sinkhorn_points = model_config.get('sinkhorn_num_points', 512)
sinkhorn_weight = model_config.get('sinkhorn_loss_weight', 0.0)
if sinkhorn_backend != 'tensorized':
    raise RuntimeError(
        'This job environment requires sinkhorn_backend=tensorized; '
        'the PyKeOps CUDA backend is unavailable on this node.')
if not 0 < sinkhorn_points <= 5000:
    raise RuntimeError('sinkhorn_num_points must be between 1 and 5000')
print('sinkhorn config:', sinkhorn_backend, sinkhorn_points, 'points')

if torch.cuda.is_available() and sinkhorn_weight > 0:
    warmup_points = min(32, sinkhorn_points)
    x = torch.rand(2, warmup_points, 3, device='cuda', requires_grad=True)
    y = torch.rand(2, warmup_points, 3, device='cuda')
    sinkhorn = geomloss.SamplesLoss(
        loss='sinkhorn',
        p=model_config.get('sinkhorn_p', 2),
        blur=model_config.get('sinkhorn_blur', 0.02),
        scaling=model_config.get('sinkhorn_scaling', 0.8),
        debias=model_config.get('sinkhorn_debias', True),
        backend='tensorized')
    warmup_loss = sinkhorn(x, y).mean()
    warmup_loss.backward()
    torch.cuda.synchronize()
    print('sinkhorn tensorized CUDA warmup:', float(warmup_loss.detach()))
"

NPROC=$("${PYTHON_BIN}" -c "import torch; print(torch.cuda.device_count())")

if [ "${NPROC}" -lt 1 ]; then
    echo "ERROR: PyTorch cannot see any CUDA GPUs."
    exit 1
fi

echo "Launching training with ${NPROC} GPU processes"

srun --export=ALL --kill-on-bad-exit=1 \
    "${PYTHON_BIN}" -m torch.distributed.run \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="${NPROC}" \
        train_sorghum_4m.py \
        --config config_4m.yaml

echo "Done at $(date)"
