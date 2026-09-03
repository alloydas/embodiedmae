#!/bin/bash
#SBATCH --job-name=mmgallery
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=logs/mmgallery_%j.out
#SBATCH --error=logs/mmgallery_%j.err
set -euo pipefail
cd /work/mech-ai-scratch/alloy/embodiedmae
mkdir -p logs
source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
conda activate det
nvidia-smi -L || true
python mask_sweep_multimodal_gallery.py \
    --checkpoint outputs/4m_run_v3/checkpoints/checkpoint_epoch_2400.pth \
    --config configs/config_4m.yaml --output_dir vis_mask_sweep --num_samples 8 --samples 0,1,2,3,4 --seed 1234
echo "Done at $(date)"
