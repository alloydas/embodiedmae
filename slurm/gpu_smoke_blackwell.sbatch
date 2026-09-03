#!/bin/bash
# Validate the cu128 env + repo model on an RTX PRO 6000 (sm_120) before
# committing a multi-day run to it. Model-only: no dataset index build.
#SBATCH --job-name=bwsmoke
#SBATCH --partition=scavenger
#SBATCH --account=mech-ai-scavenger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:20:00
#SBATCH --output=logs/bwsmoke_%j.out
#SBATCH --error=logs/bwsmoke_%j.err

set -uo pipefail
cd /work/mech-ai-scratch/alloy/embodiedmae
source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
conda activate /work/mech-ai-scratch/alloy/.conda/envs/det_cu128
export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

python - <<'PY'
import time, torch, yaml
print("torch", torch.__version__, "| cuda", torch.version.cuda)
print("device:", torch.cuda.get_device_name(0), "| cc:", torch.cuda.get_device_capability(0))
dev='cuda'
# 1. does any kernel run at all on sm_120?
a=torch.randn(4096,4096,device=dev); b=a@a; torch.cuda.synchronize()
print("matmul OK, result finite:", bool(torch.isfinite(b).all()))

# 2. the actual model, real config, real checkpoint-compatible shapes
from embodied_mae_4m import embodied_mae_4m_base
c=yaml.safe_load(open('configs/config_4m_pretrain15k_v2.yaml'))['model']
m=embodied_mae_4m_base(img_size=224, num_pc_tokens=196, target_points=8196,
    pc_loss_weight=c['pc_loss_weight'], max_leaves=c['max_leaves'],
    spline_loss_weight=c['spline_loss_weight'], depth_norm_type=c['depth_norm_type'],
    pc_loss_name=c['loss_name'], qal_threshold=c['qal_threshold'],
    qal_alpha=c['qal_alpha'], qal_use_squared=c['qal_use_squared']).to(dev)
opt=torch.optim.AdamW(m.parameters(), lr=1e-4)
B=8
rgb=torch.randn(B,3,224,224,device=dev); dep=torch.rand(B,1,224,224,device=dev)*0.06
pc=torch.randn(B,8196,3,device=dev)*0.3
par=torch.rand(B,25,9,device=dev); tv=torch.ones(B,25,device=dev)
for i in range(6):
    if i==1: torch.cuda.synchronize(); t0=time.time()
    opt.zero_grad()
    tot,(lr_,ld,lp,lt),_,_=m(rgb,dep,pc,par,tv, mask_ratio=c['mask_ratio'])
    tot.backward(); opt.step()
torch.cuda.synchronize()
dt=(time.time()-t0)/5
print(f"step time (bs{B}, 8196 pts): {dt*1000:.0f} ms   loss {float(tot):.4f}  finite={bool(torch.isfinite(tot))}")
print(f"peak mem: {torch.cuda.max_memory_allocated()/2**30:.1f} GiB of {torch.cuda.get_device_properties(0).total_memory/2**30:.0f} GiB")

# 3. open3d still loads a real ply in this env
import open3d as o3d, glob
f=glob.glob('/work/mech-ai-scratch/alloy/shorgum_data/new_data_50K/Sorghum_15K/val/*/*_nc_cam.ply')[:1]
if f:
    p=o3d.io.read_point_cloud(f[0]); print("open3d ply:", len(p.points), "points")
print("SMOKE_OK")
PY
