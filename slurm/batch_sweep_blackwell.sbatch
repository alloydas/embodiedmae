#!/bin/bash
#SBATCH --job-name=bsweep
#SBATCH --partition=scavenger
#SBATCH --account=mech-ai-scavenger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=logs/bsweep_%j.out
#SBATCH --error=logs/bsweep_%j.err
set -uo pipefail
cd /work/mech-ai-scratch/alloy/embodiedmae
source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
conda activate /work/mech-ai-scratch/alloy/.conda/envs/det_cu128
export PYTHONUNBUFFERED=1
python - <<'PY'
import time, torch, yaml
from embodied_mae_4m import embodied_mae_4m_base
c=yaml.safe_load(open('configs/config_4m_pretrain15k_v2.yaml'))['model']
dev='cuda'
tot_mem=torch.cuda.get_device_properties(0).total_memory/2**30
print(torch.cuda.get_device_name(0), f"{tot_mem:.0f} GiB\n")
print(f"{'bs/GPU':>7} {'step ms':>9} {'samples/s':>10} {'peak GiB':>9} {'status':>8}")
for B in (8, 16, 24, 32, 48, 64):
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        m=embodied_mae_4m_base(img_size=224, num_pc_tokens=196, target_points=8196,
            pc_loss_weight=c['pc_loss_weight'], max_leaves=c['max_leaves'],
            spline_loss_weight=c['spline_loss_weight'], depth_norm_type=c['depth_norm_type'],
            pc_loss_name=c['loss_name'], qal_threshold=c['qal_threshold'],
            qal_alpha=c['qal_alpha'], qal_use_squared=c['qal_use_squared']).to(dev)
        opt=torch.optim.AdamW(m.parameters(), lr=1e-4)
        rgb=torch.randn(B,3,224,224,device=dev); dep=torch.rand(B,1,224,224,device=dev)*0.06
        pc=torch.randn(B,8196,3,device=dev)*0.3
        par=torch.rand(B,25,9,device=dev); tv=torch.ones(B,25,device=dev)
        for i in range(5):
            if i==1: torch.cuda.synchronize(); t0=time.time()
            opt.zero_grad()
            t,_,_,_=m(rgb,dep,pc,par,tv, mask_ratio=c['mask_ratio'])
            t.backward(); opt.step()
        torch.cuda.synchronize()
        dt=(time.time()-t0)/4
        pk=torch.cuda.max_memory_allocated()/2**30
        print(f"{B:>7} {dt*1000:>9.0f} {B/dt:>10.1f} {pk:>9.1f} {'ok':>8}")
        del m,opt,rgb,dep,pc,par,tv
    except torch.cuda.OutOfMemoryError:
        print(f"{B:>7} {'-':>9} {'-':>10} {'-':>9} {'OOM':>8}")
        break
    except Exception as e:
        print(f"{B:>7} {'-':>9} {'-':>10} {'-':>9}  {type(e).__name__}")
        break
print("\nSWEEP_DONE")
PY
