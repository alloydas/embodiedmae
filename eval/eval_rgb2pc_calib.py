"""Calibrate the RGB->PC Chamfer: how good is ~0.001, really?

Compares, over the same validation plants:
  generated-from-RGB vs its own GT      (the claim)
  a DIFFERENT plant's GT vs this GT     (what "wrong plant" scores)
  the dataset mean cloud vs this GT     (what "generic plant shape" scores)
Chamfer is forgiving of diffuse clouds, so these two references say whether the
number reflects plant-specific geometry or just plausible plant-shaped volume.
"""
# Repo root on sys.path: this script lives one level down but imports the
# top-level modules (embodied_mae*, sorghum_dataset*, train_*).
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import numpy as np, torch, yaml
from torch.utils.data import DataLoader
from sorghum_dataset_4m import SorghumDataset4M
from embodied_mae import chamfer_distance
import train_sorghum_4m_distill as DIS

DEV='cuda'; N_BATCH=25
cfg=yaml.safe_load(open('configs/config_4m_distill_15k_all.yaml'))
args=DIS.config_to_namespace(cfg); root=cfg['data']['data_root']
ds=SorghumDataset4M(f"{root}/val", img_size=args.img_size, num_points=args.num_points)
dl=DataLoader(ds,batch_size=16,shuffle=False,num_workers=8,pin_memory=True)
m=DIS.build_model(args,DEV)
DIS.load_weights_into(m,'./outputs/4m_distill_15k_all/best_model.pth',DEV,'distilled')
m.eval()

gen,wrong,meanc,n=[],[],[],0
acc_sum=None; acc_n=0
with torch.no_grad():
    for rgb,depth,pc,params,tv,_ in dl:
        rgb,depth,pc=rgb.to(DEV),depth.to(DEV),pc.to(DEV)
        params,tv=params.to(DEV),tv.to(DEV)
        _,_,(_,_,ppc,_),_=m(rgb,depth,pc,params,tv,visible={'rgb'})
        B=pc.shape[0]
        for i in range(B):
            gen.append(chamfer_distance(ppc[i:i+1],pc[i:i+1]).item())
            j=(i+1)%B
            wrong.append(chamfer_distance(pc[j:j+1],pc[i:i+1]).item())
        acc_sum = pc.mean(0,keepdim=True) if acc_sum is None else acc_sum+pc.mean(0,keepdim=True)
        acc_n += 1
        n+=1
        if n>=N_BATCH: break
mean_cloud = acc_sum/acc_n
with torch.no_grad():
    for rgb,depth,pc,params,tv,_ in dl:
        pc=pc.to(DEV)
        for i in range(pc.shape[0]):
            meanc.append(chamfer_distance(mean_cloud,pc[i:i+1]).item())
        break
f=lambda a:(np.mean(a),np.std(a))
print(f"\nplants scored: {len(gen)}")
print(f"  generated from RGB : {f(gen)[0]:.5f} +/- {f(gen)[1]:.5f}")
print(f"  a DIFFERENT plant  : {f(wrong)[0]:.5f} +/- {f(wrong)[1]:.5f}")
print(f"  dataset mean cloud : {f(meanc)[0]:.5f} +/- {f(meanc)[1]:.5f}   (n={len(meanc)})")
print(f"  ratio wrong/gen    : {f(wrong)[0]/f(gen)[0]:.2f}x")
print(f"  ratio meancloud/gen: {f(meanc)[0]/f(gen)[0]:.2f}x")
print("CALIB_DONE")
