"""How well does RGB -> point cloud survive a partially hidden photograph?

The model's `visible=` path is all-or-nothing per modality, so this replicates
forward_encoder_select but keeps only a random subset of the RGB patch tokens
(standard MAE shuffle/restore), with depth, PC and spline still fully masked.
Produces a qualitative plate at several mask levels plus an aggregate Chamfer
sweep over many validation plants.
"""
# Repo root on sys.path: this script lives one level down but imports the
# top-level modules (embodied_mae*, sorghum_dataset*, train_*).
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import numpy as np, torch, yaml, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, Subset

from sorghum_dataset_4m import SorghumDataset4M
from embodied_mae import chamfer_distance
import train_sorghum_4m_distill as DIS

DEV='cuda'
CKPT='./outputs/4m_distill_15k_all/best_model.pth'
OUT=Path('figures_fixed'); OUT.mkdir(exist_ok=True)
LEVELS=[0.0, 0.20, 0.50, 0.80, 0.95]     # fraction of RGB tokens HIDDEN
PLANT='Sorghum_10001_00'
N_BATCH=13                                # ~208 plants per level for the sweep
GT_C, GEN_C = '#5b6870', '#1baf7a'

@torch.no_grad()
def forward_partial_rgb(m, rgb, depth, pc, params, tv, mask_frac, seed=0):
    """Encoder with a random (1-mask_frac) subset of RGB tokens visible; depth,
    pc and text contribute zero tokens. Returns the decoder predictions."""
    x_rgb   = m.rgb_embed(rgb)     + m.pos_embed_2d   + m.modality_embed_rgb
    x_depth = m.depth_embed(depth) + m.pos_embed_2d   + m.modality_embed_depth
    x_pc    = m.pc_embed(pc)       + m.pos_embed_pc   + m.modality_embed_pc
    x_text  = m.param_embed(params)+ m.pos_embed_text + m.modality_embed_text
    B, L, D = x_rgb.shape
    dev = rgb.device
    g = torch.Generator(device=dev).manual_seed(seed)
    noise = torch.rand(B, L, device=dev, generator=g)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    len_keep = max(1, int(round(L * (1.0 - mask_frac))))
    ids_keep = ids_shuffle[:, :len_keep]
    xr_v = torch.gather(x_rgb, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
    mr = torch.ones(B, L, device=dev); mr.scatter_(1, ids_keep, 0.0)

    ar = lambda n: torch.arange(n, device=dev).unsqueeze(0).expand(B, n)
    md = torch.ones(B, x_depth.shape[1], device=dev)
    mp = torch.ones(B, x_pc.shape[1],    device=dev)
    mt = torch.ones(B, x_text.shape[1],  device=dev)

    x = torch.cat([xr_v, x_depth[:, :0], x_pc[:, :0], x_text[:, :0]], dim=1)
    x = torch.cat([m.cls_token.expand(B, -1, -1), x], dim=1)
    for blk in m.encoder_blocks: x = blk(x)
    x = m.encoder_norm(x)

    out = m.forward_decoder(x, ids_restore, ar(x_depth.shape[1]),
                            ar(x_pc.shape[1]), ar(x_text.shape[1]),
                            len_keep, 0, 0, 0)
    pred_rgb, pred_depth, pred_pc, pred_par = out[:4]
    return pred_pc, mr, len_keep

cfg=yaml.safe_load(open('configs/config_4m_distill_15k_all.yaml'))
args=DIS.config_to_namespace(cfg); root=cfg['data']['data_root']
ds=SorghumDataset4M(f"{root}/val", img_size=args.img_size, num_points=args.num_points)
names=[p.name for p in ds.samples]
m=DIS.build_model(args,DEV); DIS.load_weights_into(m,CKPT,DEV,'distilled'); m.eval()

# ── aggregate sweep ───────────────────────────────────────────────────────
dl=DataLoader(ds,batch_size=16,shuffle=False,num_workers=8,pin_memory=True)
agg={lv:[] for lv in LEVELS}
with torch.no_grad():
    for bi,(rgb,depth,pc,params,tv,_) in enumerate(dl):
        rgb,depth,pc=rgb.to(DEV),depth.to(DEV),pc.to(DEV)
        params,tv=params.to(DEV),tv.to(DEV)
        for lv in LEVELS:
            ppc,_,_=forward_partial_rgb(m,rgb,depth,pc,params,tv,lv,seed=bi)
            for i in range(pc.shape[0]):
                agg[lv].append(chamfer_distance(ppc[i:i+1],pc[i:i+1]).item())
        if bi+1>=N_BATCH: break
print(f"\nAggregate over {len(agg[LEVELS[0]])} validation plants")
print(f"{'RGB hidden':>11} {'visible tok':>12} {'Chamfer':>10} {'vs 0%':>8}")
base=np.mean(agg[0.0])
for lv in LEVELS:
    mu=np.mean(agg[lv])
    print(f"{lv*100:>10.0f}% {int(round(196*(1-lv))):>9}/196 {mu:>10.5f} {mu/base:>7.2f}x")

# ── qualitative plate for one plant ───────────────────────────────────────
idx=[i for i,n in enumerate(names) if PLANT in n][0]
pl=DataLoader(Subset(ds,[idx]),batch_size=1,shuffle=False,num_workers=2)
rgb,depth,pc,params,tv,nm=next(iter(pl))
rgb,depth,pc=rgb.to(DEV),depth.to(DEV),pc.to(DEV); params,tv=params.to(DEV),tv.to(DEV)
mi,si=torch.tensor([0.485,0.456,0.406]).view(3,1,1),torch.tensor([0.229,0.224,0.225]).view(3,1,1)
rgb_img=(rgb.cpu()*si+mi).clamp(0,1).numpy()[0].transpose(1,2,0)
gt=pc[0].cpu().numpy()

def sub(a,n=2200):
    return a[np.random.default_rng(0).choice(a.shape[0],n,replace=False)] if a.shape[0]>n else a
def cloud(ax,pts,c,s=1.4,a=0.75):
    ax.scatter(pts[:,0],pts[:,2],pts[:,1],s=s,c=c,alpha=a,linewidths=0,edgecolors='none')
    ax.set_box_aspect((1,1,1)); ax.view_init(elev=16,azim=-62)
    for pane in (ax.xaxis,ax.yaxis,ax.zaxis):
        pane.set_pane_color((1,1,1,0)); pane.line.set_color((0,0,0,0)); pane.set_ticks([])
    ax.grid(False); L=0.72
    ax.set_xlim(-L,L); ax.set_ylim(-L,L); ax.set_zlim(-L,L)

P=m.patch_size; G=224//P
fig=plt.figure(figsize=(10.2,2.9*len(LEVELS)))
for r,lv in enumerate(LEVELS):
    ppc,mr,keep=forward_partial_rgb(m,rgb,depth,pc,params,tv,lv,seed=0)
    ch=chamfer_distance(ppc,pc).item()
    vis=(1-mr[0].cpu().numpy()).reshape(G,G)
    shown=rgb_img.copy()
    for gy in range(G):
        for gx in range(G):
            if vis[gy,gx]==0:
                shown[gy*P:(gy+1)*P, gx*P:(gx+1)*P] = 0.16
    ax=fig.add_subplot(len(LEVELS),3,r*3+1); ax.imshow(shown); ax.axis('off')
    ax.set_title(f"{lv*100:.0f}% of RGB hidden   ({keep}/196 patches)",
                 fontsize=9,fontweight='bold',color='#0f1619')
    ax=fig.add_subplot(len(LEVELS),3,r*3+2,projection='3d')
    cloud(ax,sub(ppc[0].cpu().numpy()),GEN_C)
    ax.set_title('Generated point cloud',fontsize=9,color='#1baf7a')
    ax=fig.add_subplot(len(LEVELS),3,r*3+3,projection='3d')
    cloud(ax,sub(gt),GT_C,s=1.2,a=0.32); cloud(ax,sub(ppc[0].cpu().numpy()),GEN_C,s=1.2,a=0.55)
    ax.set_title(f'Overlay on ground truth  ·  Chamfer {ch:.5f}',fontsize=9,color='#43535a')
fig.suptitle(f'Point-cloud generation as the photograph is hidden  —  {nm[0]}',
             fontsize=11.5,fontweight='bold',y=0.997,color='#0f1619')
plt.tight_layout(rect=[0,0,1,0.977])
plt.savefig(OUT/'rgb_mask_sweep.png',dpi=145,bbox_inches='tight',facecolor='white')
print(f"\nsaved {OUT/'rgb_mask_sweep.png'}")
print("SWEEP_DONE")
