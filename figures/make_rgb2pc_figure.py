"""RGB -> point cloud generation plate for the results artifact.

Conditions the distilled model on RGB alone (depth, PC and spline fully masked)
and renders the point cloud it invents, beside ground truth and an overlay.
Point clouds use one flat colour each so the reader compares SHAPE, not a
depth-mapped rainbow.
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

DEV = 'cuda'
CKPT = './outputs/4m_distill_15k_all/best_model.pth'
OUT = Path('figures_fixed'); OUT.mkdir(exist_ok=True)
PLANTS = ['Sorghum_10001_00', 'Sorghum_10016_00', 'Sorghum_1001_00', 'Sorghum_10065_00']

GT_C, GEN_C = '#5b6870', '#1baf7a'      # neutral slate vs the artifact's PC hue

cfg = yaml.safe_load(open('configs/config_4m_distill_15k_all.yaml'))
args = DIS.config_to_namespace(cfg)
root = cfg['data']['data_root']
ds = SorghumDataset4M(f"{root}/val", img_size=args.img_size, num_points=args.num_points)
names = [p.name for p in ds.samples]
idxs = []
for want in PLANTS:
    hit = [i for i, n in enumerate(names) if want in n]
    if hit: idxs.append(hit[0])
    else:   print(f"  ! {want} not in val split, skipping")
print(f"rendering {len(idxs)} plants")

dl = DataLoader(Subset(ds, idxs), batch_size=len(idxs), shuffle=False, num_workers=4)
model = DIS.build_model(args, DEV)
DIS.load_weights_into(model, CKPT, DEV, 'distilled')
model.eval()

rgb, depth, pc, params, tv, nm = next(iter(dl))
rgb, depth, pc = rgb.to(DEV), depth.to(DEV), pc.to(DEV)
params, tv = params.to(DEV), tv.to(DEV)
with torch.no_grad():
    _, _, (_, _, ppc, _), _ = model(rgb, depth, pc, params, tv, visible={'rgb'})

ch = [chamfer_distance(ppc[i:i+1], pc[i:i+1]).item() for i in range(len(idxs))]
mean_i, std_i = torch.tensor([0.485,0.456,0.406]).view(3,1,1), torch.tensor([0.229,0.224,0.225]).view(3,1,1)
rgb_np = (rgb.cpu()*std_i+mean_i).clamp(0,1).numpy()
pc_np, ppc_np = pc.cpu().numpy(), ppc.cpu().numpy()

def sub(a, n=2200):
    return a[np.random.default_rng(0).choice(a.shape[0], n, replace=False)] if a.shape[0] > n else a

def cloud(ax, pts, c, s=1.4, a=0.75):
    ax.scatter(pts[:,0], pts[:,2], pts[:,1], s=s, c=c, alpha=a, linewidths=0, edgecolors='none')
    ax.set_box_aspect((1,1,1)); ax.view_init(elev=16, azim=-62)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((1,1,1,0)); pane.line.set_color((0,0,0,0))
        pane.set_ticks([])
    ax.grid(False)
    lim = 0.72
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)

n = len(idxs)
fig = plt.figure(figsize=(13.2, 3.35*n))
for i in range(n):
    ax = fig.add_subplot(n, 4, i*4+1)
    ax.imshow(rgb_np[i].transpose(1,2,0)); ax.axis('off')
    ax.set_title('RGB  ← the only input', fontsize=9, fontweight='bold', color='#0f1619')
    ax.text(0.5,-0.07, nm[i], transform=ax.transAxes, ha='center', va='top',
            fontsize=8, color='#71838b', family='monospace')

    g, p = sub(pc_np[i]), sub(ppc_np[i])
    ax = fig.add_subplot(n, 4, i*4+2, projection='3d'); cloud(ax, g, GT_C)
    ax.set_title('Ground-truth point cloud', fontsize=9, color='#43535a')
    ax = fig.add_subplot(n, 4, i*4+3, projection='3d'); cloud(ax, p, GEN_C)
    ax.set_title('Generated from RGB', fontsize=9, fontweight='bold', color='#1baf7a')
    ax = fig.add_subplot(n, 4, i*4+4, projection='3d')
    cloud(ax, g, GT_C, s=1.2, a=0.35); cloud(ax, p, GEN_C, s=1.2, a=0.55)
    ax.set_title(f'Overlay  ·  Chamfer {ch[i]:.5f}', fontsize=9, color='#43535a')

fig.suptitle('Point clouds generated from a single RGB image  —  distilled 4M model, held-out validation plants',
             fontsize=11.5, fontweight='bold', y=0.996, color='#0f1619')
plt.tight_layout(rect=[0,0,1,0.975])
out = OUT/'rgb2pc_plate.png'
plt.savefig(out, dpi=145, bbox_inches='tight', facecolor='white')
print(f"saved {out}")
print("per-plant chamfer:", " ".join(f"{c:.5f}" for c in ch))
print(f"mean over these {n}: {np.mean(ch):.5f}")
print("RGB2PC_DONE")
