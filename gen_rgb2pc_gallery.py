"""Generate RGB -> point cloud examples across DISTINCT validation plants.

Why this exists rather than generate_crossmodal.py: that script builds its loader
with shuffle=False and takes the first `num_samples` folders, and the split is laid
out as Sorghum_<plant>_<view> sorted by name -- so "6 samples" is six VIEWS of one
plant, not six plants. The run's own training visualisations have the same problem
(all six are Sorghum_10001_00..05). This strides by `--stride` so every row is a
different plant.

Runs on CPU by default: the GPUs are busy, and a handful of forward passes is cheap.
"""
import argparse, json, pathlib, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from embodied_mae_4m import embodied_mae_4m_base, chamfer_distance
from sorghum_dataset_4m import SorghumDataset4M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='outputs/4m_distill_15k_rgb2pc_full/best_model.pth')
    ap.add_argument('--data_root', default='/work/mech-ai-scratch/alloy/shorgum_data/new_data_50K/Sorghum_15K')
    ap.add_argument('--split', default='val')
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--stride', type=int, default=137,   # coprime with 10 views/plant
                    help='folder stride; must not be a multiple of 10 or every row '
                         'is the same view index and rows correlate')
    ap.add_argument('--num_points', type=int, default=8196)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', default='vis_gallery')
    args = ap.parse_args()

    dev = torch.device(args.device)
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)

    ds = SorghumDataset4M(args.data_root, split=args.split, num_points=args.num_points)
    idx = [(i * args.stride) % len(ds) for i in range(args.n)]
    print(f"dataset {len(ds)} folders; picking {len(idx)} with stride {args.stride}")

    model = embodied_mae_4m_base(target_points=args.num_points).to(dev).eval()
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    sd = ck.get('model_state_dict', ck)
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"checkpoint epoch {ck.get('epoch','?')}  missing={len(missing)} unexpected={len(unexpected)}")
    assert not missing and not unexpected, "checkpoint does not match the model"

    rows = []
    for k, i in enumerate(idx):
        rgb, depth, pc, params, tvalid, name = ds[i]
        b = lambda t: t.unsqueeze(0).to(dev)
        with torch.no_grad():
            _, _, (_, _, pred_pc, _), _ = model(
                b(rgb), b(depth), b(pc), b(params), b(tvalid), visible=['rgb'])
            cd = chamfer_distance(pred_pc, b(pc)).item()
            # random-plant control: same prediction against a DIFFERENT plant's cloud
            j = idx[(k + 1) % len(idx)]
            cd_rand = chamfer_distance(pred_pc, ds[j][2].unsqueeze(0).to(dev)).item()
        rows.append(dict(name=name, chamfer=cd, chamfer_random=cd_rand,
                         rgb=rgb, gt=pc.numpy(), gen=pred_pc[0].cpu().numpy()))
        print(f"  {k+1}/{len(idx)}  {name:<22} chamfer {cd:.5f}   random-plant {cd_rand:.5f}")

    # ---- figure: RGB | GT cloud | generated cloud, one row per plant ----
    MEAN = np.array([0.485, 0.456, 0.406]); STD = np.array([0.229, 0.224, 0.225])
    n = len(rows)
    fig = plt.figure(figsize=(8.4, 1.62 * n + 0.30), dpi=125)
    gs = fig.add_gridspec(n, 3, wspace=0.0, hspace=0.06,
                          left=0.005, right=0.995, top=1 - 0.30 / (1.62 * n + 0.30), bottom=0.004)
    for r, row in enumerate(rows):
        im = (row['rgb'].permute(1, 2, 0).numpy() * STD + MEAN).clip(0, 1)
        ax = fig.add_subplot(gs[r, 0]); ax.imshow(im); ax.axis('off')
        ax.text(0.02, 0.965, row['name'], transform=ax.transAxes, fontsize=8,
                color='white', family='monospace', va='top',
                bbox=dict(fc='#0f1619', ec='none', alpha=.55, pad=1.8))
        for c, (key, col) in enumerate((('gt', '#9aa7ad'), ('gen', '#1baf7a')), start=1):
            pts = row[key]
            ax = fig.add_subplot(gs[r, c], projection='3d')
            sel = np.random.default_rng(0).choice(len(pts), min(2500, len(pts)), replace=False)
            ax.scatter(pts[sel, 0], pts[sel, 2], pts[sel, 1], s=1.0, c=col, linewidths=0)
            ax.set_axis_off(); ax.view_init(elev=14, azim=-62)
            for lim in (ax.set_xlim, ax.set_ylim, ax.set_zlim): lim(-0.62, 0.62)
            ax.set_box_aspect((1, 1, 1), zoom=1.36)
            if c == 2:
                ax.text2D(0.97, 0.05, f"Chamfer {row['chamfer']:.5f}", transform=ax.transAxes,
                          ha='right', fontsize=8, color='#1baf7a', family='monospace')
            if r == 0:
                ax.set_title('ground truth' if key == 'gt' else 'generated from RGB alone',
                             fontsize=9.5, color='#43535a', pad=-2)
    fig.savefig(out / 'gallery.png', bbox_inches='tight', facecolor='white')

    cds = np.array([r['chamfer'] for r in rows])
    rnd = np.array([r['chamfer_random'] for r in rows])
    summary = dict(checkpoint=args.checkpoint, epoch=int(ck.get('epoch', -1)),
                   n=n, stride=args.stride, split=args.split,
                   plants=[r['name'] for r in rows],
                   chamfer=[float(x) for x in cds],
                   chamfer_mean=float(cds.mean()), chamfer_std=float(cds.std()),
                   chamfer_min=float(cds.min()), chamfer_max=float(cds.max()),
                   random_plant_mean=float(rnd.mean()))
    (out / 'gallery.json').write_text(json.dumps(summary, indent=2))
    print(f"\nmean {cds.mean():.5f}  sd {cds.std():.5f}  "
          f"range {cds.min():.5f}-{cds.max():.5f}   random-plant control {rnd.mean():.5f}")
    print(f"wrote {out}/gallery.png and gallery.json")


if __name__ == '__main__':
    main()
