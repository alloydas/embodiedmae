"""Capability shot: ONE test plant, several camera views, cloud generated from RGB alone.

figures/gen_rgb2pc_gallery_pct.py answers "how well does it score" by spanning the error
distribution across many plants.  This answers a different question: hold the plant
fixed and vary the camera.  Is a single plant reconstructed consistently from every
viewpoint, or do some viewpoints collapse?

The 10 folders Sorghum_<id>_00..09 are not 10 random cameras -- camera_pose.json
shows view index is a fixed ELEVATION ladder (~+71 deg looking down at _00, through
side-on near _06, to ~-52 deg looking up at _09) with azimuth randomised.  So the
rows below are an elevation sweep, and the per-row Chamfer is a direct readout of how
much viewing elevation costs the RGB->PC model.

Chamfer per row is averaged over --draws independent point-cloud resamples, because
SorghumDataset.load_pointcloud draws a fresh random 8196-point subsample on every
__getitem__ (sorghum_dataset.py:231); one draw carries 0.0001-0.00025 of noise, which
is 15-35% of the model's own error.  The rendered cloud is the first draw.

Defaults reproduce vis_gallery/smr50_multiview.png:
    python figures/gen_rgb2pc_multiview.py --device cpu
"""
# Repo root on sys.path: this script lives one level down but imports the
# top-level modules (embodied_mae*, sorghum_dataset*, train_*).
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import argparse, json, pathlib
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from embodied_mae_4m import embodied_mae_4m_base
from sorghum_dataset_4m import SorghumDataset4M
from eval_rgb2pc_quant import chamfer_both

# Sorghum_174 is the representative pick, not the flattering one: its report draw
# (Sorghum_174_07, 0.000702) sits at the 50.1st percentile of the 2250-plant test
# distribution and its 10-view mean (0.000688) at the 48.6th.  See --report.
DEFAULT_PLANT = 'Sorghum_174'
DEFAULT_VIEWS = '0,2,4,6,8,9'          # even stride through the ladder + both extremes
IMNET_MEAN = np.array([0.485, 0.456, 0.406])
IMNET_STD = np.array([0.229, 0.224, 0.225])
GT_COL, GEN_COL = '#9aa7ad', '#1baf7a'
INK = '#43535a'


def camera_elevation(folder):
    """Elevation of the VIEW DIRECTION in degrees; negative = camera looking down.

    Taken from the cameraToWorld forward vector rather than the camera position:
    forward = -M[:3, 2], and its y component is exactly -0.9 + 0.2 * view_index
    across all 2250 test folders (sd 0.0 within each view group), so this is the
    same quantity figures/plot_view_elevation_effect.py puts on its x axis. The camera
    position elevation is a different number, varies per plant, and has the
    opposite sign.
    """
    f = folder / 'camera_pose.json'
    if not f.is_file():
        return None
    M = np.array(json.loads(f.read_text())['cameraToWorld']).reshape(4, 4)
    return float(np.degrees(np.arcsin(np.clip(-M[:3, 2][1], -1.0, 1.0))))


def load_model(checkpoint, num_points, dev):
    model = embodied_mae_4m_base(target_points=num_points).to(dev).eval()
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    sd = ck.get('model_state_dict', ck)
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, 'checkpoint does not match the model'
    return model, ck.get('epoch', '?')


def run_view(model, ds, idx, draws, seed0, dev):
    """Chamfer over `draws` resamples; returns (mean, std, rgb, gt, gen) of draw 0."""
    vals, keep = [], None
    for d in range(draws):
        np.random.seed(seed0 + d)
        rgb, depth, pc, par, tv, _ = ds[idx]
        b = lambda t: t.unsqueeze(0).to(dev)
        with torch.no_grad():
            _, _, (_, _, pred, _), _ = model(b(rgb), b(depth), b(pc), b(par), b(tv),
                                             visible=['rgb'])
        f, r = chamfer_both(pred.cpu(), pc.unsqueeze(0))
        vals.append(float(f + r))
        if d == 0:
            keep = (rgb, pc.numpy(), pred[0].cpu().numpy())
    v = np.asarray(vals)
    return v.mean(), v.std(), *keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='outputs/4m_distill_15k_rgb2pc_smr50/best_model.pth')
    ap.add_argument('--report', default='reports/quant_smr50_seed1_test.json',
                    help='only used to place this plant in the population distribution')
    ap.add_argument('--data_root',
                    default='/work/mech-ai-scratch/alloy/shorgum_data/new_data_50K/Sorghum_15K')
    ap.add_argument('--split', default='test')
    ap.add_argument('--plant', default=DEFAULT_PLANT)
    ap.add_argument('--views', default=DEFAULT_VIEWS, help='view indices to draw as rows')
    ap.add_argument('--draws', type=int, default=3, help='PC resamples averaged per view')
    ap.add_argument('--num_points', type=int, default=8196)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', default='vis_gallery/smr50_multiview.png')
    args = ap.parse_args()

    dev = torch.device(args.device)
    ds = SorghumDataset4M(args.data_root, split=args.split, num_points=args.num_points)
    by_name = {f.name: i for i, f in enumerate(ds.samples)}
    model, epoch = load_model(args.checkpoint, args.num_points, dev)

    pop = None
    if args.report and pathlib.Path(args.report).is_file():
        pop = np.asarray(json.loads(pathlib.Path(args.report).read_text())['per_sample']['model'])
    pct = (lambda x: float((pop < x).mean() * 100)) if pop is not None else (lambda x: float('nan'))
    print(f"checkpoint epoch {epoch}   plant {args.plant}   split {args.split}   "
          f"draws {args.draws}   source visible=['rgb']")

    # Score every one of the 10 views so the footer can state the true spread, even
    # though only --views get drawn.  Drawing a subset and quoting only the subset's
    # range would understate it.
    root = pathlib.Path(args.data_root) / args.split
    allv = {}
    for v in range(10):
        nm = f"{args.plant}_{v:02d}"
        if nm not in by_name:
            continue
        m, s, rgb, gt, gen = run_view(model, ds, by_name[nm], args.draws, 7000 + 13 * v, dev)
        allv[v] = dict(name=nm, mean=m, std=s, elev=camera_elevation(root / nm),
                       rgb=rgb, gt=gt, gen=gen)
        e = allv[v]['elev']
        print(f"  {nm:<20} elev {e:6.1f}  chamfer {m:.5f} +/- {s:.5f}  p{pct(m):.0f}")

    ms = np.array([allv[v]['mean'] for v in sorted(allv)])
    lo, hi = int(np.argmin(ms)), int(np.argmax(ms))
    print(f"  10-view mean {ms.mean():.5f} (p{pct(ms.mean()):.0f})  "
          f"min view {sorted(allv)[lo]:02d} {ms.min():.5f}  max view {sorted(allv)[hi]:02d} "
          f"{ms.max():.5f}  ratio {ms.max()/ms.min():.2f}x  cv {ms.std()/ms.mean():.3f}")

    rows = [allv[int(v)] for v in args.views.split(',') if int(v) in allv]
    n = len(rows)
    fig_h = 1.62 * n + 0.66 + 0.42
    fig = plt.figure(figsize=(8.9, fig_h), dpi=125)
    gs = fig.add_gridspec(n, 3, wspace=0.0, hspace=0.06, left=0.052, right=0.995,
                          top=1 - 0.66 / fig_h, bottom=0.42 / fig_h)
    for r_, row in enumerate(rows):
        im = (row['rgb'].permute(1, 2, 0).numpy() * IMNET_STD + IMNET_MEAN).clip(0, 1)
        ax = fig.add_subplot(gs[r_, 0]); ax.imshow(im); ax.axis('off')
        ax.text(0.02, 0.965, row['name'], transform=ax.transAxes, fontsize=8, color='white',
                family='monospace', va='top',
                bbox=dict(fc='#0f1619', ec='none', alpha=.55, pad=1.8))
        lab = 'view %s' % row['name'][-2:]
        if row['elev'] is not None:
            lab += ' · elev %+.0f°' % row['elev']
        ax.text(-0.052, 0.5, lab, transform=ax.transAxes, fontsize=8.5, rotation=90,
                ha='center', va='center', color=INK, family='monospace')
        for c, (key, col) in enumerate((('gt', GT_COL), ('gen', GEN_COL)), start=1):
            pts = row[key]
            ax = fig.add_subplot(gs[r_, c], projection='3d')
            sel = np.random.default_rng(0).choice(len(pts), min(2500, len(pts)), replace=False)
            ax.scatter(pts[sel, 0], pts[sel, 2], pts[sel, 1], s=1.0, c=col, linewidths=0)
            ax.set_axis_off(); ax.view_init(elev=14, azim=-62)
            for lim in (ax.set_xlim, ax.set_ylim, ax.set_zlim): lim(-0.62, 0.62)
            ax.set_box_aspect((1, 1, 1), zoom=1.36)
            if c == 2:
                ax.text2D(0.97, 0.05, f"Chamfer {row['mean']:.5f} ±{row['std']:.5f}",
                          transform=ax.transAxes, ha='right', fontsize=8,
                          color=GEN_COL, family='monospace')
            if r_ == 0:
                ax.set_title('ground truth' if key == 'gt' else 'generated from RGB alone',
                             fontsize=9.5, color=INK, pad=-2)
    fig.text(0.052, 1 - 0.16 / fig_h,
             f"{args.plant} — one plant, one RGB image per row, point cloud generated from that "
             f"image alone (smr50, epoch {epoch})",
             fontsize=9.5, color='#20303a', va='top')

    strip = '  '.join(f"{v:02d}:{allv[v]['mean']*1e3:.2f}" for v in sorted(allv))
    fig.text(0.052, 0.245 / fig_h,
             f"all 10 views, Chamfer ×10⁻³ by view index (elevation −64° → +64°):  {strip}",
             fontsize=7.6, color=INK, family='monospace', va='bottom')
    fig.text(0.052, 0.075 / fig_h,
             f"10-view mean {ms.mean():.5f} = {pct(ms.mean()):.0f}th pct of the {0 if pop is None else len(pop)}-plant "
             f"test distribution; hardest view {ms.max()/ms.min():.1f}× the easiest — "
             f"steeply off-horizontal cameras cost the most, looking up as much as looking down.",
             fontsize=7.6, color=INK, va='bottom')

    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches='tight', facecolor='white')
    print(f"wrote {out}   ({n} rows: views {[r['name'][-2:] for r in rows]})")


if __name__ == '__main__':
    main()
