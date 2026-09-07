"""Render an RGB -> point-cloud gallery that spans the ERROR DISTRIBUTION.

gen_rgb2pc_gallery.py strides through the split from index 0, which on this
lexicographically-sorted layout means its eight rows all come from the first
4.3% of the folders and land at a mean Chamfer of 0.00072 against a population
mean of 0.00089 -- six of its eight rows sit below the population median and its
worst row is only the 72nd percentile.  Nothing in the hard quartile ever
appears, so the figure reads as better than the model is.

This picks rows by PERCENTILE of an eval_rgb2pc_quant.py report instead, so the
best case, the median and the genuine failures are all on the page and labelled
as such.  Same three columns, same view, same checkpoint loading.
"""
import argparse, json, pathlib
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from embodied_mae_4m import embodied_mae_4m_base
from sorghum_dataset_4m import SorghumDataset4M
from eval_rgb2pc_quant import chamfer_both

DEFAULT_PCT = '0,10,25,50,75,90,97,100'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', required=True, help='JSON written by eval_rgb2pc_quant.py')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--data_root', default='/work/mech-ai-scratch/alloy/shorgum_data/new_data_50K/Sorghum_15K')
    ap.add_argument('--percentiles', default=DEFAULT_PCT)
    ap.add_argument('--num_points', type=int, default=8196)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    rep = json.loads(pathlib.Path(args.report).read_text())
    vals = np.asarray(rep['per_sample']['model'])
    names = rep['plants']
    order = np.argsort(vals)                      # easiest -> hardest
    pcts = [float(p) for p in args.percentiles.split(',')]
    picks, seen = [], set()
    for p in pcts:
        r = int(round(p / 100 * (len(order) - 1)))
        while r in seen and r < len(order) - 1:   # keep rows distinct
            r += 1
        seen.add(r)
        picks.append((p, names[order[r]], float(vals[order[r]])))

    dev = torch.device(args.device)
    ds = SorghumDataset4M(args.data_root, split=rep['split'], num_points=args.num_points)
    by_name = {f.name: i for i, f in enumerate(ds.samples)}

    model = embodied_mae_4m_base(target_points=args.num_points).to(dev).eval()
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    sd = ck.get('model_state_dict', ck)
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, 'checkpoint does not match the model'
    print(f"checkpoint epoch {ck.get('epoch','?')}   report {rep['label']} n={len(vals)}")

    # SorghumDataset.load_pointcloud draws a fresh random 8196-point subsample from
    # the raw .ply on every __getitem__ (sorghum_dataset.py:231) and only then centres
    # and unit-scales it, so the target -- frame and all -- is different every load.
    # Reloading one plant twice gives two clouds ~0.0001-0.00025 apart in Chamfer,
    # which is 11-27% of the model's own error.  Seed per row so this figure is
    # reproducible, and print the report's draw alongside this one rather than
    # pretending they agree.
    rows = []
    for k, (p, name, cd_ref) in enumerate(picks):
        np.random.seed(1000 + k)
        rgb, depth, pc, par, tv, nm = ds[by_name[name]]
        b = lambda t: t.unsqueeze(0).to(dev)
        with torch.no_grad():
            _, _, (_, _, pred, _), _ = model(b(rgb), b(depth), b(pc), b(par), b(tv), visible=['rgb'])
        f, r = chamfer_both(pred.cpu(), pc.unsqueeze(0))
        cd = float(f + r)
        rows.append(dict(pct=p, name=nm, chamfer=cd, chamfer_report=cd_ref, rgb=rgb,
                         gt=pc.numpy(), gen=pred[0].cpu().numpy()))
        print(f"  p{p:<5.0f} {nm:<22} chamfer {cd:.5f}  (report draw {cd_ref:.5f})")

    MEAN = np.array([0.485, 0.456, 0.406]); STD = np.array([0.229, 0.224, 0.225])
    n = len(rows)
    fig = plt.figure(figsize=(8.9, 1.62 * n + 0.34), dpi=125)
    gs = fig.add_gridspec(n, 3, wspace=0.0, hspace=0.06, left=0.038, right=0.995,
                          top=1 - 0.34 / (1.62 * n + 0.34), bottom=0.004)
    for r_, row in enumerate(rows):
        im = (row['rgb'].permute(1, 2, 0).numpy() * STD + MEAN).clip(0, 1)
        ax = fig.add_subplot(gs[r_, 0]); ax.imshow(im); ax.axis('off')
        ax.text(0.02, 0.965, row['name'], transform=ax.transAxes, fontsize=8, color='white',
                family='monospace', va='top', bbox=dict(fc='#0f1619', ec='none', alpha=.55, pad=1.8))
        lab = {0: 'best', 100: 'worst'}.get(row['pct'], f"p{row['pct']:.0f}")
        ax.text(-0.055, 0.5, lab, transform=ax.transAxes, fontsize=9, rotation=90,
                ha='center', va='center', color='#43535a', family='monospace')
        for c, (key, col) in enumerate((('gt', '#9aa7ad'), ('gen', '#1baf7a')), start=1):
            pts = row[key]
            ax = fig.add_subplot(gs[r_, c], projection='3d')
            sel = np.random.default_rng(0).choice(len(pts), min(2500, len(pts)), replace=False)
            ax.scatter(pts[sel, 0], pts[sel, 2], pts[sel, 1], s=1.0, c=col, linewidths=0)
            ax.set_axis_off(); ax.view_init(elev=14, azim=-62)
            for lim in (ax.set_xlim, ax.set_ylim, ax.set_zlim): lim(-0.62, 0.62)
            ax.set_box_aspect((1, 1, 1), zoom=1.36)
            if c == 2:
                ax.text2D(0.97, 0.05, f"Chamfer {row['chamfer']:.5f}", transform=ax.transAxes,
                          ha='right', fontsize=8, color='#1baf7a', family='monospace')
            if r_ == 0:
                ax.set_title('ground truth' if key == 'gt' else 'generated from RGB alone',
                             fontsize=9.5, color='#43535a', pad=-2)
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches='tight', facecolor='white')
    print(f"wrote {out}   (rows span {rows[0]['chamfer']:.5f} to {rows[-1]['chamfer']:.5f})")


if __name__ == '__main__':
    main()
