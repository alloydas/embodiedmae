"""Side-by-side RGB -> point cloud: UNMASKED control (smr 0.0) vs SOURCE-MASKED (smr 0.5).

The number this figure is supposed to illustrate is small: on the 2250-plant test
split the source-masked student's symmetric Chamfer is 0.000791 against the
unmasked control's 0.000840, a 5.9% mean improvement.  Two things make an honest
picture of that hard, and both are handled here rather than hidden:

1. ROW SELECTION.  Picking the plants where smr50 wins biggest would make a
   5.9% mean look like a categorical difference.  Rows are instead chosen to
   span the per-plant DIFFERENCE distribution -- the two biggest wins, two rows
   at the median, and the two plants where the masked model is WORSE -- and each
   row is labelled with which it is.

2. THE PAIRING.  reports/quant_*.json were produced by separate runs of
   eval_rgb2pc_quant.py, and SorghumDataset draws a fresh random 8196-point
   subsample of the .ply on every __getitem__.  So the two reports scored the two
   checkpoints against DIFFERENT draws of the same plant, and a per-plant
   difference taken across reports carries a resampling term of the same order
   (~7e-5, measured below) as the effect being measured (population median diff
   -3.4e-5).  The cross-report difference still correlates r=0.91 with the truly
   paired one, so it is not pure noise -- but it is inflated at the tails, which
   is exactly where the extreme rows are chosen from, so those rows would
   regress to the mean and overstate the gain.

   So the difference distribution is re-measured properly: a random pool of
   plants is drawn from the report's own plant list, each plant is loaded ONCE
   under a fixed seed, and both checkpoints are run on that one cloud.  Rows come
   from the percentiles of that truly paired difference, and the script prints
   the pool's paired statistics and its correlation with the cross-report
   difference so the size of the noise problem is on the record.

Rendering conventions (view, limits, GT colour, generated colour) are copied from
gen_rgb2pc_gallery_pct.py; the control gets its own colour so the two generated
columns can be told apart.

  python gen_rgb2pc_arm_compare.py --out vis_gallery/arm_compare_smr50_vs_smr00.png
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

C_GT   = '#9aa7ad'
C_SMR  = '#1baf7a'    # source-masked student, the model under test
C_CTL  = '#3f7fb0'    # unmasked control
C_TXT  = '#43535a'


def load_model(path, num_points, dev):
    model = embodied_mae_4m_base(target_points=num_points).to(dev).eval()
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = ck.get('model_state_dict', ck)
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, f'checkpoint does not match the model: {path}'
    return model, int(ck.get('epoch', -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report_ctl',  default='reports/quant_smr00_full_test.json')
    ap.add_argument('--report_smr',  default='reports/quant_smr50_seed1_test.json')
    ap.add_argument('--ckpt_ctl', default='outputs/4m_distill_15k_rgb2pc_full/best_model.pth')
    ap.add_argument('--ckpt_smr', default='outputs/4m_distill_15k_rgb2pc_smr50/best_model.pth')
    ap.add_argument('--data_root', default='/work/mech-ai-scratch/alloy/shorgum_data/new_data_50K/Sorghum_15K')
    ap.add_argument('--pool', type=int, default=64, help='plants re-scored under a shared cloud')
    ap.add_argument('--pool_seed', type=int, default=7)
    ap.add_argument('--num_points', type=int, default=8196)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    rep_c = json.loads(pathlib.Path(args.report_ctl).read_text())
    rep_s = json.loads(pathlib.Path(args.report_smr).read_text())
    assert rep_c['plants'] == rep_s['plants'], 'reports are not over the same plant list'
    assert rep_c['split'] == rep_s['split']
    names_all = rep_c['plants']
    cd_c_rep = np.asarray(rep_c['per_sample']['model'])
    cd_s_rep = np.asarray(rep_s['per_sample']['model'])
    d_rep_all = cd_s_rep - cd_c_rep
    print(f"reports: n={len(names_all)} split={rep_c['split']}  "
          f"control {cd_c_rep.mean():.6f}  smr50 {cd_s_rep.mean():.6f}  "
          f"({100 * (cd_s_rep.mean() / cd_c_rep.mean() - 1):+.1f}%)  "
          f"cross-report per-plant diff median {np.median(d_rep_all):+.6f}, "
          f"smr50 better on {100 * (d_rep_all < 0).mean():.1f}%")

    dev = torch.device(args.device)
    ds = SorghumDataset4M(args.data_root, split=rep_c['split'], num_points=args.num_points)
    by_name = {f.name: i for i, f in enumerate(ds.samples)}

    m_ctl, ep_c = load_model(args.ckpt_ctl, args.num_points, dev)
    m_smr, ep_s = load_model(args.ckpt_smr, args.num_points, dev)
    print(f"control  {args.ckpt_ctl} epoch {ep_c}")
    print(f"smr50    {args.ckpt_smr} epoch {ep_s}")

    # --- paired re-scoring pool: one cloud per plant, both checkpoints on it ---
    rng = np.random.default_rng(args.pool_seed)
    pool = sorted(rng.choice(len(names_all), size=min(args.pool, len(names_all)), replace=False).tolist())
    recs = []
    for k, j in enumerate(pool):
        name = names_all[j]
        np.random.seed(3000 + k)                       # freeze the .ply subsample for this row
        rgb, depth, pc, par, tv, nm = ds[by_name[name]]
        b = lambda t: t.unsqueeze(0).to(dev)
        outs = {}
        with torch.no_grad():
            for key, mdl in (('ctl', m_ctl), ('smr', m_smr)):
                _, _, (_, _, pred, _), _ = mdl(b(rgb), b(depth), b(pc), b(par), b(tv), visible=['rgb'])
                f, r = chamfer_both(pred.cpu(), pc.unsqueeze(0))
                outs[key] = (float(f + r), pred[0].cpu().numpy())
        recs.append(dict(name=nm, rgb=rgb, gt=pc.numpy(),
                         cd_ctl=outs['ctl'][0], gen_ctl=outs['ctl'][1],
                         cd_smr=outs['smr'][0], gen_smr=outs['smr'][1],
                         diff=outs['smr'][0] - outs['ctl'][0], d_rep=float(d_rep_all[j])))
        print(f"  [{k + 1:3d}/{len(pool)}] {nm:<20} ctl {recs[-1]['cd_ctl']:.5f}  "
              f"smr50 {recs[-1]['cd_smr']:.5f}  diff {recs[-1]['diff']:+.5f} "
              f"(cross-report {recs[-1]['d_rep']:+.5f})", flush=True)

    d = np.array([r['diff'] for r in recs])
    dr = np.array([r['d_rep'] for r in recs])
    cc = np.array([r['cd_ctl'] for r in recs]); cs = np.array([r['cd_smr'] for r in recs])
    print(f"\npaired pool n={len(d)}: control {cc.mean():.6f}  smr50 {cs.mean():.6f}  "
          f"({100 * (cs.mean() / cc.mean() - 1):+.1f}%)")
    print(f"  diff mean {d.mean():+.6f}  median {np.median(d):+.6f}  "
          f"smr50 better on {100 * (d < 0).mean():.1f}% of the pool")
    print(f"  paired-vs-cross-report diff correlation r={np.corrcoef(d, dr)[0, 1]:.3f} "
          f"(cross-report sd {dr.std():.6f} vs paired sd {d.std():.6f})")

    # --- rows: span the paired difference distribution, do not cherry-pick ---
    order = np.argsort(d)                                        # smr50 best -> worst
    n = len(order)
    mid = n // 2
    roles = [(order[0], 'biggest gain'), (order[1], 'large gain'),
             (order[mid - 1], 'median'), (order[mid], 'median'),
             (order[-2], None), (order[-1], None)]
    rows = []
    for i, role in roles:
        r = recs[i]
        if role is None:                       # tail of the distribution: name it by its sign
            role = 'smr50 worse' if r['diff'] > 0 else 'smallest gain'
        rows.append(dict(r, tag=role))

    MEAN = np.array([0.485, 0.456, 0.406]); STD = np.array([0.229, 0.224, 0.225])
    nr = len(rows)
    fig = plt.figure(figsize=(11.4, 1.62 * nr + 0.72), dpi=125)
    top_pad, bot_pad = 0.36, 0.60
    H = 1.62 * nr + top_pad + bot_pad
    gs = fig.add_gridspec(nr, 4, wspace=0.0, hspace=0.06, left=0.085, right=0.995,
                          top=1 - top_pad / H, bottom=bot_pad / H)
    titles = ['input: RGB only', 'ground truth',
              'unmasked control  (smr 0.0)', 'source-masked  (smr 0.5)']
    for r_, row in enumerate(rows):
        im = (row['rgb'].permute(1, 2, 0).numpy() * STD + MEAN).clip(0, 1)
        ax = fig.add_subplot(gs[r_, 0]); ax.imshow(im); ax.axis('off')
        ax.text(0.02, 0.965, row['name'], transform=ax.transAxes, fontsize=8, color='white',
                family='monospace', va='top', bbox=dict(fc='#0f1619', ec='none', alpha=.55, pad=1.8))
        ax.text(-0.075, 0.5, f"{row['tag']}\n{row['diff']:+.5f}", transform=ax.transAxes,
                fontsize=8.5, rotation=90, ha='center', va='center', linespacing=1.45,
                color=(C_SMR if row['diff'] < 0 else '#b4553a'), family='monospace')
        if r_ == 0:
            ax.set_title(titles[0], fontsize=9.5, color=C_TXT, pad=4)
        panels = ((row['gt'], C_GT, None), (row['gen_ctl'], C_CTL, row['cd_ctl']),
                  (row['gen_smr'], C_SMR, row['cd_smr']))
        for c, (pts, col, cd) in enumerate(panels, start=1):
            ax = fig.add_subplot(gs[r_, c], projection='3d')
            sel = np.random.default_rng(0).choice(len(pts), min(2500, len(pts)), replace=False)
            ax.scatter(pts[sel, 0], pts[sel, 2], pts[sel, 1], s=1.0, c=col, linewidths=0)
            ax.set_axis_off(); ax.view_init(elev=14, azim=-62)
            for lim in (ax.set_xlim, ax.set_ylim, ax.set_zlim): lim(-0.62, 0.62)
            ax.set_box_aspect((1, 1, 1), zoom=1.36)
            if cd is not None:
                ax.text2D(0.97, 0.03, f"Chamfer {cd:.5f}", transform=ax.transAxes,
                          ha='right', fontsize=8, color=col, family='monospace',
                          bbox=dict(fc='white', ec='none', alpha=.72, pad=1.4))
            if r_ == 0:
                ax.set_title(titles[c], fontsize=9.5, color=C_TXT, pad=-2)

    fig.text(0.5, 0.008,
             "\u0394 = source-masked \u2212 control symmetric Chamfer; negative (green) = source masking better.  "
             "Both cloud columns are generated from the RGB frame alone, on the SAME loaded point cloud per row.\n"
             f"Test split, {len(names_all)} plants: control {cd_c_rep.mean():.6f} vs source-masked "
             f"{cd_s_rep.mean():.6f} in the MEAN ({100 * (cd_s_rep.mean() / cd_c_rep.mean() - 1):+.1f}%).  "
             f"On a {len(d)}-plant paired re-scoring the median plant is a wash (\u0394 {np.median(d):+.5f}) and "
             f"source masking wins on only {100 * (d < 0).mean():.0f}%: the mean gain lives in the hard tail.\n"
             "Rows span that difference distribution, not the wins.",
             ha='center', va='bottom', fontsize=7.6, color=C_TXT, family='monospace',
             linespacing=1.5)

    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches='tight', facecolor='white')
    print(f"\nwrote {out}  rows: " + ', '.join(f"{r['name']} ({r['tag']} {r['diff']:+.5f})" for r in rows))


if __name__ == '__main__':
    main()
