#!/usr/bin/env python3
"""Colour the ground-truth point cloud by whether the model predicted it.

For every GT point we take the distance to its NEAREST predicted point. If that
distance exceeds --threshold, nothing the model produced lands near that point:
the model failed to cover it. Those points are drawn RED, the covered ones grey.

The direction matters and is the whole reason this is not just "plot the error".
Chamfer has two halves and they mean different things:

    GT -> pred   (this script, default)  what the model MISSED       -> red
    pred -> GT   (--direction spurious)  what the model INVENTED     -> red

Missed geometry is the failure you can see in a render: a whole leaf blade that
the reconstruction simply does not have. Invented geometry is the opposite
failure, points floating where the plant is not. Averaged into one number they
cancel into "the chamfer is 0.0012" and neither is visible.

The default threshold is 0.01, which is not arbitrary -- it is `qal_threshold`
from the QAL runs, the Euclidean distance at the sigmoid midpoint, i.e. exactly
the distance above which the training loss already started penalising a point.
So red here means "the loss was actively unhappy about this point", in the
model's own units. Clouds are normalised to the unit sphere, so 0.01 is 1% of
the plant's max radius.

Usage
-----
    # one checkpoint
    python vis_pc_unpredicted.py --checkpoint outputs/4m_distill_15k_all/best_model.pth

    # before vs after distillation, side by side, same plants
    python vis_pc_unpredicted.py \
        --checkpoint outputs/4m_pretrain_15k_v2_depthfix_qal/teacher_final.pth \
        --checkpoint outputs/4m_distill_15k_all/best_model.pth \
        --label "before distillation" --label "after distillation" \
        --source pc --num_samples 4 --out vis_unpredicted

`--source` picks the cross-modal regime: the named modality is the ONLY one the
encoder sees. `--source none` instead runs the ordinary masked-autoencoder
forward pass, which is what you want for the E2 arms.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from sorghum_dataset_4m import SorghumDataset4M
import train_sorghum_4m_distill as DIS

MISSED = '#d4342a'   # red   -- GT point with no prediction near it
COVERED = '#9aa7ad'  # grey  -- GT point the model reproduced


def per_point_nn(pred, gt, direction):
    """Nearest-neighbour Euclidean distance, one value per point of interest.

    Returns (points_to_plot, distances). For 'missed' we plot the GT cloud and
    score each GT point by how far the nearest PREDICTION is; for 'spurious' we
    plot the predicted cloud and score each predicted point by how far the
    nearest GT point is.
    """
    d = torch.cdist(pred.unsqueeze(0), gt.unsqueeze(0), p=2)[0]   # (N_pred, M_gt)
    if direction == 'missed':
        return gt, d.min(dim=0).values                            # (M_gt,)
    return pred, d.min(dim=1).values                              # (N_pred,)


def load_model(ckpt, args, device):
    model = DIS.build_model(args, device)
    DIS.load_weights_into(model, ckpt, device, Path(ckpt).stem)
    model.eval()
    return model


@torch.no_grad()
def predict_pc(model, batch, device, source):
    rgb, depth, pc, params, tv, names = batch
    rgb, depth, pc = rgb.to(device), depth.to(device), pc.to(device)
    params, tv = params.to(device), tv.to(device)
    m = model.module if hasattr(model, 'module') else model
    kw = {} if source == 'none' else {'visible': {source}}
    _, _, (_, _, ppc, _), _ = m(rgb, depth, pc, params, tv, **kw)
    return pc, ppc, list(names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/config_4m_distill_15k_all.yaml')
    ap.add_argument('--checkpoint', action='append', required=True,
                    help='repeat to compare checkpoints side by side')
    ap.add_argument('--label', action='append', default=None,
                    help='column heading per checkpoint (default: its stem)')
    ap.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    ap.add_argument('--source', default='pc',
                    help="only-visible modality, or 'none' for plain masked AE")
    ap.add_argument('--threshold', type=float, default=0.01,
                    help='Euclidean NN distance above which a point counts as '
                         'unpredicted (default: qal_threshold)')
    ap.add_argument('--direction', default='missed', choices=['missed', 'spurious'])
    ap.add_argument('--num_samples', type=int, default=4)
    ap.add_argument('--elev', type=float, default=20.0)
    ap.add_argument('--azim', type=float, default=45.0)
    ap.add_argument('--point_size', type=float, default=2.5)
    ap.add_argument('--out', default='vis_unpredicted')
    ap.add_argument('--export_json', default=None,
                    help='also dump positions + per-point NN distance here, for '
                         'the interactive viewer (threshold becomes a slider)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()

    labels = a.label or [Path(c).stem for c in a.checkpoint]
    if len(labels) != len(a.checkpoint):
        raise SystemExit(f"--label given {len(labels)}x for {len(a.checkpoint)} "
                         f"checkpoints; give one per checkpoint or none at all")

    cfg = yaml.safe_load(open(a.config))
    args = DIS.config_to_namespace(cfg)
    device = torch.device(a.device)

    root = cfg['data']['data_root']
    ds = SorghumDataset4M(f"{root}/{a.split}", img_size=args.img_size,
                          num_points=args.num_points, max_leaves=args.max_leaves)
    dl = DataLoader(ds, batch_size=a.num_samples, shuffle=False, num_workers=4)
    batch = next(iter(dl))

    # Every checkpoint sees the same plants, in the same order, so a difference
    # between columns is the model and nothing else.
    cols = []
    for ckpt, lab in zip(a.checkpoint, labels):
        model = load_model(ckpt, args, device)
        gt, pred, names = predict_pc(model, batch, device, a.source)
        cols.append((lab, gt, pred, names))
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    n_s, n_c = a.num_samples, len(cols)
    fig = plt.figure(figsize=(4.6 * n_c, 4.9 * n_s))
    summary = []

    for si in range(n_s):
        for ci, (lab, gt, pred, names) in enumerate(cols):
            pts, dist = per_point_nn(pred[si], gt[si], a.direction)
            pts = pts.cpu().numpy()
            dist = dist.cpu().numpy()
            bad = dist > a.threshold
            pct = 100.0 * bad.mean()
            summary.append((names[si], lab, pct, float(dist.mean())))

            ax = fig.add_subplot(n_s, n_c, si * n_c + ci + 1, projection='3d')
            # plot covered first so red sits on top and never hides behind grey
            ax.scatter(pts[~bad, 0], pts[~bad, 2], pts[~bad, 1],
                       c=COVERED, s=a.point_size, linewidths=0, alpha=.55)
            ax.scatter(pts[bad, 0], pts[bad, 2], pts[bad, 1],
                       c=MISSED, s=a.point_size * 1.6, linewidths=0)
            ax.view_init(a.elev, a.azim)
            ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
            ax.grid(False)
            word = 'missed' if a.direction == 'missed' else 'spurious'
            ax.set_title(f"{lab}\n{names[si]} — {pct:.1f}% {word}",
                         fontsize=9, fontweight='bold', pad=2)

    ttl = ('GT points with no prediction within {t} (red = model missed it)'
           if a.direction == 'missed' else
           'Predicted points with no GT within {t} (red = model invented it)')
    fig.suptitle(ttl.format(t=a.threshold) + f'   ·   source = {a.source}',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    path = out / f'unpredicted_{a.direction}_src-{a.source}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    if a.export_json:
        # Ship the DISTANCE per point, not the boolean. The threshold is the
        # one thing a reader will want to move -- 0.01 is the loss's midpoint,
        # not a natural visual cut -- and a baked-in mask forecloses that.
        # Positions are rounded to 3dp and distances to 4dp: the clouds are
        # unit-sphere normalised, so that is ~0.1% of the plant radius, well
        # below anything visible, and it roughly halves the file.
        import json
        payload = {'threshold_default': a.threshold, 'direction': a.direction,
                   'source': a.source, 'samples': []}
        for si in range(n_s):
            entry = {'name': cols[0][3][si], 'clouds': []}
            for lab, gt, pred, names in cols:
                pts, dist = per_point_nn(pred[si], gt[si], a.direction)
                entry['clouds'].append({
                    'label': lab,
                    'xyz': [round(float(v), 3) for v in pts.cpu().numpy().ravel()],
                    'nn':  [round(float(v), 4) for v in dist.cpu().numpy().ravel()],
                })
            payload['samples'].append(entry)
        jp = Path(a.export_json); jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(payload, separators=(',', ':')))
        print(f"wrote {jp}  ({jp.stat().st_size/1e6:.1f} MB)")

    print(f"\nwrote {path}")
    print(f"\n{'sample':<22}{'checkpoint':<26}{'% ' + a.direction:>12}{'mean NN':>11}")
    for name, lab, pct, mean_d in summary:
        print(f"{name:<22}{lab:<26}{pct:>11.1f}%{mean_d:>11.5f}")
    for lab in labels:
        rows = [p for _, l, p, _ in summary if l == lab]
        print(f"\n{lab}: mean {np.mean(rows):.1f}% {a.direction} "
              f"over {len(rows)} samples at threshold {a.threshold}")


if __name__ == '__main__':
    main()
