"""Export point clouds for the interactive viewer in the results page.

Two datasets, both from the source-masked checkpoint (smr50 seed 1, epoch 89):

  percentile : the eight plants of the percentile gallery, best -> worst, each
               as ground truth + the cloud generated from RGB alone.
  elevation  : one plant across six camera elevations. The TARGET is the same
               cloud every time, rotated into each view's camera frame, so what
               changes between rows is only the photograph the model saw.

Points are subsampled, quantised to int16 over [-1, 1] (resolution 3e-5, far
below anything visible) and base64'd, so the whole payload is a few hundred KB
inlined into the page rather than a fetch the artifact CSP would block.
"""
import argparse, base64, json, pathlib, re
import numpy as np
import torch

from embodied_mae_4m import embodied_mae_4m_base
from sorghum_dataset_4m import SorghumDataset4M
from eval_rgb2pc_quant import chamfer_both

CKPT = 'outputs/4m_distill_15k_rgb2pc_smr50/best_model.pth'
ELEV = lambda v: float(np.degrees(np.arcsin(-0.9 + 0.2 * v)))


def pack(pts, n, rng):
    """Subsample to n points and quantise to base64 int16."""
    if len(pts) > n:
        pts = pts[rng.choice(len(pts), n, replace=False)]
    q = np.clip(np.round(np.asarray(pts, np.float64) * 32767), -32767, 32767).astype('<i2')
    return base64.b64encode(q.tobytes()).decode('ascii')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_points', type=int, default=2600)
    ap.add_argument('--report', default='reports/quant_smr50_seed1_test.json')
    ap.add_argument('--plant', default='Sorghum_174')
    ap.add_argument('--views', default='0,2,4,6,8,9')
    ap.add_argument('--out', default='vis_gallery/pc_web.json')
    args = ap.parse_args()

    rep = json.loads(pathlib.Path(args.report).read_text())
    vals = np.asarray(rep['per_sample']['model']); names = rep['plants']
    order = np.argsort(vals)

    ds = SorghumDataset4M('/work/mech-ai-scratch/alloy/shorgum_data/new_data_50K/Sorghum_15K',
                          split='test', num_points=8196)
    by_name = {f.name: i for i, f in enumerate(ds.samples)}

    model = embodied_mae_4m_base(target_points=8196).eval()
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    sd = ck.get('model_state_dict', ck)
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected
    print(f'checkpoint epoch {ck.get("epoch")}  n_points {args.n_points}')

    rng = np.random.default_rng(0)

    def one(name, seed):
        """Run the model on one folder and return (gt, gen, chamfer)."""
        np.random.seed(seed)                      # the loader resamples per read
        rgb, depth, pc, par, tv, nm = ds[by_name[name]]
        b = lambda t: t.unsqueeze(0)
        with torch.no_grad():
            _, _, (_, _, pred, _), _ = model(b(rgb), b(depth), b(pc), b(par), b(tv), visible=['rgb'])
        f, r = chamfer_both(pred, pc.unsqueeze(0))
        return pc.numpy(), pred[0].numpy(), float(f + r)

    out = {'checkpoint': CKPT, 'epoch': int(ck.get('epoch', -1)),
           'n_points': args.n_points, 'split': 'test'}

    # ── percentile rows: identical plants and seeds to gen_rgb2pc_gallery_pct.py ──
    pcts, rows, seen = [0, 10, 25, 50, 75, 90, 97, 100], [], set()
    for k, p in enumerate(pcts):
        r_ = int(round(p / 100 * (len(order) - 1)))
        while r_ in seen and r_ < len(order) - 1:
            r_ += 1
        seen.add(r_)
        name = names[order[r_]]
        gt, gen, cd = one(name, 1000 + k)
        rows.append(dict(label={0: 'best', 100: 'worst'}.get(p, f'p{p}'), pct=p, name=name,
                         chamfer=round(cd, 8),
                         gt=pack(gt, args.n_points, rng), gen=pack(gen, args.n_points, rng)))
        print(f'  p{p:<4} {name:<20} chamfer {cd:.5f}')
    out['percentile'] = rows

    # ── elevation sweep: one plant, the target fixed, only the input image changes ──
    rows = []
    for v in [int(x) for x in args.views.split(',')]:
        name = f'{args.plant}_{v:02d}'
        if name not in by_name:
            print(f'  skip {name}'); continue
        gt, gen, cd = one(name, 7000 + 13 * v)
        rows.append(dict(label=f'{ELEV(v):+.0f}°', view=v, elev=round(ELEV(v), 2), name=name,
                         chamfer=round(cd, 8),
                         gt=pack(gt, args.n_points, rng), gen=pack(gen, args.n_points, rng)))
        print(f'  view {v}  elev {ELEV(v):+6.1f}  chamfer {cd:.5f}')
    out['elevation'] = rows
    out['plant'] = args.plant

    p = pathlib.Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, separators=(',', ':')))
    print(f'wrote {p}  ({p.stat().st_size/1024:.0f} KB)')


if __name__ == '__main__':
    main()
