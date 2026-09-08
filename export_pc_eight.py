"""Export the "Eight different plants" gallery as 3D points for the web viewer.

The static plate on the results page is gen_rgb2pc_gallery.py's output: eight
VALIDATION folders picked as (i * 137) % len(ds) -- stride 137 is coprime with
the 10 views per plant, so no two rows are the same plant -- pushed through the
epoch-23 rgb2pc distillation checkpoint with visible=['rgb'].  This reproduces
that selection exactly and packs ground truth + generated cloud per row.

Caveat on the numbers.  gen_rgb2pc_gallery.py never seeds numpy, and
SorghumDataset.load_pointcloud draws a fresh random 8196-of-104k subsample from
the .ply on every read, so the exact clouds behind the static plate came from an
unseeded global RNG and cannot be re-created.  This script seeds per row
(SEED0 + k) so the interactive figure is reproducible; the Chamfer values
therefore land near, not on, the ones recorded in vis_gallery/gallery.json.
The difference is resampling noise -- --sensitivity prints its size.
"""
import argparse, base64, json, pathlib
import numpy as np
import torch

from embodied_mae_4m import embodied_mae_4m_base
from sorghum_dataset_4m import SorghumDataset4M
from eval_rgb2pc_quant import chamfer_both
from export_pc_web import pack

CKPT = 'outputs/4m_distill_15k_rgb2pc_full/best_model.pth'
SEED0 = 2000


def load_model(ckpt, num_points):
    model = embodied_mae_4m_base(target_points=num_points).eval()
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    sd = ck.get('model_state_dict', ck)
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, 'checkpoint does not match the model'
    return model, int(ck.get('epoch', -1))


def run_one(model, ds, i, seed):
    """(gt, gen, symmetric chamfer) for one folder at a fixed resample seed."""
    np.random.seed(seed)                       # the loader resamples per read
    rgb, depth, pc, par, tv, name = ds[i]
    b = lambda t: t.unsqueeze(0)
    with torch.no_grad():
        _, _, (_, _, pred, _), _ = model(b(rgb), b(depth), b(pc), b(par), b(tv),
                                         visible=['rgb'])
    f, r = chamfer_both(pred, pc.unsqueeze(0))
    return pc.numpy(), pred[0].numpy(), float(f + r)     # == chamfer_distance()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=CKPT)
    ap.add_argument('--data_root', default='/work/mech-ai-scratch/alloy/shorgum_data/new_data_50K/Sorghum_15K')
    ap.add_argument('--split', default='val')
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--stride', type=int, default=137)
    ap.add_argument('--num_points', type=int, default=8196)
    ap.add_argument('--n_points', type=int, default=2600, help='points kept per exported cloud')
    ap.add_argument('--sensitivity', type=int, default=0,
                    help='extra resample seeds per row, to size the seeding noise')
    ap.add_argument('--out', default='vis_gallery/pc_eight.json')
    args = ap.parse_args()

    ds = SorghumDataset4M(args.data_root, split=args.split, num_points=args.num_points)
    idx = [(i * args.stride) % len(ds) for i in range(args.n)]

    model, epoch = load_model(args.checkpoint, args.num_points)
    print(f'checkpoint {args.checkpoint}  epoch {epoch}  {len(ds)} folders  stride {args.stride}')

    rng = np.random.default_rng(0)
    rows, cds = [], []
    for k, i in enumerate(idx):
        gt, gen, cd = run_one(model, ds, i, SEED0 + k)
        name = ds.samples[i].name
        plant, view = name.split('_')[1], name.split('_')[2]
        rows.append(dict(label=plant, name=name, chamfer=round(cd, 8),
                         clouds=dict(gt=pack(gt, args.n_points, rng),
                                     gen=pack(gen, args.n_points, rng)),
                         meta=[['plant', name], ['camera view', view],
                               ['Chamfer', f'{cd:.5f}']]))
        cds.append(cd)
        extra = ''
        if args.sensitivity:
            alt = [run_one(model, ds, i, 90000 + 100 * k + s)[2] for s in range(args.sensitivity)]
            a = np.array([cd] + alt)
            extra = f'   seed spread {a.min():.5f}-{a.max():.5f} sd {a.std():.6f}'
        print(f'  {k+1}/{len(idx)}  {name:<20} chamfer {cd:.5f}{extra}')

    cds = np.array(cds)
    out = dict(
        id='eight_plants',
        source_figure='Eight different plants',
        checkpoint=args.checkpoint, epoch=epoch, split=args.split,
        n_points=args.n_points,
        panels=[{'key': 'gt', 'label': 'ground truth', 'color': '#9aa7ad'},
                {'key': 'gen', 'label': 'generated from RGB alone', 'color': '#1baf7a'}],
        rows=rows,
        note=('Same eight validation folders as the static plate: '
              'gen_rgb2pc_gallery.py --split val --n 8 --stride 137, indices '
              '(i*137) % 22500, epoch-23 rgb2pc distillation checkpoint, '
              'visible=[\'rgb\']; clouds re-run here with fixed resample seeds '
              f'{SEED0}+row because the original run left numpy unseeded.'))

    p = pathlib.Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, separators=(',', ':')))
    print(f'\nmean {cds.mean():.5f}  sd {cds.std():.5f}  '
          f'range {cds.min():.5f}-{cds.max():.5f}')
    print(f'wrote {p}  ({p.stat().st_size/1024:.0f} KB)')


if __name__ == '__main__':
    main()
