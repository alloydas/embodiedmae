"""Export the "Generated from one photograph" clouds for the interactive viewer.

The static plate on the results page is four camera views of ONE held-out
validation plant, columns RGB / GT cloud / generated cloud / GT depth /
generated depth.  Only the two cloud columns are needed here.

PROVENANCE.  That plate is not the output of any figure script in the repo --
the page footer credits `make_rgb2pc_figure.py`, but that script renders four
DIFFERENT plants (Sorghum_10001_00, 10016_00, 1001_00, 10065_00) at view 00
from `outputs/4m_distill_15k_all/`, in a four-column layout with an overlay and
no depth.  The plate's own caption gives the real source away: "the training
visualiser samples consecutive folders".  Its panels are crops of
`visualize_crossmodal` dumps in
`outputs/4m_distill_15k_rgb2pc_full/visualizations/`, which writes six samples
per dump -- Sorghum_10001_00 .. _05, i.e. six consecutive views of one plant.

Which four of the six was settled by pixels, not by guessing.  Matching the
plate's ground-truth depth column against the raw `depth.png` of all ten views
of Sorghum_10001 (normalised cross-correlation on the silhouette) gives
0.92 / 0.91 / 0.75 / 0.81 for views 00 / 01 / 02 / 04 against <= 0.38 for every
other view, so the rows are views 00, 01, 02 and 04 -- view 03 is skipped.  The
same ranking comes back independently from the visualiser PNGs' own GT-depth
panels (0.98 / 0.99 / 0.83 / 0.88).  The plate's first column correlates with
the visualiser's `RGB <- INPUT` panel (0.97) rather than its `GT RGB` panel
(0.93): the reader is shown the model's RGB reconstruction, not the raw render.

CHECKPOINT.  The visualiser dumps every ten epochs and the run's best epoch is
23, so the plate was cropped from the epoch-020 or epoch-030 dump ("near the
best checkpoint", as the page says); the two cannot be told apart from pixels,
because RGB reconstruction and generated depth barely move over ten epochs.
The clouds here come from `best_model.pth` (epoch 23) itself, which is the
checkpoint the page's numbers are quoted from.

SEEDS.  The static plate's own point draws are NOT reproducible: the visualiser
pulls its batch through a multi-worker DataLoader, so the 8196-point resample
inside `SorghumDataset.load_pointcloud` is seeded by torch's per-worker RNG,
which depends on the epoch and the worker count of a finished 8-GPU job.  This
script therefore seeds explicitly and states the scheme rather than pretending
to reproduce that draw; a resample is worth 0.0001-0.00025 of Chamfer, so the
numbers below sit near, not exactly on, the visualiser's printed values
(epoch 020: 0.00097 / 0.00071 / 0.00054 / 0.00061 for views 00/01/02/04;
epoch 030: 0.00091 / 0.00065 / 0.00051 / 0.00053).

Points are packed by `export_pc_web.pack`: subsampled, quantised to int16 over
[-1, 1] and base64'd, so the payload inlines into the page rather than a fetch
the artifact CSP would block.
"""
import argparse, json, pathlib
import numpy as np
import torch

from embodied_mae_4m import embodied_mae_4m_base
from sorghum_dataset_4m import SorghumDataset4M
from eval_rgb2pc_quant import chamfer_both
from export_pc_web import pack

CKPT = 'outputs/4m_distill_15k_rgb2pc_full/best_model.pth'
DATA_ROOT = '/work/mech-ai-scratch/alloy/shorgum_data/new_data_50K/Sorghum_15K'
PLANT = 'Sorghum_10001'
VIEWS = [0, 1, 2, 4]          # the four rows of the static plate; 03 is skipped
SEED = lambda v: 4200 + 13 * v

# View index is an exact elevation ladder: the cameraToWorld forward vector
# f = -M[:3, 2] satisfies f_y = -0.9 + 0.2 * view across every folder, so the
# view-direction elevation is arcsin of that.  Negative = camera looking down.
ELEV = lambda v: float(np.degrees(np.arcsin(-0.9 + 0.2 * v)))


def camera_elevation(folder):
    """Elevation read from camera_pose.json, or None if the file is missing."""
    f = folder / 'camera_pose.json'
    if not f.is_file():
        return None
    M = np.array(json.loads(f.read_text())['cameraToWorld']).reshape(4, 4)
    return float(np.degrees(np.arcsin(np.clip(-M[:3, 2][1], -1.0, 1.0))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=CKPT)
    ap.add_argument('--data_root', default=DATA_ROOT)
    ap.add_argument('--split', default='val')
    ap.add_argument('--plant', default=PLANT)
    ap.add_argument('--views', default=','.join(str(v) for v in VIEWS))
    ap.add_argument('--num_points', type=int, default=8196)
    ap.add_argument('--n_points', type=int, default=2600, help='points per exported cloud')
    ap.add_argument('--out', default='vis_gallery/pc_fourview.json')
    args = ap.parse_args()

    views = [int(v) for v in args.views.split(',')]
    root = pathlib.Path(args.data_root) / args.split

    ds = SorghumDataset4M(args.data_root, split=args.split, num_points=args.num_points)
    by_name = {f.name: i for i, f in enumerate(ds.samples)}

    model = embodied_mae_4m_base(target_points=args.num_points).eval()
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    sd = ck.get('model_state_dict', ck)
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, 'checkpoint does not match the model'
    epoch = int(ck.get('epoch', -1))
    print(f'checkpoint {args.checkpoint}  epoch {epoch}  split {args.split}  '
          f'plant {args.plant}  views {views}')

    rng = np.random.default_rng(0)
    rows = []
    for v in views:
        name = f'{args.plant}_{v:02d}'
        if name not in by_name:
            print(f'  skip {name}: not in the {args.split} split')
            continue
        np.random.seed(SEED(v))            # the loader resamples the .ply per read
        rgb, depth, pc, par, tv, nm = ds[by_name[name]]
        b = lambda t: t.unsqueeze(0)
        with torch.no_grad():
            _, _, (_, _, pred, _), _ = model(b(rgb), b(depth), b(pc), b(par), b(tv),
                                             visible=['rgb'])
        f_, r_ = chamfer_both(pred, pc.unsqueeze(0))
        f_, r_ = float(f_), float(r_)
        cd = f_ + r_

        elev = camera_elevation(root / name)
        if elev is None:
            elev = ELEV(v)
        assert abs(elev - ELEV(v)) < 1e-6, f'{name}: elevation off the ladder'
        look = 'looking down' if elev < 0 else 'looking up'

        rows.append({
            'label': f'view {v:02d}',
            'name': nm,
            'chamfer': round(cd, 8),
            'clouds': {'gt':  pack(pc.numpy(), args.n_points, rng),
                       'gen': pack(pred[0].numpy(), args.n_points, rng)},
            'meta': [['camera elevation', f'{elev:+.1f}° ({look})'],
                     ['view index', f'{v:02d} of 09'],
                     ['accuracy  pred→gt', f'{f_:.5f}'],
                     ['completeness  gt→pred', f'{r_:.5f}']],
        })
        print(f'  {nm:<20} elev {elev:+6.1f}  chamfer {cd:.5f} '
              f'(pred->gt {f_:.5f}  gt->pred {r_:.5f})')

    out = {
        'id': 'four_views',
        'source_figure': 'Generated from one photograph',
        'checkpoint': args.checkpoint,
        'epoch': epoch,
        'split': args.split,
        'n_points': args.n_points,
        'panels': [{'key': 'gt',  'label': 'ground truth',              'color': '#9aa7ad'},
                   {'key': 'gen', 'label': 'generated from RGB alone',  'color': '#1baf7a'}],
        'rows': rows,
        'note': ('The static plate is four crops from the training visualiser '
                 '(visualize_crossmodal in train_sorghum_4m_distill.py) dumped by the '
                 'outputs/4m_distill_15k_rgb2pc_full run, not from make_rgb2pc_figure.py, '
                 'which renders four different plants from a different run; matching the '
                 'plate’s ground-truth depth column against the raw depth.png of all ten '
                 'views of Sorghum_10001 identifies the rows as views 00, 01, 02 and 04 '
                 '(NCC 0.92/0.91/0.75/0.81 against ≤0.38 for every other view), and the '
                 'clouds here are regenerated from best_model.pth, epoch 23, with '
                 'np.random.seed(4200 + 13*view) before each read because the visualiser’s '
                 'own DataLoader-worker resample seed is not recoverable.'),
    }
    p = pathlib.Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, separators=(',', ':')))
    print(f'wrote {p}  ({p.stat().st_size / 1024:.0f} KB, {len(rows)} rows)')


if __name__ == '__main__':
    main()
