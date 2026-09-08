"""Export the "RGB in, geometry out" plate as rotatable point clouds.

Replaces the static four-row plate (figures_fixed/rgb2pc_plate.png, built by
make_rgb2pc_figure.py) with the raw 3D points behind it, so the results page can
spin each cloud instead of showing one fixed matplotlib camera.

Provenance is copied from make_rgb2pc_figure.py verbatim:

  checkpoint : outputs/4m_distill_15k_all/best_model.pth   (hard-coded there)
  split      : val
  plants     : Sorghum_10001_00, 10016_00, 1001_00, 10065_00  (hard-coded there)
  input      : visible={'rgb'} — depth, point cloud and spline fully masked
  chamfer    : embodied_mae.chamfer_distance, i.e. the SUM of both directions,
               which is exactly chamfer_both(...)[0] + [1] used elsewhere.

One thing cannot be copied: make_rgb2pc_figure.py never seeds numpy, and
SorghumDataset4M draws a fresh random 8196-point subsample of the .ply on every
read, so the ground-truth clouds of the static plate are not recoverable. That
only perturbs the TARGET. The generated cloud is a function of the RGB image
alone (a masked modality contributes zero encoder tokens; the decoder fills it
from mask tokens), so the green cloud exported here is bit-identical to the
plate's, and only the grey one is a different draw of the same surface. Seeds
are fixed here (SEED_BASE + row) so this export is reproducible, and --probe
reports the Chamfer spread across draws to show the residual is sampling noise.

The fourth "overlay" column of the plate needs no export: the viewer draws the
two panels on top of each other.
"""
import argparse, json, pathlib
import numpy as np
import torch

from embodied_mae_4m import embodied_mae_4m_base
from sorghum_dataset_4m import SorghumDataset4M
from eval_rgb2pc_quant import chamfer_both
from export_pc_web import pack                      # shared int16/base64 packer

DATA_ROOT = '/work/mech-ai-scratch/alloy/shorgum_data/new_data_50K/Sorghum_15K'
CKPT = 'outputs/4m_distill_15k_all/best_model.pth'   # as in make_rgb2pc_figure.py
PLANTS = ['Sorghum_10001_00', 'Sorghum_10016_00', 'Sorghum_1001_00', 'Sorghum_10065_00']
PLATE_CD = [0.00100, 0.00156, 0.00111, 0.00097]      # printed in the page's figcaption
SEED_BASE = 4000

GT_C, GEN_C = '#9aa7ad', '#1baf7a'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_points', type=int, default=2600)
    ap.add_argument('--out', default='vis_gallery/pc_rgbgeo.json')
    ap.add_argument('--probe', type=int, default=2,
                    help='extra GT resample draws per plant, Chamfer only, to size the noise')
    args = ap.parse_args()

    ds = SorghumDataset4M(DATA_ROOT, split='val', num_points=8196)
    names = [p.name for p in ds.samples]
    idx = {}
    for want in PLANTS:
        hit = [i for i, n in enumerate(names) if want in n]   # same match as the plate
        if not hit:
            raise SystemExit(f'{want} not in val split')
        idx[want] = hit[0]

    model = embodied_mae_4m_base(target_points=8196).eval()
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    sd = ck.get('model_state_dict', ck)
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (len(missing), len(unexpected))
    epoch = int(ck.get('epoch', -1))
    print(f'{CKPT}  epoch {epoch}  n_points {args.n_points}')

    rng = np.random.default_rng(0)
    rows = []
    for k, want in enumerate(PLANTS):
        i = idx[want]
        np.random.seed(SEED_BASE + k)            # the loader resamples per read
        rgb, depth, pc, par, tv, nm = ds[i]
        b = lambda t: t.unsqueeze(0)
        with torch.no_grad():
            _, _, (_, _, pred, _), _ = model(b(rgb), b(depth), b(pc), b(par), b(tv),
                                             visible=['rgb'])
        f, r = chamfer_both(pred, pc.unsqueeze(0))
        cd = float(f + r)

        # same prediction, other draws of the same surface -> pure sampling noise
        alts = []
        for j in range(args.probe):
            np.random.seed(SEED_BASE + 100 * (j + 1) + k)
            gt2 = ds[i][2]
            f2, r2 = chamfer_both(pred, gt2.unsqueeze(0))
            alts.append(float(f2 + r2))

        plant = nm.rsplit('_', 2)[-2]
        rows.append(dict(
            label=f'plant {plant}',
            name=nm,
            chamfer=round(cd, 8),
            clouds=dict(gt=pack(pc.numpy(), args.n_points, rng),
                        gen=pack(pred[0].numpy(), args.n_points, rng)),
            meta=[['camera view', nm.rsplit('_', 1)[-1]],
                  ['input', 'RGB only — depth, PC, spline masked'],
                  ['static plate Chamfer', f'{PLATE_CD[k]:.5f}']],
        ))
        print(f'  {nm:<18} chamfer {cd:.5f}  (plate {PLATE_CD[k]:.5f}, '
              f'other draws {" ".join(f"{a:.5f}" for a in alts)})')

    out = dict(
        id='rgb_in_geometry_out',
        source_figure='RGB in, geometry out',
        checkpoint=CKPT, epoch=epoch, split='val',
        n_points=args.n_points,
        panels=[dict(key='gt', label='ground truth', color=GT_C),
                dict(key='gen', label='generated from RGB alone', color=GEN_C)],
        rows=rows,
        note=('Same four plants and the same hard-coded checkpoint as '
              'make_rgb2pc_figure.py (outputs/4m_distill_15k_all/best_model.pth, '
              f'epoch {epoch}, val split, visible={{rgb}}); that script left the '
              'dataset\'s 8196-point resample unseeded, so the generated clouds match '
              'the plate exactly while ground truth is a fresh seeded draw of the same '
              'surface.'),
    )
    p = pathlib.Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, separators=(',', ':')))
    print(f'wrote {p}  ({p.stat().st_size/1024:.0f} KB)')


if __name__ == '__main__':
    main()
