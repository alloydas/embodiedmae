"""Empirical statistics behind the v2 param encoding in embodied_mae_4m.py.

v1 divides every param by a hand-picked constant (_PLANT_SCALE / _LEAF_SCALE),
which leaves most of them squeezed into a sliver of [0, 1] (branching_angle
uses 2-12 % of the range) and five plant dims constant at 0.  v2 z-scores the
linear params with the mean/std printed here, encodes angles as sin/cos, and
drops the constant dims.  The geometry-conditioning token z-scores the point
cloud's normalisation radius with the same kind of statistic.

Re-run this and paste the printed arrays into embodied_mae_4m.py whenever the
dataset changes:

    python compute_param_stats.py /path/to/Sorghum_15K --num_samples 800
"""

import argparse
import random
from pathlib import Path

import numpy as np
import yaml

_Loader = getattr(yaml, 'CSafeLoader', yaml.SafeLoader)

REQUIRED_LEAF = ('starting_point', 'length', 'roll_angle', 'branching_angle',
                 'waviness_frequency', 'waviness_period_start')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('data_root')
    ap.add_argument('--split', default='train')
    ap.add_argument('--num_samples', type=int, default=800)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    split_dir = Path(args.data_root) / args.split
    folders = sorted(p for p in split_dir.iterdir() if p.is_dir())
    random.Random(args.seed).shuffle(folders)

    plant_rows, leaf_rows, radii = [], [], []
    for folder in folders:
        if len(plant_rows) >= args.num_samples:
            break
        yml = next(folder.glob('*_spline.yml'), None)
        if yml is None:
            continue
        with open(yml) as f:
            plant = yaml.load(f, Loader=_Loader)['Sorghums'][0]
        p = plant['Parameters']
        leaves = [l for l in plant['Leaves'] if all(k in l for k in REQUIRED_LEAF)]
        plant_rows.append([float(p['stem_length']), float(p['stem_direction'][0]),
                           float(p['stem_direction'][2]), float(len(leaves))])
        leaf_rows += [[float(l['starting_point']), float(l['length']),
                       float(l['branching_angle']), float(l['waviness_frequency'])]
                      for l in leaves]
        ply = next(folder.glob('*_nc_cam.ply'), None)
        if ply is not None:
            import open3d as o3d
            pts = np.asarray(o3d.io.read_point_cloud(str(ply)).points)
            if len(pts):
                radii.append(np.linalg.norm(pts - pts.mean(0), axis=1).max())

    P, L, R = np.array(plant_rows), np.array(leaf_rows), np.array(radii)
    fmt = lambda a: '[' + ', '.join(f'{v:.6g}' for v in a) + ']'
    print(f"# {len(P)} plants, {len(L)} leaves, {len(R)} clouds from {split_dir}")
    print("# plant z-scored dims: [stem_length, stem_dir_x, stem_dir_z, n_leaves]")
    print(f"_PLANT_MEAN_V2 = np.array({fmt(P.mean(0))}, np.float32)")
    print(f"_PLANT_STD_V2  = np.array({fmt(P.std(0))}, np.float32)")
    print("# leaf z-scored dims: [starting_point, length, branching_angle, waviness_frequency]")
    print(f"_LEAF_MEAN_V2  = np.array({fmt(L.mean(0))}, np.float32)")
    print(f"_LEAF_STD_V2   = np.array({fmt(L.std(0))}, np.float32)")
    print("# point-cloud normalisation radius (max distance from the centroid)")
    print(f"_PC_RADIUS_MEAN, _PC_RADIUS_STD = {R.mean():.6g}, {R.std():.6g}")


if __name__ == '__main__':
    main()
