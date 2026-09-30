#!/usr/bin/env python3
"""Which points of each sorghum cloud a given camera actually sees.

This builds the inputs of the 1-view vs 3-view point-cloud evaluation
(eval/pc_view_eval.py). Every cloud the model trained on is the COMPLETE plant:
`<plant>_nc_cam.ply` is the full `<plant>_nc.ply` moved rigidly into one
camera's frame (checked point for point, max error 4e-16 over all ten views), so
no model here has ever seen a partial scan. A depth camera sees only the surface
facing it. For every plant and every requested view, this script marks which of
the cloud's points that camera sees:

  1. Snap each point to the closest point on the plant's source mesh,
     `<plant>.obj`, which is the geometry that was rendered. The cloud sits
     1.3-1.6 mm (median) and at most 5 mm off that surface, so testing the raw
     point would call it "occluded" by the very leaf it lies on.
  2. Cast a ray from the camera centre (`camera_pose.json` "position") to the
     snapped point. The point is visible unless the ray hits something more than
     TAU before reaching it.
  3. Clip to the camera frustum: vertical FOV 40 deg, square image, principal
     point at the centre. These were fitted from the renders' silhouettes (f is
     about 1406 px at 1024 px). A plant taller than the frame loses its top or
     base here, as a real camera would.

Masks are stored per split as packed bits, in the cloud's own point order. That
order is the same in `_nc.ply` and in every view's `_nc_cam.ply`, and it is
asserted against view 00 for every plant. Plants are listed from
assignment.csv, never by listing a split directory (a readdir of a
105,000-entry split on this NFS can hang for most of an hour).

    python eval/pc_view_masks.py --splits val --limit 20 --validate 5   # smoke
    python eval/pc_view_masks.py --workers 32                           # all splits

Output: outputs/_pcview_masks/<split>__v<views>.npz, written atomically.
`--validate K` also checks K val plants against the renderer's own depth buffer.
Visible points should land on the depth silhouette at the depth the render
recorded; occluded points should land behind it.
"""
# Repo root on sys.path: this script lives in eval/ but imports top-level modules.
import sys as _sys
import pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import argparse
import csv
import json
import math
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

DATA_ROOT = '/work/mech-ai-scratch/alloy/shorgum_data/new_data_50K/Sorghum_15K'
REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / 'outputs' / '_pcview_masks'

FOV_DEG = 40.0       # vertical = horizontal (square renders)
IMG = 1024           # render resolution the FOV was fitted at
TAU = 5e-4           # 0.5 mm: a ray to a surface point that stops sooner is occluded
DEPTH_SCALE = 50.0   # depth.png decodes to z / 50 (plus a ~22 mm offset, fitted in --validate)


def mask_path(split, views, out_dir=OUT_DIR):
    return Path(out_dir) / f"{split}__v{''.join(f'{v:02d}' for v in views)}.npz"


def plants_of(data_root, split):
    """Plant ids of a split, from assignment.csv (no directory listing)."""
    with open(Path(data_root) / 'assignment.csv') as fh:
        return sorted(int(r['plant']) for r in csv.DictReader(fh) if r['split'] == split)


def _pose(folder):
    cp = json.loads((folder / 'camera_pose.json').read_text())
    return (np.asarray(cp['position'], dtype=np.float64),
            np.asarray(cp['worldToCamera'], dtype=np.float64).reshape(4, 4))


def _project(M, P):
    """World points -> (u, v, z) in the render's pixel grid (camera looks down -z)."""
    C = (M[:3, :3] @ P.T).T + M[:3, 3]
    z = -C[:, 2]
    f = (IMG / 2) / math.tan(math.radians(FOV_DEG / 2))
    zs = np.where(z > 1e-9, z, 1e-9)
    return IMG / 2 + f * C[:, 0] / zs, IMG / 2 - f * C[:, 1] / zs, z


def plant_visibility(data_root, split, plant, views):
    """Boolean visibility masks (one per view) over the plant cloud's points."""
    import open3d as o3d
    root = Path(data_root) / split
    base = root / f'Sorghum_{plant}_00'
    P = np.asarray(o3d.io.read_point_cloud(str(base / f'Sorghum_{plant}_nc.ply')).points,
                   dtype=np.float64)
    Pc0 = np.asarray(o3d.io.read_point_cloud(str(base / f'Sorghum_{plant}_nc_cam.ply')).points,
                     dtype=np.float64)
    _, M0 = _pose(base)
    if len(P) != len(Pc0):
        raise RuntimeError(f'{base}: _nc.ply has {len(P)} points, _nc_cam.ply {len(Pc0)}')
    err = float(np.abs((M0[:3, :3] @ P.T).T + M0[:3, 3] - Pc0).max())
    if err > 1e-5:
        # The masks index _nc_cam.ply by _nc.ply's point order; if the two ever
        # stopped being the same points in the same order, every mask is wrong.
        raise RuntimeError(f'{base}: _nc_cam.ply is not worldToCamera @ _nc.ply (max err {err:.2e})')

    mesh = o3d.io.read_triangle_mesh(str(base / f'Sorghum_{plant}.obj'))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    S = scene.compute_closest_points(o3d.core.Tensor(P.astype(np.float32)), nthreads=1)
    S = S['points'].numpy().astype(np.float64)

    masks = {}
    for v in views:
        cam, M = _pose(root / f'Sorghum_{plant}_{v:02d}')
        ray = S - cam
        t = np.linalg.norm(ray, axis=1)
        ray /= t[:, None]
        rays = np.hstack([np.broadcast_to(cam, S.shape), ray]).astype(np.float32)
        hit = scene.cast_rays(o3d.core.Tensor(rays), nthreads=1)['t_hit'].numpy()
        u, vv, z = _project(M, P)
        in_frame = (z > 0) & (u >= 0) & (u < IMG) & (vv >= 0) & (vv < IMG)
        masks[v] = (hit >= t - TAU) & in_frame
    return P, masks


def _work(job):
    data_root, split, plant, views = job
    try:
        P, masks = plant_visibility(data_root, split, plant, views)
    except Exception as e:  # reported and counted, never silently dropped
        return plant, None, repr(e)
    return plant, (len(P), {v: np.packbits(m) for v, m in masks.items()}), None


def build(data_root, split, views, workers, limit=None, out_dir=OUT_DIR):
    plants = plants_of(data_root, split)
    if limit:
        plants = plants[:limit]
    jobs = [(data_root, split, p, tuple(views)) for p in plants]
    res = {}
    fails = []
    t0 = time.time()
    with Pool(workers) as pool:
        for i, (plant, out, err) in enumerate(pool.imap_unordered(_work, jobs, chunksize=4)):
            if err:
                fails.append((plant, err))
            else:
                res[plant] = out
            if (i + 1) % 500 == 0 or i + 1 == len(jobs):
                print(f'  {split}: {i + 1}/{len(jobs)} plants  ({time.time() - t0:.0f}s)', flush=True)
    if fails:
        for p, e in fails[:10]:
            print(f'  FAILED plant {p}: {e}')
        raise SystemExit(f'{split}: {len(fails)} plants failed; no mask file written')

    order = sorted(res)
    n = np.asarray([res[p][0] for p in order], dtype=np.int64)
    arrays = {'plants': np.asarray(order, dtype=np.int64), 'n_points': n,
              'views': np.asarray(views, dtype=np.int64),
              'fov_deg': np.float64(FOV_DEG), 'tau': np.float64(TAU)}
    nbytes = (n + 7) // 8
    arrays['byte_offsets'] = np.concatenate([[0], np.cumsum(nbytes)]).astype(np.int64)
    for v in views:
        arrays[f'bits_v{v:02d}'] = np.concatenate([res[p][1][v] for p in order])
    out = mask_path(split, views, out_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + f'.tmp{os.getpid()}.npz')
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, out)

    cov = {v: [] for v in views}
    union = []
    for p in order:
        k, bits = res[p]
        m = {v: np.unpackbits(bits[v], count=k).astype(bool) for v in views}
        for v in views:
            cov[v].append(m[v].mean())
        union.append(np.logical_or.reduce([m[v] for v in views]).mean())
    print(f'  wrote {out}  ({len(order)} plants, {time.time() - t0:.0f}s)')
    for v in views:
        c = np.asarray(cov[v])
        print(f'    view {v:02d}: visible {c.mean():.3f} +- {c.std():.3f} of each cloud '
              f'(min {c.min():.3f}, max {c.max():.3f})')
    u = np.asarray(union)
    print(f'    union of views {views}: {u.mean():.3f} +- {u.std():.3f} (min {u.min():.3f})')
    return out


class Masks:
    """Read side: plant id -> {view: bool mask over that plant's cloud points}."""

    def __init__(self, split, views, out_dir=OUT_DIR):
        z = np.load(mask_path(split, views, out_dir))
        self.views = [int(v) for v in z['views']]
        self._n = dict(zip(z['plants'].tolist(), z['n_points'].tolist()))
        self._row = {p: i for i, p in enumerate(z['plants'].tolist())}
        self._off = z['byte_offsets']
        self._bits = {v: z[f'bits_v{v:02d}'] for v in self.views}

    def __contains__(self, plant):
        return plant in self._row

    def get(self, plant, views):
        i, n = self._row[plant], self._n[plant]
        a, b = self._off[i], self._off[i + 1]
        return np.logical_or.reduce(
            [np.unpackbits(self._bits[v][a:b], count=n).astype(bool) for v in views])

    def n_points(self, plant):
        return self._n[plant]


def validate(data_root, k, views, seed=0):
    """Check the ray-cast masks against the renderer's own depth buffer."""
    from PIL import Image
    import open3d as o3d  # noqa: F401  (plant_visibility imports it)
    plants = plants_of(data_root, 'val')
    rng = np.random.default_rng(seed)
    pick = rng.choice(plants, size=min(k, len(plants)), replace=False)
    print(f'\nValidation against depth.png on {len(pick)} val plants:')
    rows = []
    for plant in pick:
        P, masks = plant_visibility(data_root, 'val', int(plant), views)
        for v in views:
            folder = Path(data_root) / 'val' / f'Sorghum_{plant}_{v:02d}'
            a = np.asarray(Image.open(folder / 'depth.png')).astype(np.float64)
            D = (a[..., 0] * 256.0 ** 3 + a[..., 1] * 256.0 ** 2
                 + a[..., 2] * 256.0 + a[..., 3]) / (256.0 ** 4 - 1)
            _, M = _pose(folder)
            u, vv, z = _project(M, P)
            ok = (u >= 0) & (u < IMG) & (vv >= 0) & (vv < IMG) & (z > 0)
            ui = np.clip(u.astype(int), 0, IMG - 1)
            vi = np.clip(vv.astype(int), 0, IMG - 1)
            d = DEPTH_SCALE * D[vi, ui]
            fg = ok & (d > 0)
            res = z - d
            vis, occ = masks[v] & fg, (~masks[v]) & fg
            off = float(np.median(res[vis])) if vis.any() else float('nan')
            near = lambda m: float((np.abs(res[m] - off) < 0.005).mean()) if m.any() else float('nan')
            rows.append((int(plant), v, masks[v].mean(), off, near(vis), near(occ),
                         float(fg[masks[v]].mean()) if masks[v].any() else float('nan')))
            print(f'  plant {plant} view {v:02d}: visible {masks[v].mean():.3f}  '
                  f'render offset {off * 1000:+.1f} mm  within 5 mm of the render: '
                  f'visible {rows[-1][4]:.3f} vs occluded {rows[-1][5]:.3f}  '
                  f'visible on silhouette {rows[-1][6]:.3f}')
    r = np.asarray([x[2:] for x in rows], dtype=np.float64)
    print(f'  mean: visible {r[:, 0].mean():.3f}, offset {np.nanmean(r[:, 1]) * 1000:+.1f} mm, '
          f'within 5 mm visible {np.nanmean(r[:, 2]):.3f} vs occluded {np.nanmean(r[:, 3]):.3f}, '
          f'visible on silhouette {np.nanmean(r[:, 4]):.3f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--data-root', default=DATA_ROOT)
    ap.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    ap.add_argument('--views', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    ap.add_argument('--limit', type=int, default=None, help='first N plants per split (smoke test)')
    ap.add_argument('--out-dir', default=str(OUT_DIR))
    ap.add_argument('--validate', type=int, default=0, metavar='K',
                    help='also check K val plants against depth.png')
    args = ap.parse_args()
    if 0 not in args.views:
        raise SystemExit('view 00 must be included: it is the view every probe input comes from')
    for split in args.splits:
        print(f'\n== {split}: views {args.views}, FOV {FOV_DEG} deg, tau {TAU * 1000:.1f} mm')
        build(args.data_root, split, args.views, args.workers, args.limit, args.out_dir)
    if args.validate:
        validate(args.data_root, args.validate, args.views)


if __name__ == '__main__':
    main()
