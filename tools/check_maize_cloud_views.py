#!/usr/bin/env python3
"""Is a maize `pointcloud_cam.ply` the whole plant, or what its own camera sees?

Sorghum's `<plant>_nc_cam.ply` is the COMPLETE plant cloud moved into the view's
camera frame (checked point for point; see eval/pc_view_masks.py), which is why the
sorghum partial-cloud study had to build visibility masks. This script settles the
same question for maize, where every view folder carries its own `pointcloud_cam.ply`
of exactly 8,192 points and no mesh is on disk. Three measurements per plant:

  own      fraction of view v's cloud that camera v sees: the point projects onto a
           foreground pixel whose rendered depth (camera_pose.json: "(z - near) /
           (far - near)") is within TAU of the point's own z. A complete cloud would
           have a large hidden fraction from any single camera; a depth-camera
           capture has ~none.
  cross    the same for view v's cloud seen from camera 0 (clouds moved through
           cameraToWorld / worldToCamera). If clouds were complete this would match
           `own`; if each is its camera's surface it falls to what camera 0 shares.
  backproj camera 0's depth image back-projected to points, and the NN distance from
           each cloud point to that set, both ways.

    python tools/check_maize_cloud_views.py --data-root /work/hdd/bepk/adas14/Maize \
        --split val --plants 20
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

TAU = 0.005     # 5 mm, in the renderer's metres; the clouds sit ~2-4 mm off each other


def pose(folder):
    m = json.loads((folder / 'camera_pose.json').read_text())
    return (m, np.asarray(m['cameraToWorld'], dtype=np.float64).reshape(4, 4),
            np.asarray(m['worldToCamera'], dtype=np.float64).reshape(4, 4))


def depth_m(folder, m):
    """Rendered depth in metres (0 = background), from the packed RGBA uint32."""
    a = np.asarray(Image.open(folder / 'depth.png')).astype(np.float64)
    d = (a[..., 0] * 256 ** 3 + a[..., 1] * 256 ** 2 + a[..., 2] * 256 + a[..., 3]) / (256 ** 4 - 1)
    z = m['near'] + d * (m['far'] - m['near'])
    z[d <= 1e-6] = 0.0
    return z


def cloud(folder):
    import open3d as o3d
    return np.asarray(o3d.io.read_point_cloud(str(folder / 'pointcloud_cam.ply')).points, dtype=np.float64)


def project(Pc, m):
    """Camera-frame points (camera looks down -z) -> pixel (u, v) and depth z."""
    K = m['intrinsics']
    z = -Pc[:, 2]
    zs = np.where(z > 1e-9, z, 1e-9)
    return K['cx'] + K['fx'] * Pc[:, 0] / zs, K['cy'] - K['fy'] * Pc[:, 1] / zs, z


def visible(Pc, m, Z, tau=TAU):
    """(seen, hidden, off_silhouette): camera-frame points against the camera's depth."""
    K = m['intrinsics']
    u, v, z = project(Pc, m)
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    inf = (z > 0) & (ui >= 0) & (ui < K['width']) & (vi >= 0) & (vi < K['height'])
    zz = np.zeros(len(z))
    zz[inf] = Z[vi[inf], ui[inf]]
    fg = inf & (zz > 0)
    seen = fg & (np.abs(z - zz) <= tau)
    hidden = fg & (z - zz > tau)
    return seen, hidden, inf & ~fg


def backproject(m, Z, n=20000, rng=None):
    K = m['intrinsics']
    vi, ui = np.nonzero(Z > 0)
    if rng is not None and len(ui) > n:
        k = rng.choice(len(ui), n, replace=False)
        vi, ui = vi[k], ui[k]
    z = Z[vi, ui]
    x = (ui - K['cx']) * z / K['fx']
    y = -(vi - K['cy']) * z / K['fy']
    return np.stack([x, y, -z], 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--split', default='val')
    ap.add_argument('--plants', type=int, default=20)
    ap.add_argument('--stride', type=int, default=97)
    args = ap.parse_args()
    root = Path(args.data_root)
    names = [l.strip() for l in (root / f'plants_{args.split}.txt').read_text().splitlines() if l.strip()]
    names = names[::args.stride][:args.plants]
    rng = np.random.default_rng(0)

    own, cross, hid_own, bp_c2d, bp_d2c, union_hidden = [], [], [], [], [], []
    for name in names:
        folders = [root / args.split / f'{name}_{v:02d}' for v in range(10)]
        P = [cloud(f) for f in folders]
        M = [pose(f) for f in folders]
        Z = [depth_m(f, m[0]) for f, m in zip(folders, M)]
        # own camera
        o = []
        for v in range(10):
            s, h, _ = visible(P[v], M[v][0], Z[v])
            o.append(s.mean()); hid_own.append(h.mean())
        own.append(np.mean(o))
        # every cloud moved into camera 0
        c = []
        W0 = M[0][2]
        for v in range(10):
            Pw = (M[v][1][:3, :3] @ P[v].T).T + M[v][1][:3, 3]
            P0 = (W0[:3, :3] @ Pw.T).T + W0[:3, 3]
            s, h, _ = visible(P0, M[0][0], Z[0])
            c.append(s.mean())
            if v:
                union_hidden.append(h.mean())
        cross.append(c)
        # back-projection of camera 0's depth against cloud 0
        B = backproject(M[0][0], Z[0], rng=rng)
        bp_c2d.append(np.median(cKDTree(B).query(P[0])[0]) * 1000)
        bp_d2c.append(np.median(cKDTree(P[0]).query(B)[0]) * 1000)
        print(f'{name}: own-camera seen {np.mean(o):.3f} (hidden {np.mean(hid_own[-10:]):.3f})  '
              f'cloud v in cam 0: ' + ' '.join(f'{x:.2f}' for x in c) +
              f'  backproj NN cloud->depth {bp_c2d[-1]:.1f} mm, depth->cloud {bp_d2c[-1]:.1f} mm', flush=True)

    cross = np.array(cross)
    print('\nSUMMARY over', len(names), 'plants')
    print(f'  a view\'s cloud seen by ITS OWN camera:      {np.mean(own):.3f}  (hidden behind the surface: {np.mean(hid_own):.3f})')
    print(f'  other views\' clouds seen by camera 0:       {cross[:, 1:].mean():.3f}  (hidden: {np.mean(union_hidden):.3f})')
    print(f'  camera-0 depth back-projected vs cloud 0:   median NN cloud->depth {np.median(bp_c2d):.1f} mm, '
          f'depth->cloud {np.median(bp_d2c):.1f} mm')
    verdict = ('each view\'s cloud is (close to) the surface its own camera sees, not the whole plant'
               if np.mean(hid_own) < 0.05 and np.mean(union_hidden) > 0.10 else
               'the clouds look complete: a single camera hides a similar fraction of every view\'s cloud')
    print('  ->', verdict)


if __name__ == '__main__':
    main()
