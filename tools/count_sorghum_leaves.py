"""Estimate leaf counts for the real sorghum scans from their row point clouds (E10 first upload).

The 30 scanned sorghum plants behind tab:e10 carry a height label only, so the table's sorghum
Leaves +-1 cells are n/a. This tool asks how far an automatic count from the clouds themselves
can get: per camera stop it takes the TARGET plant's labelled points (the PCDs include points
hidden behind other leaves, so row occlusion does not remove geometry), strips the stem, clusters
the rest, and counts leaf-sized clusters. Per scan it reports the median / mode / spread of the
per-stop counts over both layouts.

THESE ARE ESTIMATES, NOT GROUND TRUTH. Sorghum leaves overlap at the whorl (clusters merge, an
undercount) and thin tips drop below clustering density (clusters split, an overcount). Use the
output to judge plausibility against a manual count, or validate against the field1000 CSV counts
(--ref-field1000) where scan ids overlap, before putting any number in the paper.

Run on the machine that holds /work/mech-ai/alloydas (CPU only; ~1-2 s per stop):
  python tools/count_sorghum_leaves.py --out reports/sorghum_leafcount_rows2026.csv \
      --viz-dir vis_gallery/leafcount --save-ply 6
  # smoke: one scan, three stops
  python tools/count_sorghum_leaves.py --rows d1_156248 --stops 1 9 18 --save-ply 3 \
      --out /tmp/leafcount_smoke.csv
Nova CPU job (no GPU, starts at once): sbatch --wait --partition=nova --account=mech-ai \
  --qos=normal --cpus-per-task=8 --mem=32G --time=02:00:00 --job-name=leafcount \
  --output=logs/%x_%j.out --wrap "python tools/count_sorghum_leaves.py --out reports/sorghum_leafcount_rows2026.csv"
"""
import argparse, csv, pathlib, re
from collections import Counter

import numpy as np

# The paths and the PCD reader mirror eval/real_rows.py (kept inline so this tool needs no torch).
ROOT = pathlib.Path('/work/mech-ai/alloydas')
SORGHUM_LAYOUTS = {'parallel': ROOT / 'sorghum_rows_parallel_25000pts_seed1_d1_156248-p1d1V_585524',
                   'perpendicular': ROOT / 'sorghum_rows_perpendicular_25000pts_seed1_d1_156248-p1d1V_585524'}
FIELD1000 = ROOT / 'e10_field_1000'


def read_pcd(path):                                     # = eval.real_rows.read_pcd
    raw = pathlib.Path(path).read_bytes(); i = raw.find(b'DATA'); j = raw.find(b'\n', i) + 1
    hdr = {l.split()[0]: l.split()[1:] for l in raw[:j].decode('ascii').splitlines() if l and not l.startswith('#')}
    if hdr['DATA'] != ['binary']:
        raise ValueError(f'{path}: DATA {hdr["DATA"]}, expected binary')
    kind = {('F', '4'): '<f4', ('U', '4'): '<u4', ('U', '1'): 'u1', ('I', '4'): '<i4'}
    dt = np.dtype([(f, kind[(t, s)]) for f, t, s in zip(hdr['FIELDS'], hdr['TYPE'], hdr['SIZE'])])
    return np.frombuffer(raw, dtype=dt, count=int(hdr['POINTS'][0]), offset=j)


def row_id(d):                                          # 'row_d1_156248_seed1' -> 'd1_156248'
    return d.name[len('row_'):].rsplit('_seed', 1)[0]


def scan_id(plant_id):                                  # 'd1_195039_t4' -> 'd1_195039'
    return re.sub(r'_t\d+$', '', str(plant_id))


def stem_mask(X, h, bin_m=0.05, radius_m=0.035, track_m=0.08):
    """True for points near the stalk axis. The axis is TRACKED from the plant base upward: each
    height bin's centre is the median of the horizontal positions within track_m of the bin below,
    so leaves (which leave the stalk quickly) cannot pull it sideways. X is (n, 3) in the camera
    frame (y down), h the height above the plant base in metres."""
    bins = np.clip((h / bin_m).astype(np.int64), 0, None)
    nb = int(bins.max()) + 1
    centre = np.zeros((nb, 2))
    hpos = X[:, [0, 2]]
    base = bins <= 1                                     # the lowest 2 bins are stalk-dominated
    prev = np.median(hpos[base], axis=0) if base.any() else np.median(hpos, axis=0)
    for b in range(nb):
        sel = bins == b
        near = sel & (np.linalg.norm(hpos - prev, axis=1) < track_m)
        if near.sum() >= 5:
            prev = np.median(hpos[near], axis=0)
        centre[b] = prev
    d = np.linalg.norm(hpos - centre[bins], axis=1)
    return d < radius_m


def count_leaves(X, part=None, eps_mult=4.0, min_pts=30, min_len=0.12, stem_radius=0.035):
    """-> dict(count=..., n_clusters=..., spacing=..., labels=(n,) int, -1 = stem / noise / small).
    X: the target plant's points, camera frame (metres, y down). part: optional per-point part id
    (2 = leaf in the newer uploads); None falls back to the geometric stem fit. Clustering is
    connected components of the radius graph at eps = eps_mult x the median NN spacing (single
    linkage; what DBSCAN reduces to on curve-like leaf surfaces)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    labels = np.full(len(X), -1, np.int64)
    h = X[:, 1].max() - X[:, 1]                                     # height above the base (y is down)
    leaf_sel = (part == 2) if part is not None else ~stem_mask(X, h, radius_m=stem_radius)
    L = X[leaf_sel]
    if len(L) < min_pts:
        return dict(count=0, n_clusters=0, spacing=float('nan'), labels=labels)
    tree = cKDTree(L)
    spacing = float(np.median(tree.query(L, k=2)[0][:, 1]))
    pairs = np.array(list(tree.query_pairs(eps_mult * max(spacing, 1e-4))), np.int64).reshape(-1, 2)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(L), len(L)))
    _, lab = connected_components(g, directed=False)
    keep, nxt = {}, 0
    for c in range(lab.max() + 1):
        pts = L[lab == c]
        if len(pts) < min_pts:
            continue
        extent = float(np.linalg.norm(pts.max(0) - pts.min(0)))     # diagonal of the bounding box
        if extent < min_len:
            continue
        keep[c] = nxt; nxt += 1
    out = np.full(len(L), -1, np.int64)
    for c, k in keep.items():
        out[lab == c] = k
    labels[np.flatnonzero(leaf_sel)] = out
    return dict(count=nxt, n_clusters=int(lab.max() + 1), spacing=spacing, labels=labels)


def save_ply(path, X, labels):
    """Segment-coloured ASCII ply for eyeballing: stem / dropped grey, each kept cluster a colour."""
    rng = np.random.default_rng(0)
    pal = rng.integers(40, 256, (max(int(labels.max()) + 1, 1), 3))
    rgb = np.where(labels[:, None] >= 0, pal[np.clip(labels, 0, None)], 120)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as fh:
        fh.write(f'ply\nformat ascii 1.0\nelement vertex {len(X)}\n'
                 'property float x\nproperty float y\nproperty float z\n'
                 'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        for p, c in zip(X, rgb):
            fh.write(f'{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n')


def iter_stops(layouts, rows, stops):
    """-> (layout, scan, stop, pcd path, target plant number) for the first-upload sorghum rows."""
    for layout in layouts:
        base = SORGHUM_LAYOUTS[layout]
        if not base.exists():
            print(f'[skip] {base} not found (run where /work/mech-ai/alloydas is mounted)')
            continue
        for d in sorted(base.glob('row_*_seed1')):
            rid = row_id(d)
            if rows is not None and rid not in rows:
                continue
            f_csv, p_csv = d / f'row_{rid}_seed1_frames.csv', d / f'row_{rid}_seed1_plants.csv'
            if not (f_csv.exists() and p_csv.exists()):
                continue
            frames = list(csv.DictReader(open(f_csv)))
            plants = list(csv.DictReader(open(p_csv)))
            xs = np.array([float(p['x_m']) for p in plants])
            for f in frames:
                stop = int(f['frame'])
                if stops is not None and stop not in stops:
                    continue
                tp = plants[int(np.argmin(np.abs(xs - float(f['camera_x_m']))))]
                yield layout, rid, stop, d / f['pcd'], int(tp['plant_number'])


def field1000_ref():
    """scan -> leaf count from the field1000 sorghum index (middle plant's CSV rows), if present."""
    idx = FIELD1000 / 'sorghum' / 'index.csv'
    if not idx.exists():
        return {}
    ref = {}
    for m in csv.DictReader(open(idx)):
        if int(m.get('leaves') or -1) >= 0:
            ref.setdefault(m.get('scan') or scan_id(m['plant_id']), []).append(int(m['leaves']))
    return {s: int(np.median(v)) for s, v in ref.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--layouts', nargs='+', default=['parallel', 'perpendicular'])
    ap.add_argument('--rows', nargs='+', default=None, help='scan ids to keep (default: all 30)')
    ap.add_argument('--stops', nargs='+', type=int, default=None, help='stop numbers to keep (default: all 18)')
    ap.add_argument('--eps-mult', type=float, default=4.0, help='linkage radius = this x median NN spacing')
    ap.add_argument('--min-pts', type=int, default=30, help='smallest cluster kept as a leaf')
    ap.add_argument('--min-len', type=float, default=0.12, help='smallest bbox diagonal kept as a leaf (m)')
    ap.add_argument('--stem-radius', type=float, default=0.035, help='geometric stem radius (m), used only without a part field')
    ap.add_argument('--out', default='reports/sorghum_leafcount_rows2026.csv')
    ap.add_argument('--viz-dir', default=None, help='where segment-coloured plys go')
    ap.add_argument('--save-ply', type=int, default=0, help='dump the first N stops as coloured plys')
    ap.add_argument('--ref-field1000', action='store_true', help='print the field1000 CSV count per scan beside ours')
    a = ap.parse_args()

    rows_out, per_scan, saved = [], {}, 0
    for layout, rid, stop, pcd, target in iter_stops(a.layouts, a.rows and set(a.rows), a.stops and set(a.stops)):
        P = read_pcd(pcd)
        sel = P['label'].astype(np.int64) == target
        X = np.stack([P['x'][sel], P['y'][sel], P['z'][sel]], 1).astype(np.float64)
        part = P['part'][sel].astype(np.int64) if 'part' in P.dtype.names else None
        if len(X) < 100:
            continue
        r = count_leaves(X, part, a.eps_mult, a.min_pts, a.min_len, a.stem_radius)
        rows_out.append(dict(layout=layout, scan=rid, stop=stop, n_points=len(X),
                             part_field=int(part is not None), spacing_mm=round(r['spacing'] * 1e3, 2),
                             n_clusters=r['n_clusters'], leaf_count=r['count']))
        per_scan.setdefault(rid, []).append(r['count'])
        if a.save_ply and saved < a.save_ply:
            save_ply(pathlib.Path(a.viz_dir or 'vis_gallery/leafcount') / f'{rid}_{layout}_stop{stop:02d}.ply',
                     X, r['labels'])
            saved += 1
        print(f'{rid} {layout} stop {stop:2d}: {len(X):6d} pts -> {r["count"]} leaves '
              f'({r["n_clusters"]} raw clusters, spacing {r["spacing"]*1e3:.1f} mm)')

    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0])); w.writeheader(); w.writerows(rows_out)
    ref = field1000_ref() if a.ref_field1000 else {}
    print(f'\nwrote {len(rows_out)} stop rows -> {out}\n')
    print(f'{"scan":<16} {"median":>6} {"mode":>5} {"p25-p75":>9} {"stops":>5}' + ('  field1000' if ref else ''))
    for rid in sorted(per_scan):
        c = np.array(per_scan[rid])
        mode = Counter(c.tolist()).most_common(1)[0][0]
        line = (f'{rid:<16} {np.median(c):6.1f} {mode:5d} '
                f'{np.percentile(c, 25):4.0f}-{np.percentile(c, 75):.0f} {len(c):5d}')
        if ref:
            line += f'  {ref.get(rid, "-")}'
        print(line)
    print('\nEstimates only: validate against a manual count (or --ref-field1000) before use in tab:e10.')


if __name__ == '__main__':
    main()
