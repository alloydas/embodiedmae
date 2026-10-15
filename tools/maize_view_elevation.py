#!/usr/bin/env python3
"""Per-view camera elevation table for maize: the input of the maize E5 view regime.

Sorghum's ten views are an exact elevation ladder (sin elev = 0.9 - 0.2 i), so its
E5 arms restrict the train draw to a band of VIEW INDICES (configs/config_e5_*_d2.yaml,
SorghumDataset4M.train_views). Maize's ten views are not: measured over the train
split, each view index spans elevations of about -85 to +81 degrees with a random
azimuth, so view 3 of one plant looks down at 70 degrees and view 3 of the next looks
up at 75. A band of indices would be a random subset of a random hemisphere.

The maize regime is therefore defined per PLANT from the actual camera: this table
holds, for every view folder, the camera's elevation above the plant centre (world
+Y up: elev > 0 looks down at the plant), its azimuth and its distance, read from
camera_pose.json ("position" and "plantCenter"). MaizeDataset4M.train_views ranks
a plant's ten views by sin(elev) from this table ('top:2' = the two most overhead
cameras, 'horizon:2' = the two nearest the horizon, 'bottom:2' = the two most
underneath), so every plant keeps exactly k views and the plant set, the steps per
epoch and the schedule are unchanged -- the E3 confound sorghum's design avoids.

    python tools/maize_view_elevation.py --data-root /work/hdd/bepk/adas14/Maize \
        --out data_split/maize_view_elevation.csv.gz --workers 16

Columns: split, plant, view, sin_elev, elev_deg, az_deg, dist. 150,000 rows (10 per
plant, every split). The file is read by MaizeDataset4M when train_views is a rank
spec, and by eval/pc_view_eval_maize.py for its coverage report.
"""
import argparse
import csv
import gzip
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

VIEWS = tuple(range(10))


def plants_of(root, split):
    f = Path(root) / f'plants_{split}.txt'
    return [l.strip() for l in f.read_text().splitlines() if l.strip()]


def one_plant(job):
    root, split, plant = job
    rows, err = [], None
    for v in VIEWS:
        p = Path(root) / split / f'{plant}_{v:02d}' / 'camera_pose.json'
        try:
            with open(p) as fh:
                m = json.load(fh)
            rel = [a - b for a, b in zip(m['position'], m['plantCenter'])]
            d = math.sqrt(sum(x * x for x in rel))
            s = rel[1] / d
            rows.append((split, plant, v, s, math.degrees(math.asin(max(-1.0, min(1.0, s)))),
                         math.degrees(math.atan2(rel[0], rel[2])), d))
        except Exception as e:      # reported, never silently dropped
            err = f'{p}: {e!r}'
            break
    return plant, rows, err


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--out', default=str(Path(__file__).resolve().parent.parent
                                         / 'data_split' / 'maize_view_elevation.csv.gz'))
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--splits', default='train,val,test')
    ap.add_argument('--limit', type=int, default=None, help='plants per split (smoke)')
    args = ap.parse_args()

    jobs = []
    for split in args.splits.split(','):
        plants = plants_of(args.data_root, split)
        if args.limit:
            plants = plants[:args.limit]
        jobs += [(args.data_root, split, p) for p in plants]
    print(f'{len(jobs)} plants x {len(VIEWS)} views, {args.workers} workers', flush=True)

    rows, bad, t0 = [], [], time.time()
    with Pool(args.workers) as pool:
        for i, (plant, r, err) in enumerate(pool.imap_unordered(one_plant, jobs, chunksize=64)):
            if err:
                bad.append(err)
            else:
                rows.extend(r)
            if (i + 1) % 2000 == 0:
                print(f'  {i + 1}/{len(jobs)} plants ({time.time() - t0:.0f}s)', flush=True)
    if bad:
        print(f'FAILED on {len(bad)} plants, e.g. {bad[0]}', file=sys.stderr)
        sys.exit(1)
    rows.sort(key=lambda r: (('train', 'val', 'test').index(r[0]), r[1], r[2]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f'.tmp{os.getpid()}')
    opener = gzip.open if out.suffix == '.gz' else open
    with opener(tmp, 'wt', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['split', 'plant', 'view', 'sin_elev', 'elev_deg', 'az_deg', 'dist'])
        for r in rows:
            w.writerow([r[0], r[1], r[2], f'{r[3]:.6f}', f'{r[4]:.3f}', f'{r[5]:.3f}', f'{r[6]:.4f}'])
    os.replace(tmp, out)
    print(f'📄 {out}  ({len(rows)} rows, {time.time() - t0:.0f}s)')

    # What the table says: no index is an elevation; the ranked views are.
    import numpy as np
    by_view = {v: np.array([r[4] for r in rows if r[2] == v]) for v in VIEWS}
    print('elevation (deg) per view index, all splits:')
    for v in VIEWS:
        e = by_view[v]
        print(f'  view {v}: mean {e.mean():6.1f}  sd {e.std():5.1f}  min {e.min():6.1f}  max {e.max():6.1f}')
    per_plant = {}
    for r in rows:
        per_plant.setdefault((r[0], r[1]), []).append((r[2], r[3]))
    top2, hor2 = [], []
    for views in per_plant.values():
        s = sorted(views, key=lambda t: -t[1])
        top2.append([math.degrees(math.asin(t[1])) for t in s[:2]])
        h = sorted(views, key=lambda t: abs(t[1]))
        hor2.append([math.degrees(math.asin(t[1])) for t in h[:2]])
    top2, hor2 = np.array(top2), np.array(hor2)
    print(f"'top:2'     realised elevation: 1st {np.median(top2[:, 0]):.1f} (p10 {np.percentile(top2[:, 0], 10):.1f}), "
          f"2nd {np.median(top2[:, 1]):.1f} (p10 {np.percentile(top2[:, 1], 10):.1f}) deg")
    print(f"'horizon:2' realised |elevation|: 1st {np.median(np.abs(hor2[:, 0])):.1f} (p90 {np.percentile(np.abs(hor2[:, 0]), 90):.1f}), "
          f"2nd {np.median(np.abs(hor2[:, 1])):.1f} (p90 {np.percentile(np.abs(hor2[:, 1]), 90):.1f}) deg")


if __name__ == '__main__':
    main()
