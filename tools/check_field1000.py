"""Check the field1000 upload (2026-11-15) and its loader (eval/real_rows.py source='field1000').

Builds <FIELD1000>/<species>/index.csv (one pass over the PCD headers), summarises rows / stops / plants / scans / leaf
counts, then runs RealRowDataset on 60 random stops per species and reports: load errors, the window's plant count, the
hidden fraction, the scene / target point counts, and (sorghum) the middle plant's height and leaf count against
design/field_design_sorghum_3000.csv.
  sbatch --wait --partition=nova --account=mech-ai --qos=normal --cpus-per-task=16 --mem=32G --output=logs/%x_%j.out \
         --wrap "python tools/check_field1000.py"
"""
import sys, pathlib, csv, collections
R = pathlib.Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R), str(R / 'eval')]
import numpy as np
import real_rows as RR

if __name__ == '__main__':
    design = {r['plant_id']: r for r in csv.DictReader(open(RR.FIELD_DESIGN['sorghum']))}
    for sp in ('maize', 'sorghum'):
        RR.build_index_3000(sp, source='field1000')
        idx = RR.load_index_3000(sp, 'field1000')
        lv = np.array([int(m['leaves']) for m in idx])
        print(f"\n=== {sp}: {len(idx)} stops, {len({m['row'] for m in idx})} rows, {len({m['plant_id'] for m in idx})} middle plants, "
              f"{len({m['scan'] for m in idx})} scans; png ok {sum(int(m['png_ok']) for m in idx)}; "
              f"middle-plant leaves per stop min/median/max {lv.min()}/{int(np.median(lv))}/{lv.max()}; "
              f"hfov {sorted({round(float(m['hfov_deg']), 1) for m in idx})[:3]}..; image sizes {len({m['image_px'] for m in idx})} distinct", flush=True)
        ds = RR.RealRowDataset(species=sp, source='field1000', rgb_gain=1.85 if sp == 'sorghum' else None)
        rng = np.random.default_rng(0); pick = rng.choice(len(ds), 60, replace=False)
        rows, errs = [], []
        for i in pick:
            try:
                rgb, dep, pin, par, valid, pgt, info = ds[int(i)]
            except Exception as e:
                errs.append((ds.items[int(i)][4]['name'], repr(e))); continue
            m = ds.items[int(i)][4]; d = design.get(m['plant_id']) if sp == 'sorghum' else None
            rows.append(dict(h=info['height'], n=info['leaves'], occ=info['occ_frac'], pw=info['plants_in_window'], nt=info['n_target_visible'],
                             ns=info['n_scene_visible'], fg=float((dep > 0).float().mean()), nleaf_csv=(len(info['leaf_raw']) if info['leaf_raw'] is not None else -1),
                             dh=(info['height'] - float(d['height_m'])) if d else np.nan, dn=(info['leaves'] - int(d['n_leaves'])) if d else np.nan))
        a = {k: np.array([r[k] for r in rows], float) for k in rows[0]}
        print(f"  loaded {len(rows)}/60, errors {len(errs)} {errs[:3]}")
        for k, lab in (('h', 'height m'), ('n', 'leaves'), ('occ', 'hidden fraction'), ('pw', 'plants in window'), ('nt', 'target visible pts'),
                       ('ns', 'scene visible pts'), ('fg', 'depth fg fraction')):
            print(f"  {lab:20s} min {a[k].min():.3f}  median {np.median(a[k]):.3f}  max {a[k].max():.3f}")
        print(f"  leaf CSV rows == leaf count on {int((a['nleaf_csv'] == a['n']).sum())}/{len(rows)} stops"); ls_ = np.array([float(m.get('len_scale') or 1) for m in idx]); print(f'  len_scale != 1 on {int((ls_ != 1).sum())} stops ({len({m["plant_id"] for m in idx if float(m.get("len_scale") or 1) != 1})} plants), range {ls_.min():.3f}-{ls_.max():.3f}')
        if sp == 'sorghum':
            print(f"  vs design: height diff median {np.nanmedian(a['dh']) * 100:.1f} cm, max |diff| {np.nanmax(np.abs(a['dh'])) * 100:.1f} cm; "
                  f"leaf count equal on {int((a['dn'] == 0).sum())}/{len(rows)}")
