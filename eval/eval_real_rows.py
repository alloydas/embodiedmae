"""E10: the maize models on the rows built from the 520 real plants (eval/real_rows.py converts each camera stop
into the row model's input conventions). For every stop the model reads the row (photograph, depth, point cloud;
the recipe is never an input) and writes the centre plant: its cloud is scored against the centre plant as the
camera sees it (Chamfer in m^2 and the share of its points missed by more than 5 mm, as eval/eval_row.py), its
recipe on the plant's true height and leaf count (the files carry no per-leaf recipe).
Sources: 'all' = photograph + depth + cloud at the run's mask ratio (the Regime III protocol), or one stream alone.
usage: python eval/eval_real_rows.py --out reports/e10_real_rows_<job> [--models run:ckpt ...] [--sources all rgb depth pc]
                                     [--layouts parallel perpendicular] [--rows 0001 0305] [--stops 9] [--device cuda]
"""
import sys, pathlib; sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[1]), str(pathlib.Path(__file__).resolve().parent)]
import argparse, json, time
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from eval_row import build_model, predict, score_cloud, recipe_scored, bootstrap_ci
from real_rows import RealRowDataset, collate
from embodied_mae_4m_maize import _PLANT_SCALE_V2, _PLANT_SHIFT_V2, PLANT_FIELDS_V2, _LEAF_SCALE_V2, _LEAF_SHIFT_V2, LEAF_FIELDS_V2
from real_rows import LEAF_TRUTH_FIELDS
L_IDX = [LEAF_FIELDS_V2.index(f) for f in LEAF_TRUTH_FIELDS]; A_IDX = LEAF_FIELDS_V2.index('leafAngle')
LSHORT = {'distance': 'internode', 'leafLength': 'leaf_len', 'leafWidth': 'leaf_width', 'avgCurvature': 'curvature'}
import embodied_mae_4m as S4

DEFAULT_MODELS = {'maize': ['maize_row_ft_e1:checkpoints/checkpoint_epoch_100.pth',
                            'maize_row_iso_e1:checkpoints/checkpoint_epoch_100.pth',
                            'maize_4m_pretrain_15k_d2:checkpoints/checkpoint_epoch_1000.pth'],
                  'sorghum': ['sorghum_row_ft:checkpoints/checkpoint_epoch_100.pth',
                              'sorghum_row_iso:checkpoints/checkpoint_epoch_100.pth',
                              '4m_pretrain_15k_d2_sink:checkpoints/checkpoint_epoch_1000.pth']}
ap = argparse.ArgumentParser()
ap.add_argument('--species', default='maize', choices=('maize', 'sorghum'),
                help='sorghum: the 30 scanned plants (eval/real_rows.py); height truth is the plant height, compared '
                     'with the predicted stem length; no leaf-count truth')
ap.add_argument('--models', nargs='+', default=None, help='run:ckpt ... (default per species)')
ap.add_argument('--sources', nargs='+', default=['all', 'rgb', 'depth', 'pc'])
ap.add_argument('--layouts', nargs='+', default=['parallel', 'perpendicular'])
ap.add_argument('--rows', nargs='+', default=None); ap.add_argument('--stops', nargs='+', type=int, default=None)
ap.add_argument('--source', default='rows3000', choices=('rows3000', 'rows2026'),
                help='rows3000: the 3000-row upload of 2026-11-13 (the E10 data); rows2026: the first upload')
ap.add_argument('--plants', nargs='+', default=None, help='rows3000: scan ids to score (e.g. the held-out test plants)')
ap.add_argument('--plants-file', default=None, help='rows3000: CSV with plant,split columns; scores split == test')
ap.add_argument('--batch-size', type=int, default=8); ap.add_argument('--num-workers', type=int, default=8)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
ap.add_argument('--out', required=True, help='prefix: writes <out>.csv and <out>.json')
a = ap.parse_args()
a.models = a.models or DEFAULT_MODELS[a.species]
dev = torch.device(a.device)
if a.species == 'maize':
    H_IDX, N_IDX = PLANT_FIELDS_V2.index('height'), PLANT_FIELDS_V2.index('leafCount')
    PSCALE, PSHIFT = _PLANT_SCALE_V2, _PLANT_SHIFT_V2
else:
    H_IDX, N_IDX = 0, None                           # sorghum plant token: stem_length first; leaf count is not a field
    PSCALE, PSHIFT = S4._PLANT_SCALE, S4._PLANT_SHIFT

models = []
for spec in a.models:
    run, ckpt = spec.split(':', 1)
    m, cfg, ep = build_model(pathlib.Path('outputs') / run, ckpt, dev, a.species)
    models.append(dict(run=run, ckpt=ckpt, epoch=ep, model=m, mask=float(cfg.get('mask_ratio', 0.8)),
                       recipe=recipe_scored(cfg, a.species, m),
                       gain=float(cfg.get('rgb_brighten_gain', 1.0)) if a.species == 'sorghum' else None,
                       # real-row fine-tunes may widen the leaf decoding ranges (train_real_rows.py --leaf-ranges)
                       leaf_sc=np.array(cfg.get('real_finetune', {}).get('leaf_scale', _LEAF_SCALE_V2), np.float32),
                       leaf_sh=np.array(cfg.get('real_finetune', {}).get('leaf_shift', _LEAF_SHIFT_V2), np.float32)))
    print(f'{run} @ {ckpt} (epoch {ep}): mask ratio {models[-1]["mask"]}, recipe scored {models[-1]["recipe"]}', flush=True)

gains = {M['gain'] for M in models}
assert len(gains) == 1, f'models trained at different RGB gains {gains}: score them in separate runs'
if a.plants_file:
    a.plants = [r['plant'] for r in pd.read_csv(a.plants_file, dtype=str).to_dict('records') if r['split'] == 'test']
ds = RealRowDataset(layouts=a.layouts, rows=a.rows, stops=set(a.stops) if a.stops else None, seed=a.seed,
                    species=a.species, rgb_gain=gains.pop(), source=a.source, plants=a.plants)
loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers, collate_fn=collate, persistent_workers=False)
rows, t0 = [], time.time()
for bi, (rgb, depth, pc, params, valid, gt, info) in enumerate(loader):
    rgb, depth, pc, params, gt = (x.to(dev) for x in (rgb, depth, pc, params, gt))
    scale = torch.tensor([i['scale'] for i in info], device=dev)
    for M in models:
        for src in a.sources:
            if dev.type == 'cuda': torch.cuda.synchronize()
            tf = time.time()
            pred_pc, pred_par = predict(M['model'], src, rgb, depth, pc, params, M['mask'] if src == 'all' else 0.0, a.seed + 1000003 * bi)
            if dev.type == 'cuda': torch.cuda.synchronize()
            ms = (time.time() - tf) * 1000 / len(info)
            if pred_pc is None:
                continue
            cd, cd_m2, miss, miss5 = score_cloud(pred_pc.float(), gt, scale)
            if M['recipe']:
                p0 = pred_par[:, 0].clamp(0, 1).float().cpu().numpy()
                h = p0[:, H_IDX] * PSCALE[H_IDX] - PSHIFT[H_IDX]
                n = (np.rint(p0[:, N_IDX] * PSCALE[N_IDX] - PSHIFT[N_IDX]) if N_IDX is not None
                     else np.full(len(info), -1))
                if a.species == 'maize':                     # per-leaf recipe, raw units (leaf tokens 1..28)
                    pl = pred_par[:, 1:29].clamp(0, 1).float().cpu().numpy() * M['leaf_sc'] - M['leaf_sh']
                    isum_pred = p0[:, PLANT_FIELDS_V2.index('stemInternodeSum')] * _PLANT_SCALE_V2[PLANT_FIELDS_V2.index('stemInternodeSum')]
            for k, inf in enumerate(info):
                r = dict(model=M['run'], epoch=M['epoch'], source=src, **inf, chamfer=float(cd[k]), chamfer_m2=float(cd_m2[k]),
                         miss_0p01=float(miss[k]), miss_5mm=float(miss5[k]), ms_per_stop=ms)
                if M['recipe']:
                    r.update(height_pred=float(h[k]), leaves_pred=int(n[k]), height_rel_err=abs(float(h[k]) - inf['height']) / inf['height'],
                             leaves_abs_err=(abs(int(n[k]) - inf['leaves']) if inf['leaves'] >= 0 and n[k] >= 0 else np.nan))
                if M['recipe'] and a.species == 'maize' and inf.get('leaf_raw') is not None and len(inf['leaf_raw']):
                    tr = inf['leaf_raw']; k_ = min(len(tr), 28)            # leaves matched by index from the base
                    for j, f in enumerate(LEAF_TRUTH_FIELDS):
                        r[f'{LSHORT[f]}_mae'] = float(np.abs(pl[k, :k_, L_IDX[j]] - tr[:k_, j]).mean())
                        r[f'{LSHORT[f]}_true_mean'] = float(tr[:, j].mean()); r[f'{LSHORT[f]}_pred_mean'] = float(pl[k, :k_, L_IDX[j]].mean())
                    r['internode_sum_true'] = float(tr[:, 0].sum()); r['internode_sum_pred'] = float(isum_pred[k])
                    r['internode_sum_rel_err'] = abs(r['internode_sum_pred'] - r['internode_sum_true']) / max(1e-6, r['internode_sum_true'])
                    r['_angle_pairs'] = (pl[k, :k_, A_IDX].tolist(), inf['leaf_zenith'][:k_].tolist())
                rows.append(r)
    if bi % 20 == 0:
        print(f'  {min((bi + 1) * a.batch_size, len(ds))}/{len(ds)} stops ({time.time() - t0:.0f} s)', flush=True)

for r in rows:                                       # info arrays and angle pairs stay out of the CSV
    r.pop('leaf_raw', None); r.pop('leaf_zenith', None)
angle_pairs = {}
for r in rows:
    if '_angle_pairs' in r:
        p_, z_ = r.pop('_angle_pairs'); angle_pairs.setdefault((r['model'], r['source']), []).extend(zip(p_, z_))
df = pd.DataFrame(rows); df.to_csv(f'{a.out}.csv', index=False)
summary = {'source': a.source, 'n_stops': len(ds), 'rows': int(df[['layout', 'row']].drop_duplicates().shape[0]) if len(df) else 0,
           'plants': int(df['plant'].nunique()) if len(df) else 0, 'groups': []}
for (model, src, layout), g in list(df.groupby(['model', 'source', 'layout'])) + [((m, s, 'both'), g) for (m, s), g in df.groupby(['model', 'source'])]:
    per_plant = g.groupby('plant')                  # rows2026: one plant per row; rows3000: the scan
    e = dict(model=model, source=src, layout=layout, n_stops=len(g), n_plants=per_plant.ngroups)
    for col in ('chamfer_m2', 'miss_5mm'):
        mean, lo, hi = bootstrap_ci(per_plant[col].mean().values)
        e[col] = dict(mean=mean, lo=lo, hi=hi, median=float(g[col].median()))
    if 'height_rel_err' in g and g['height_rel_err'].notna().any():
        e['height_rel_err_median'] = float(g['height_rel_err'].median())
        e['height_mae_m'] = float((g['height_pred'] - g['height']).abs().mean())
        if g['leaves_abs_err'].notna().any():
            e['leaves_acc_pm1'] = float((g['leaves_abs_err'] <= 1).mean())
            e['leaves_acc_exact'] = float((g['leaves_abs_err'] == 0).mean())
            e['leaves_mae'] = float(g['leaves_abs_err'].mean())
    for f in LSHORT.values():                        # maize rows3000: per-leaf recipe vs the scanned leaves
        if f'{f}_mae' in g and g[f'{f}_mae'].notna().any():
            e[f'{f}_mae'] = float(g[f'{f}_mae'].mean()); e[f'{f}_true_mean'] = float(g[f'{f}_true_mean'].mean())
            e[f'{f}_pred_mean'] = float(g[f'{f}_pred_mean'].mean())
            e[f'{f}_mae_ci'] = bootstrap_ci(per_plant[f'{f}_mae'].mean().dropna().values)
    if 'internode_sum_rel_err' in g and g['internode_sum_rel_err'].notna().any():
        e['internode_sum_rel_err_median'] = float(g['internode_sum_rel_err'].median())
    if layout == 'both' and (model, src) in angle_pairs and len(angle_pairs[(model, src)]) > 2:
        pa = np.array(angle_pairs[(model, src)])
        e['leafAngle_vs_zenith_pearson_r'] = float(np.corrcoef(pa[:, 0], pa[:, 1])[0, 1])    # exploratory: not one quantity
    e['occ_frac_mean'] = float(g['occ_frac'].mean()); e['ms_per_stop'] = float(g['ms_per_stop'].mean())
    summary['groups'].append(e)
json.dump(summary, open(f'{a.out}.json', 'w'), indent=1)
print(f'wrote {a.out}.csv ({len(df)} rows) and {a.out}.json')
for e in summary['groups']:
    if e['layout'] == 'both':
        print(f"{e['model']:26s} {e['source']:5s} chamfer {e['chamfer_m2']['mean'] * 1e4:6.1f}e-4 m2 [{e['chamfer_m2']['lo'] * 1e4:.1f}, {e['chamfer_m2']['hi'] * 1e4:.1f}]"
              f"  miss5mm {e['miss_5mm']['mean']:.2f}" + (f"  height err {e['height_rel_err_median'] * 100:.1f} %  leaves +-1 {e['leaves_acc_pm1']:.2f}" if 'leaves_acc_pm1' in e else '')
              + (f"  leaf MAE len {e['leaf_len_mae'] * 100:.1f} cm width {e['leaf_width_mae'] * 100:.1f} cm internode {e['internode_mae'] * 100:.1f} cm curv {e['curvature_mae']:.2f}/m" if 'leaf_len_mae' in e else '')
              + f"  {e['ms_per_stop']:.1f} ms/stop")
