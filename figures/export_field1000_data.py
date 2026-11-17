"""Latest field data (Maize_Field_1000 / Sorghum_Field_1000, the 2026-11-15 upload) for the user (2026-11-16: "Did you
shared the latest data with me in github?", "Yes build it and send to me").

One camera stop per HELD-OUT TEST plant (maize 130, sorghum 13 scans; data_split/e10_real_split{,_sorghum}_field1000.csv):
the first row in which the plant is the middle plant, its middle stop. Per stop folder <species>/<plant>_row<RR>_stop<SS>/:
  photo.jpg                          the stop's photograph, reduced to 1,200 px high (the original is in the upload)
  leaves.csv                         the stop's per-leaf CSV, middle plant only (blank duplicate rows dropped)
  truth.json / .yml / .xml           height (m, vertical extent of the middle plant's points), leaf count (= leaf rows),
                                     every leaf's CSV values; sorghum original scans: leaf_length_m rescaled (len_scale)
  target_point_cloud_cam.ply         the middle plant's points of the stop's PCD (OpenCV camera frame, metres)
  model_input_image.png / _depth.png the 224 px 'plant' window the models receive (image at the model's colour gain;
                                     depth shown 8-bit, near = bright), model_input_scene_cloud_cam.ply (24,576 visible
                                     scene points, OpenCV camera frame, metres)
  predicted_<model>.{xml,yml}        the procedural model written from all three streams (recipe never an input)
  predicted_<model>_point_cloud_cam.ply   the predicted target-plant cloud, OpenCV camera frame, metres
  models: synthetic_row_finetune (before: trained on simulated rows only) and field_finetune (after: + 30 epochs on the
  training plants of this upload, best_model.pth). Predictions are one masked draw (mask ratio 0.8, seed = stop index),
  so per-stop numbers differ slightly from the evaluation reports, which average over every stop.
summary.csv: per stop and model, Chamfer to the middle plant (1e-4 m^2), height and leaf count written vs true.
Sorghum leaf count: the sorghum recipe has no leaf-count field; the count written is the generator's rule from the stem
length (0.05 m per leaf, capped at the 24 leaf slots), which does not hold for real plants (see README).
  sbatch --wait --partition=nova --account=mech-ai --qos=normal --cpus-per-task=8 --mem=32G --time=02:00:00 \
         --output=logs/%x_%j.out --wrap "python figures/export_field1000_data.py"
"""
import sys, pathlib, csv, json, shutil, zipfile
R = pathlib.Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R), str(R / 'figures'), str(R / 'eval')]
import numpy as np, torch
from PIL import Image
import real_rows as RR
from real_rows import RealRowDataset
from eval_row import build_model, predict, score_cloud
import embodied_mae_4m as S4
from embodied_mae_4m_maize import _PLANT_SCALE_V2, _PLANT_SHIFT_V2, PLANT_FIELDS_V2, LEAF_FIELDS_V2, _LEAF_SCALE_V2, _LEAF_SHIFT_V2
import _plantviz as V
import export_procedural_models as EP
import procedural_formats as PF

OUT = pathlib.Path('/work/mech-ai/alloydas/embodiedmae_outputs/field1000_data')
MODELS = {'maize': [('synthetic_row_finetune', 'maize_row_ft_e1', 'checkpoints/checkpoint_epoch_100.pth'),
                    ('field_finetune', 'maize_real_ft_f1000', 'best_model.pth')],
          'sorghum': [('synthetic_row_finetune', 'sorghum_row_ft', 'checkpoints/checkpoint_epoch_100.pth'),
                      ('field_finetune', 'sorghum_real_ft_f1000', 'best_model.pth')]}
SPLIT = {'maize': R / 'data_split/e10_real_split_field1000.csv', 'sorghum': R / 'data_split/e10_real_split_sorghum_field1000.csv'}
GL2CV = np.array([1.0, -1.0, -1.0])
IMNET_M, IMNET_S = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])

_sub = RR.subsample; CAPT = []
def _capture(pts, k, rng):                      # records the scene subsample, whose mean is the cloud's centre
    out = _sub(pts, k, rng); CAPT.append(out); return out
RR.subsample = _capture


def write_ply(P, path):
    P = np.asarray(P, np.float32)
    with open(path, 'wb') as fh:
        fh.write(f'ply\nformat binary_little_endian 1.0\nelement vertex {len(P)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n'.encode())
        fh.write(P.astype('<f4').tobytes())


def decode(sp, par, M):
    par = np.clip(np.asarray(par, np.float64), 0, 1)
    if sp == 'sorghum':
        return V.decode_recipe('sorghum', par, None, predicted=True)
    p = par[0] * _PLANT_SCALE_V2 - _PLANT_SHIFT_V2
    n = int(np.clip(np.rint(p[PLANT_FIELDS_V2.index('leafCount')]), 0, 28)); nf = len(LEAF_FIELDS_V2)
    raw = par[1:1 + n, :nf] * M['leaf_sc'][:nf] - M['leaf_sh'][:nf]
    return dict(height=float(p[PLANT_FIELDS_V2.index('height')]), n=n, leaves=[dict(zip(LEAF_FIELDS_V2, r)) for r in raw])


def pick_items(ds, test):
    """one stop per test plant: the first row with the plant in the middle, its middle stop"""
    by = {}
    for i, it in enumerate(ds.items):
        m = it[4]; sc = m.get('scan') or RR.scan_id(m['plant_id'])
        if sc in test: by.setdefault(sc, {}).setdefault(it[2], []).append((it[3], i))
    out = []
    for sc in sorted(by):
        row = sorted(by[sc])[0]; st = sorted(by[sc][row]); out.append((sc, st[len(st) // 2][1]))
    return out


if __name__ == '__main__':
    torch.set_num_threads(8); fit = EP.fits(); summary = []
    if OUT.exists(): shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for sp in ('maize', 'sorghum'):
        test = {r['plant'] for r in csv.DictReader(open(SPLIT[sp])) if r['split'] == 'test'}
        Ms = []
        for name, run, ck in MODELS[sp]:
            m, cfg, ep = build_model(R / 'outputs' / run, ck, 'cpu', sp)
            Ms.append(dict(name=name, run=run, ckpt=ck, epoch=ep, model=m, mask=float(cfg.get('mask_ratio', 0.8)),
                           gain=float(cfg.get('rgb_brighten_gain', 1.0)) if sp == 'sorghum' else None,
                           leaf_sc=np.array(cfg.get('real_finetune', {}).get('leaf_scale', _LEAF_SCALE_V2), np.float64),
                           leaf_sh=np.array(cfg.get('real_finetune', {}).get('leaf_shift', _LEAF_SHIFT_V2), np.float64)))
            print(sp, name, run, ck, 'epoch', ep, flush=True)
        gains = {M['gain'] for M in Ms}; assert len(gains) == 1, gains
        ds = RealRowDataset(species=sp, rgb_gain=gains.pop(), source='field1000', plants=sorted(test))
        base = RR.ROWS_ROOT['field1000'] / sp
        for sc, idx in pick_items(ds, test):
            m = ds.items[idx][4]; name = m['name']
            CAPT.clear(); rgb, dep, pin, par, valid, pgt, inf = ds[idx]
            centre = CAPT[0].mean(0); scale = float(inf['scale'])
            d = OUT / sp / f"{sc}_row{int(m['row']):02d}_stop{int(m['stop']):02d}"; d.mkdir(parents=True)
            # the stop's data
            im = Image.open(base / 'png' / f'{name}.png').convert('RGB'); im.thumbnail((10_000, 1200)); im.save(d / 'photo.jpg', quality=90)
            rows = RR.middle_leaf_rows(base / 'csv' / f'{name}.csv')
            with open(d / 'leaves.csv', 'w', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
            ls = float(m.get('len_scale') or 1.0); leaves_csv = []
            for r in rows:
                e = {k: (float(v) if k not in ('photo', 'plant_id') and v not in ('', None) else v) for k, v in r.items() if k not in ('photo', 'is_middle_plant')}
                if ls != 1.0: e['leaf_length_m_raw_csv'] = e['leaf_length_m']; e['leaf_length_m'] = e['leaf_length_m'] * ls
                leaves_csv.append(e)
            truth = dict(species=sp, plant_id=m['plant_id'], scan=sc, photo=name, height_m=float(inf['height']), leaf_count=len(rows),
                         occluded_fraction=float(inf['occ_frac']), len_scale_applied=ls, leaves_csv=leaves_csv)
            (d / 'truth.json').write_text(json.dumps(truth, indent=1))
            P = RR.read_pcd(base / 'pcd' / f'{name}.pcd'); lab = P['label'].astype(int)
            X = np.stack([P['x'], P['y'], P['z']], 1)[lab == int(inf['target'])]
            write_ply(X, d / 'target_point_cloud_cam.ply')
            write_ply((pin.numpy() * scale + centre) * GL2CV, d / 'model_input_scene_cloud_cam.ply')
            img = (rgb.numpy().transpose(1, 2, 0) * IMNET_S + IMNET_M).clip(0, 1)
            Image.fromarray((img * 255).astype(np.uint8)).save(d / 'model_input_image.png')
            dd = dep[0].numpy(); fg = dd > 0; vis = np.zeros_like(dd)
            if fg.any(): lo, hi = dd[fg].min(), dd[fg].max(); vis[fg] = 1 - 0.85 * (dd[fg] - lo) / max(hi - lo, 1e-9)
            Image.fromarray((vis * 255).astype(np.uint8)).save(d / 'model_input_depth.png')
            # predictions
            b = lambda x: x[None]
            for M in Ms:
                with torch.no_grad():
                    pp, ppar = predict(M['model'], 'all', b(rgb), b(dep), b(pin), b(par), M['mask'], idx)
                _, cd_m2, _, miss5 = score_cloud(pp.float(), b(pgt), torch.tensor([scale]))
                rec = decode(sp, ppar[0].numpy(), M)
                note = (f"field1000 test: {sp} {sc} ({name}) - {M['name']} ({M['run']}, epoch {M['epoch']}), all three inputs; "
                        + ('leaf count = stem-length rule of the generator (0.05 m per leaf, max 24), not a learned count'
                           if sp == 'sorghum' else 'leaf count read from the plant token'))
                stem = d / f"predicted_{M['name']}"
                if sp == 'sorghum': EP.write_sorghum(rec, stem.with_suffix('.yml'), note)
                else: EP.write_maize(rec, stem.with_suffix('.xml'), note, fit)
                write_ply((pp[0].numpy() * scale + centre) * GL2CV, d / f"predicted_{M['name']}_point_cloud_cam.ply")
                summary.append(dict(species=sp, plant=sc, stop_folder=d.name, photo=name, model=M['name'], run=M['run'], epoch=M['epoch'],
                                    chamfer_1e4_m2=float(cd_m2[0]) * 1e4, miss_5mm=float(miss5[0]), height_true=truth['height_m'],
                                    height_pred=rec['height'], height_rel_err=abs(rec['height'] - truth['height_m']) / truth['height_m'],
                                    leaves_true=truth['leaf_count'], leaves_pred=rec['n']))
            s = summary[-2:]
            print(sp, sc, name, 'true h %.2f n %d' % (truth['height_m'], truth['leaf_count']),
                  [(r['model'][:5], round(r['chamfer_1e4_m2'], 1), round(r['height_pred'], 2), r['leaves_pred']) for r in s], flush=True)
    PF.convert(OUT)
    with open(OUT / 'summary.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    (OUT / 'README.md').write_text(open(R / 'figures/field1000_README.md').read())
    for z in OUT.parent.glob('field1000_data_*.zip'): z.unlink()
    for sp_dir in sorted(p for p in OUT.iterdir() if p.is_dir()):
        groups, cur, size = [], [], 0
        for p in sorted(sp_dir.iterdir()):
            s_ = sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
            if cur and size + s_ > 45e6: groups.append(cur); cur, size = [], 0
            cur.append(p); size += s_
        if cur: groups.append(cur)
        for gi, g in enumerate(groups, 1):
            zp = OUT.parent / f"field1000_data_{sp_dir.name}{'' if len(groups) == 1 else f'_{gi}'}.zip"
            with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as zf:
                for f in ('README.md', 'summary.csv'): zf.write(OUT / f, f'field1000_data/{f}')
                for p in g:
                    for f in sorted(p.rglob('*')):
                        if f.is_file(): zf.write(f, 'field1000_data/' + str(f.relative_to(OUT)))
            print('wrote', zp.name, zp.stat().st_size // 1_000_000, 'MB', len(g), 'stops', flush=True)
