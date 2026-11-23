"""Real-plant figures and the project page's real-plant section rebuilt on the FIELD data (user, 2026-11-20: use the
data section's real-plant photos in 'From synthetic rows to real plants' and in S4, S5, S10, S11 -> "Switch to field
data").

Data: the 2026-11-15 field upload (Maize / Sorghum_Field_1000), held-out TEST plants only (data_split/
e10_real_split*_field1000.csv). Models per species:
  head    the headline model, no row training          maize_4m_pretrain_15k_d2 / 4m_pretrain_15k_d2_sink, epoch 1000
  row     the row fine-tune on synthetic rows           maize_row_ft_e1 / sorghum_row_ft, epoch 100
  field   the row fine-tune further trained on the field data's training plants
          maize_real_ft_f1000 (best); sorghum_real_ft_f1000_count2 (best; the leaf-slot run that writes a leaf count)
Stops are picked by rule from the field model's test report (all three streams in):
  S4 / S5 (e10)      the stops nearest the median Chamfer, distinct plants: maize 4, sorghum 2
  S10 (maize)        12 stops on distinct plants at the 5th..96th percentile of Chamfer, two blocks of six
  S11 (sorghum)      8 of the 13 test scans, evenly spaced in their median-stop Chamfer; each scan's median stop
  page               the S10 / S11 stops plus the S4 / S5 ones
Per stop: the original whole-row photograph (as the data section), the model inputs (window depth, scene cloud), the
middle plant's cloud and measured height / leaf count, and per model the predicted cloud, Chamfer (1e-4 m^2), the written
procedural model (figures/_procfig.decode_pred) with its height and leaf count. One masked draw per model (seed by stop),
so a stop's numbers can differ slightly from the report's.
Writes /work/.../_fig12/supp_field_{gallery,real}_{sp}.pkl (for figures/make_supp_tikz.py field builders) and replaces the
page's site/data/real/<sp>/ items.
  sbatch --wait --partition=nova --account=mech-ai --qos=normal --cpus-per-task=8 --mem=48G --time=02:00:00 \
         --output=logs/%x_%j.out --wrap "python figures/make_field_dump.py"
"""
import sys, pathlib, csv, json, pickle
R = pathlib.Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R), str(R / 'figures'), str(R / 'eval'), str(R / 'tools')]
sys.argv = [sys.argv[0], '--tasks', 'none']                          # export_site_data parses its arguments at import
import numpy as np, torch
from PIL import Image
from eval_row import build_model, predict
from real_rows import RealRowDataset, ROWS_ROOT
import embodied_mae_4m as S4
from embodied_mae_4m_maize import _PLANT_SCALE_V2, _PLANT_SHIFT_V2, PLANT_FIELDS_V2
import _procfig as PF
import export_site_data as ESD

torch.set_num_threads(8)
OUT12 = pathlib.Path('/work/mech-ai/alloydas/embodiedmae_outputs/_fig12')
REP = {'maize': ('reports/e10_real_ft_f1000_test_17491823.csv', 'maize_real_ft_f1000'),
       'sorghum': ('reports/e10_real_ft_sorghum_f1000_count2_test_17588524.csv', 'sorghum_real_ft_f1000_count2')}
MODELS = {'maize': [('head', 'maize_4m_pretrain_15k_d2', 'checkpoints/checkpoint_epoch_1000.pth'),
                    ('row', 'maize_row_ft_e1', 'checkpoints/checkpoint_epoch_100.pth'), ('field', 'maize_real_ft_f1000', 'best_model.pth')],
          'sorghum': [('head', '4m_pretrain_15k_d2_sink', 'checkpoints/checkpoint_epoch_1000.pth'),
                      ('row', 'sorghum_row_ft', 'checkpoints/checkpoint_epoch_100.pth'), ('field', 'sorghum_real_ft_f1000_count2', 'best_model.pth')]}
LABEL = {'head': 'No row training', 'row': 'Synthetic-row fine-tune', 'field': 'Field fine-tune'}


def picks(sp):
    path, model = REP[sp]
    rows = [r for r in csv.DictReader(open(R / path)) if r['model'] == model and r['source'] == 'all']
    for r in rows: r['ch'] = float(r['chamfer_m2'])
    srt = sorted(rows, key=lambda r: r['ch']); med = srt[len(srt) // 2]['ch']
    e10, used = [], set()
    for r in sorted(rows, key=lambda r: abs(r['ch'] - med)):
        if r['plant'] in used: continue
        e10.append((r, 'median stop')); used.add(r['plant'])
        if len(e10) == (4 if sp == 'maize' else 2): break
    gal = []
    if sp == 'maize':
        P = [5, 13, 21, 29, 38, 46, 54, 63, 71, 79, 88, 96]; used = set()
        for q in P:
            k = int(round(q / 100 * (len(srt) - 1)))
            while srt[k]['plant'] in used: k += 1
            gal.append((srt[k], f'{q}th', 'A' if q < 50 else 'B')); used.add(srt[k]['plant'])
    else:
        per = {}
        for r in rows: per.setdefault(r['plant'], []).append(r)
        med_stop = sorted((sorted(v, key=lambda r: r['ch'])[len(v) // 2] for v in per.values()), key=lambda r: r['ch'])
        for k in np.round(np.linspace(0, len(med_stop) - 1, 8)).astype(int):
            gal.append((med_stop[k], f'{int(round(100 * k / (len(med_stop) - 1)))}th', 'A'))
    return e10, gal


def run(sp):
    Ms = []
    for key, run_, ck in MODELS[sp]:
        m, cfg, ep = build_model(R / 'outputs' / run_, ck, torch.device('cpu'), sp)
        Ms.append(dict(key=key, run=run_, model=m, cfg=cfg, mask=float(cfg.get('mask_ratio', 0.8))))
    gain = float(Ms[-1]['cfg'].get('rgb_brighten_gain', 1.0)) if sp == 'sorghum' else None
    e10, gal = picks(sp); cache = {}

    def stop(r, i):
        key = (r['plant'], int(r['row']), int(r['stop']))
        if key in cache: return cache[key]
        ds = RealRowDataset(species=sp, source='field1000', plants=[r['plant']], rgb_gain=gain)
        j = next(j for j, it in enumerate(ds.items) if int(it[2]) == int(r['row']) and int(it[3]) == int(r['stop']))
        rgb, dep, pin, par, valid, pgt, info = ds[j]; name = ds.items[j][4]['name']
        photo = Image.open(ROWS_ROOT['field1000'] / sp / 'png' / f'{name}.png').convert('RGB'); s = 900 / photo.height
        photo = photo.resize((round(photo.width * s), 900), Image.LANCZOS)
        out = dict(plant=r['plant'], row=int(r['row']), stop=int(r['stop']), name=name, photo=np.asarray(photo) / 255.0,
                   depth=np.where(dep[0].numpy() > 0, dep[0].numpy(), np.nan), pin=pin.numpy(), pgt=pgt.numpy(),
                   scale=float(info['scale']), height=float(info['height']), leaves=int(info['leaves']), occ=float(info['occ_frac']), pred={})
        for M in Ms:
            with torch.no_grad():
                pp, ppar = predict(M['model'], 'all', rgb[None], dep[None], pin[None], par[None], M['mask'], 1000003 * i)
            P = pp[0].float().numpy(); rec = PF.decode_pred(sp, ppar[0].float().numpy(), M['cfg'])
            ch = ESD.chamfer(P, out['pgt']) * out['scale'] ** 2 * 1e4
            out['pred'][M['key']] = dict(P=P, rec=rec, ch=ch, height=float(rec['height']), leaves=int(rec['n']))
        cache[key] = out
        print(sp, name, r['plant'], 'true', round(out['height'], 2), out['leaves'],
              {k: (round(v['ch'], 1), round(v['height'], 2), v['leaves']) for k, v in out['pred'].items()}, flush=True)
        return out

    # S10 / S11 gallery: row and field models
    G = []
    for i, (r, lab, blk) in enumerate(gal):
        o = stop(r, i)
        G.append(dict(block=blk, label=lab, img=o['photo'], gt=o['pgt'], height=o['height'], leaves=o['leaves'], plant=o['plant'],
                      ch=[o['pred'][k]['ch'] for k in ('row', 'field')], hp=[o['pred'][k]['height'] for k in ('row', 'field')],
                      lv=[o['pred'][k]['leaves'] for k in ('row', 'field')], recs=[o['pred'][k]['rec'] for k in ('row', 'field')]))
    pickle.dump(G, open(OUT12 / f'supp_field_gallery_{sp}.pkl', 'wb'))
    # S4 / S5: all three models
    E = []
    for i, (r, lab) in enumerate(e10):
        o = stop(r, 100 + i)
        E.append(dict(label=(o['plant'], o['row'], o['stop']), img=o['photo'], depth=o['depth'], pin=o['pin'], pgt=o['pgt'],
                      height=o['height'], leaves=o['leaves'],
                      texts=[f"CD = {o['pred'][k]['ch']:.0f}\n{o['pred'][k]['height']:.2f} m · {o['pred'][k]['leaves']} leaves" for k in ('head', 'row', 'field')],
                      recs=[o['pred'][k]['rec'] for k in ('head', 'row', 'field')]))
    pickle.dump(dict(species=sp, rows=E), open(OUT12 / f'supp_field_real_{sp}.pkl', 'wb'))
    # the page's real-plant section: every stop above
    out = R / 'site/data/real' / sp; out.mkdir(parents=True, exist_ok=True)
    for old in out.glob('*'): old.unlink()
    index = []
    for (pl, row, st), o in cache.items():
        sid = f'{pl}_row{row:02d}_stop{st:02d}'
        it = ESD.Item(o['scale'], ESD.ERR_MAX[sp] * (1 if sp == 'maize' else 3))
        sc = ESD.sub(o['pin'], 2 * 4096, row); it.add('scene', sc, mark=ESD.nn_dist(sc, o['pgt']) < 0.012); it.add('truth', ESD.sub(o['pgt'], 4096, row + 1))
        metrics, recipes = {}, {}
        for k, page_k in (('head', 'head'), ('row', 'row'), ('field', 'real')):      # the page's panel keys
            v = o['pred'][k]; Ps = ESD.sub(v['P'], 4096, row + 2)
            it.add(page_k, Ps, err_mm=ESD.nn_dist(Ps, o['pgt']) * o['scale'] * 1000)
            metrics[page_k] = dict(chamfer=round(v['ch'], 2), height=round(v['height'], 3), leaves=v['leaves'])
            recipes[page_k] = ESD.full_recipe(v['rec'])
        img = ESD.save_jpg((o['photo'] * 255).astype(np.uint8), out / f'{sid}.jpg', side=900, quality=90)
        tag = next((lab for r, lab, _ in gal if (r['plant'], int(r['row']), int(r['stop'])) == (pl, row, st)), 'median stop')
        it.write(out / sid, dict(task='real', species=sp, layout='field', row=row, stop=st, plant=pl, tag=tag, img=img,
                                 truth=dict(height=round(o['height'], 3), leaves=o['leaves']), occ=round(o['occ'], 3), metrics=metrics, recipes=recipes))
        index.append(dict(id=sid, tag=tag, layout='field', json=f'{sid}.json'))
    (out / 'index.json').write_text(json.dumps(dict(task='real', species=sp, source='field1000', items=index), indent=0))
    print(sp, 'page items', len(index), flush=True)


if __name__ == '__main__':
    for sp in ('maize', 'sorghum'):
        run(sp)
