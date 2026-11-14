"""E10 procedural models for external rendering (user, 2026-11-14: "send the procedural parameters for E10 also").

Re-runs the E10 figure code UNCHANGED (same stops, models, seeds) and records the full recipe each model writes, by
wrapping eval_row.build_model / eval_row.predict and RealRowDataset.__getitem__ before the figure modules import them:
  e10_paper_maize / e10_paper_sorghum   paper Figs. fig:e10_real / fig:e10_real_sorghum (figures/make_e10_fig.py, first
                                        upload); its own figure output goes to a scratch dir, the paper copy is untouched
  supp_e10_gallery_maize / _sorghum     the supplement galleries (figures/make_e10_gallery_ugg.compute, which asserts
                                        every Chamfer equals the published one)
  e10_rows3000_examples                 the new 3,000-row examples (figures/make_e10_r3000_final.examples; images to scratch)
The real plants have no generator file: truth.json holds what the data carries (height; maize leaf count; for the
3,000-row maize also the per-leaf CSV values: internode distance, leaf length, width, curvature, zenith).
Written into the procedural_models_for_viz folder of export_procedural_models.py and re-zipped.
  sbatch ... --wrap "python figures/export_e10_recipes.py"
"""
import sys, pathlib, json, shutil, runpy
R = pathlib.Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R), str(R / 'figures'), str(R / 'eval')]
import numpy as np, torch, matplotlib
matplotlib.use('Agg')
import eval_row, real_rows
import _plantviz as V
import export_procedural_models as EP

SCR = pathlib.Path('/work/mech-ai/alloydas/embodiedmae_outputs/_e10_recipes'); SCR.mkdir(parents=True, exist_ok=True)
REC, RUN_OF, LAST = [], {}, {}
_bm, _pr, _gi = eval_row.build_model, eval_row.predict, real_rows.RealRowDataset.__getitem__


def build_model(run_dir, ckpt, *a, **k):
    out = _bm(run_dir, ckpt, *a, **k); RUN_OF[id(out[0])] = pathlib.Path(run_dir).name; return out


def predict(model, src, *a, **k):
    pp, ppar = _pr(model, src, *a, **k)
    if ppar is not None:
        REC.append(dict(run=RUN_OF.get(id(model), '?'), src=src, par=ppar[0].clamp(0, 1).float().cpu().numpy(), info=dict(LAST)))
    return pp, ppar


def getitem(self, i):
    out = _gi(self, i); LAST.clear(); LAST.update({k: v for k, v in out[-1].items()}); return out


eval_row.build_model, eval_row.predict, real_rows.RealRowDataset.__getitem__ = build_model, predict, getitem
LABEL = lambda run: ('headline_zeroshot' if 'pretrain' in run else 'synthetic_row_finetune' if 'row_ft' in run
                     else 'real_row_finetune' if 'real_ft' in run else run)


def collect(tag):
    out = list(REC); REC.clear(); return [dict(r, fig=tag) for r in out]


def run_script(path, argv, tag):
    sys.argv = [str(path)] + argv
    try: runpy.run_path(str(path), run_name='__main__')
    except SystemExit: pass
    return collect(tag)


def species_of(run):
    return 'maize' if 'maize' in run else 'sorghum'


def write(recs, fit):
    OUT = EP.OUT; n = 0; idx = {}
    for r in recs:
        sp = species_of(r['run']); inf = r['info']
        key = f"{inf.get('layout', 'rows3000')}_row{inf.get('row')}_stop{inf.get('stop')}"
        d = OUT / r['fig'] / sp / key; d.mkdir(parents=True, exist_ok=True)
        truth = dict(plant=str(inf.get('plant')), height_m=float(inf['height']), leaves=int(inf.get('leaves', -1)), occluded_fraction=float(inf.get('occ_frac', np.nan)))
        if inf.get('leaf_raw') is not None:
            cols = ('internode_dist_m', 'leaf_length_m', 'leaf_width_m', 'curvature_1pm')
            truth['leaves_csv'] = [dict(zip(cols, map(float, row)), zenith_deg=float(z)) for row, z in zip(inf['leaf_raw'], inf['leaf_zenith'])]
        (d / 'truth.json').write_text(json.dumps(truth, indent=1))
        rec = V.decode_recipe(sp, r['par'], None, predicted=True); lab = LABEL(r['run'])
        note = f"{r['fig']}: {sp} {key} (plant {truth['plant']}) - {lab} ({r['run']}), all three inputs"
        if sp == 'sorghum': EP.write_sorghum(rec, d / f'predicted_{lab}.yml', note)
        else: EP.write_maize(rec, d / f'predicted_{lab}.xml', note, fit)
        idx.setdefault(r['fig'], {}).setdefault(key, dict(species=sp, truth=truth, predicted={}))['predicted'][lab] = EP.jsonable(rec); n += 1
    return idx, n


if __name__ == '__main__':
    torch.set_num_threads(8)
    fit = EP.fits(); allrec = []
    for sp, tag in (('maize', 'e10_paper_maize'), ('sorghum', 'e10_paper_sorghum')):
        allrec += run_script(R / 'figures/make_e10_fig.py', ['--species', sp, '--out', str(SCR / f'fig_{tag}')], tag)
    import make_e10_gallery_ugg as G
    for sp in ('maize', 'sorghum'):
        G.compute(sp); allrec += collect(f'supp_e10_gallery_{sp}')
    import make_e10_r3000_final as F3
    F3.OUT = SCR
    for sp in ('maize', 'sorghum'):
        F3.examples(sp); allrec += collect('e10_rows3000_examples')
    for d in EP.OUT.glob('e10_*'): shutil.rmtree(d)
    for d in EP.OUT.glob('supp_e10_*'): shutil.rmtree(d)
    idx, n = write(allrec, fit)
    json.dump(idx, open(EP.OUT / 'recipes_e10.json', 'w'), indent=1)
    with open(EP.OUT / 'README.md', 'a') as fh:
        fh.write("""
## E10 (real plants)

Real plants have no generator file, so each stop folder `<layout>_row<row>_stop<stop>/` holds `truth.json` (true height,
maize leaf count, and for the 3,000-row maize the per-leaf CSV: internode distance, leaf length, width, curvature,
zenith) and one `predicted_<model>` file per model the figure shows (all three inputs, the figure's own seed):
`headline_zeroshot`, `synthetic_row_finetune`, `real_row_finetune`. `recipes_e10.json` collects them.

| folder | figure |
|---|---|
| e10_paper_maize / e10_paper_sorghum | Figs. e10_real / e10_real_sorghum (first upload) |
| supp_e10_gallery_maize / supp_e10_gallery_sorghum | supplement E10 galleries (first upload) |
| e10_rows3000_examples | the new 3,000-row upload: the 25/50/75th-percentile test examples |
""")
    shutil.make_archive(str(EP.OUT), 'zip', EP.OUT.parent, EP.OUT.name)
    print('wrote', n, 'predicted recipes;', {k: len(v) for k, v in idx.items()}, flush=True)
