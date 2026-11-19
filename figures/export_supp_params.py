"""Procedural parameters of every supplement figure that shows plants (user, 2026-11-19: "Share me the procedorial
parameters for each images for supplymenrtary").

Adds the two supplement figures the procedural-model export did not cover, into the same folder
(export_procedural_models.OUT), in the same layout (true.* = the generator's file, predicted_<model>.* = what that
model writes), then builds procedural_models/supplementary_procedural_models.zip from every supplement folder:
  supp_dataset_stats   fig:dataset_stats panel a (make_dataset_stats_fig.py): the eight val plants per species along
                       the leaf-count range, true parameters only
  supp_qual            fig:qual (make_qual_fig.py, defaults Sorghum_10001 / plant_0003, view 00, seed 0): the
                       parameters written from the PHOTOGRAPH ALONE by the headline model (before distillation) and the
                       distilled generalist (after), as the figure generates them
Already exported by export_procedural_models.py / export_e10_recipes.py and collected into the zip:
  fig4_any2all_maize + supp_any2all_sorghum (fig:any2any-maize / -sorghum), supp_rows_galleries (fig:supp-rows-*),
  e10_paper_maize / _sorghum (fig:e10_real / _sorghum), supp_e10_gallery_maize / _sorghum (fig:supp-e10-*).
Not covered: fig:occ_sources and fig:mixed (fig_sources_maize.png, fig_mixed_masking.png), drawn by a labmate's code
that does not record which plants it shows.
  sbatch --wait --partition=nova --account=mech-ai --qos=normal --cpus-per-task=8 --mem=32G --time=01:00:00 \
         --output=logs/%x_%j.out --wrap "python figures/export_supp_params.py"
"""
import sys, pathlib, shutil, zipfile
R = pathlib.Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R), str(R / 'figures'), str(R / 'eval')]
import numpy as np, pandas as pd, torch
import _plantviz as V
import export_procedural_models as EP
import procedural_formats as PF

SUPP = ['supp_dataset_stats', 'supp_qual', 'fig4_any2all_maize', 'supp_any2all_sorghum', 'supp_rows_galleries',
        'e10_paper_maize', 'e10_paper_sorghum', 'supp_e10_gallery_maize', 'supp_e10_gallery_sorghum']
ZIP = R / 'procedural_models' / 'supplementary_procedural_models.zip'
README = """# Supplementary material: procedural parameters of every figure that shows plants

Every model file is given as `.yml` and `.xml` with the same fields and values (sorghum: EvoEngine spline layout and its
XML; maize: generator XML and its YAML). `true.*` = the plant's generator file; `predicted_<model>.*` = the procedural
model written by that model (its header comment names the model and the input); real plants (E10) have `truth.*`
(measured height, maize leaf count and per-leaf CSV values) instead of a generator file.

| folder | supplement figure | contents |
|---|---|---|
| `supp_dataset_stats/` | Statistical overview of FieldScan, panel (a) | the eight validation plants per species along the leaf-count range (true only; folder name = plant and its position 1-8) |
| `supp_qual/` | Single-photograph 3D reconstruction | Sorghum_10001 and plant_0003, view 00: true, and the parameters written from the photograph alone by the headline model (before distillation) and the distilled generalist (after) |
| `fig4_any2all_maize/`, `supp_any2all_sorghum/` | Any-to-all cross-modal reconstruction (maize, sorghum) | true, and the distilled generalist from the photograph / depth image / point cloud alone |
| `supp_rows_galleries/` | Maize / sorghum test rows sampled evenly across the error distribution | ten test plants per species at yaw 0 and 90: true, row fine-tuned model and isolated control |
| `e10_paper_maize/`, `e10_paper_sorghum/` | Qualitative results on held-out real maize / sorghum rows | truth, and the pretrained (zero-shot), synthetic-row fine-tuned and real-row fine-tuned models |
| `supp_e10_gallery_maize/`, `supp_e10_gallery_sorghum/` | Held-out real maize rows across error percentiles; all eight held-out real sorghum plants | the same three models per camera stop |

Not included: the two occlusion-ablation figures (training occlusion sources; mixed masking), drawn by separate code
that does not record which plants it shows.
"""

if __name__ == '__main__':
    torch.set_num_threads(8); fit = EP.fits()
    for f in ('supp_dataset_stats', 'supp_qual'):
        if (EP.OUT / f).exists(): shutil.rmtree(EP.OUT / f)
    # ---- fig:dataset_stats panel a: the selection of make_dataset_stats_fig.py
    sf = pd.read_csv(V.ROOTS['sorghum'] / 'features.csv').merge(pd.read_csv(V.ROOTS['sorghum'] / 'assignment.csv')[['plant', 'split']], on='plant')
    sf['leaves'] = sf['n_leaves']; sf['name'] = 'Sorghum_' + sf['plant'].astype(str)
    mf = pd.read_csv(V.ROOTS['maize'] / 'plant_scores.csv'); mf['leaves'] = mf['leaf_count']; mf['name'] = mf['plant']
    for sp, T in (('sorghum', sf), ('maize', mf)):
        t = T[T.split == 'val'].sort_values('leaves')
        for k, q in enumerate(np.linspace(0.02, 0.98, 8)):
            pl = t.iloc[int(q * (len(t) - 1))]['name']
            EP.emit('supp_dataset_stats', sp, pl, f'{k + 1}', {}, EP.true_recipe(sp, pl), fit)
            print('dataset_stats', sp, k + 1, pl, flush=True)
    # ---- fig:qual: photograph alone, headline (before) and distilled generalist (after), seed 0 (make_qual_fig.gen)
    for sp, pl in (('sorghum', 'Sorghum_10001'), ('maize', 'plant_0003')):
        ds = V.dataset(sp, 'val'); it = V.item_by_name(ds, f'{pl}_00', seed=0); preds = {}
        for key, label in (('headline', 'headline model (before distillation)'), ('generalist', 'distilled generalist (after)')):
            m, cfg, ep = V.load_model(sp, key); g = V.generate(m, it, ['rgb'], seed=0)
            rec = V.decode_recipe(sp, g['par'].numpy(), it[4].numpy(), predicted=True)
            preds[key] = (rec, f'{label}, epoch {ep}, from the photograph alone (view 00)')
            print('qual', sp, pl, key, 'height %.3f n %d' % (rec['height'], rec['n']), 'cd %.2fe-3' % (V.chamfer(g['pc'], it[2]) * 1e3), flush=True)
        EP.emit('supp_qual', sp, pl, 'view00', preds, EP.true_recipe(sp, pl), fit)
    PF.convert(EP.OUT)
    # ---- the supplementary zip
    missing = [f for f in SUPP if not (EP.OUT / f).exists()]; assert not missing, missing
    (EP.OUT / 'README_supplementary.md').write_text(README)
    with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(EP.OUT / 'README_supplementary.md', 'supplementary_procedural_models/README.md')
        n = 0
        for f in SUPP:
            for p in sorted((EP.OUT / f).rglob('*')):
                if p.is_file(): zf.write(p, 'supplementary_procedural_models/' + str(p.relative_to(EP.OUT))); n += 1
    shutil.make_archive(str(EP.OUT), 'zip', EP.OUT.parent, EP.OUT.name)        # the full export too
    print('wrote', ZIP, ZIP.stat().st_size // 1000, 'kB,', n, 'files', flush=True)
