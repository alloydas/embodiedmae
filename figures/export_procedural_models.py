"""Export the procedural models behind the paper's figures in the generators' own formats, for external rendering
(user, 2026-11-14: "Share me the procedural models for visualization I will generate that").

For every plant a figure shows: the TRUE recipe (the generator's own file, copied) and every PREDICTED recipe the figure
draws, written in the same format:
  sorghum  <name>.yml   the EvoEngine spline-file layout: Sorghums -> Parameters (stem_length, stem_direction) -> Leaves
                        (starting_point, length, roll_angle [deg], branching_angle [deg], width [m]); no Center/Left/Right
                        Points (those are generator output)
  maize    <name>.xml   the maize generator's XML: <plant height> <Tiller> <leaves number> <leaf id distance leafLength
                        leafWidth leafAzimuthDeg leafAngle droopiness avgCurvature bendAngleDeg bladeLength/>.
                        The model predicts height, leaf count and per leaf distance, leafLength, leafWidth, leafAngle,
                        droopiness, avgCurvature and the azimuth jitter (leafAzimuthDeg = (180 * id + jitter) mod 360).
                        bendAngleDeg and bladeLength are NOT predicted: filled from linear fits on the true files
                        (bladeLength ~ leafLength, bendAngleDeg ~ droopiness), and marked in an XML comment.
Plus recipes.json (every recipe decoded, true and predicted, in one file) and README.md.

Sources (the figures' own dumps, so the recipes are exactly the drawn ones):
  Fig. 1   teaser_row_dump.pkl (make_teaser_recipe_tiles.py): row fine-tune on the test row, yaw 0
  Fig. 2   _fig12_kit dump: Stage 2 photo-only before (headline) / after (distilled generalist); Stage 3 row zero-shot
           headline / row fine-tune
  Fig. 5 and the supplement row galleries   row_gallery_recipes.pkl (make_rows_recipe_dump.py): row fine-tune and
           isolated control, yaw 0 and 90
  python figures/export_procedural_models.py   -> /work/mech-ai/alloydas/embodiedmae_outputs/procedural_models_for_viz(.zip)
"""
import sys, pathlib, pickle, json, shutil, re
R = pathlib.Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R), str(R / 'figures'), str(R / 'eval')]
import numpy as np
import _plantviz as V
import _fig12_kit as K
import make_row_gallery_v2 as RG

OUT = pathlib.Path('/work/mech-ai/alloydas/embodiedmae_outputs/procedural_models_for_viz')
SORG = pathlib.Path('/work/mech-ai-scratch/alloy/shorgum_data/new_data_50K/Sorghum_15K')
MAIZE = pathlib.Path('/work/mech-ai-scratch/alloy/Maize_1')
TEASER = pathlib.Path('/work/mech-ai/alloydas/embodiedmae_outputs/_fig12/teaser_row_dump.pkl')
ALL = {}


def find_true(sp, plant):
    if sp == 'sorghum':
        for split in ('test', 'val', 'train'):
            p = SORG / split / f'{plant}_00' / f'{plant}_spline.yml'
            if p.exists(): return p
    else:
        pid = plant.split('_')[-1]
        for split in ('test', 'val', 'train'):
            p = MAIZE / split / f'{plant}_00' / f'maize_{pid}_spline.xml'
            if p.exists(): return p
    raise FileNotFoundError(plant)


def fits():
    """bladeLength ~ a * leafLength, bendAngleDeg ~ b0 + b1 * droopiness, from 200 test maize plants"""
    L, B, D, A = [], [], [], []
    for d in sorted((MAIZE / 'test').glob('plant_*_00'))[:200]:
        x = next(d.glob('maize_*_spline.xml')).read_text()
        for m in re.finditer(r'<leaf [^>]*?leafLength="([-\d.e]+)"[^>]*?droopiness="([-\d.e]+)"[^>]*?bendAngleDeg="([-\d.e]+)" bladeLength="([-\d.e]+)"', x):
            L.append(float(m[1])); D.append(float(m[2])); A.append(float(m[3])); B.append(float(m[4]))
    L, B, D, A = map(np.array, (L, B, D, A))
    a = float((L @ B) / (L @ L)); b1, b0 = np.polyfit(D, A, 1)
    return a, float(b0), float(b1), len(L)


def write_sorghum(rec, path, note):
    lines = [f'# {note}', '# Predicted recipe (CropCode), in the layout of the EvoEngine spline file; angles in degrees, lengths in metres.',
             'Sorghums:', '  - Instance Index: 0', '    Parameters:', f"      stem_length: {rec['height']:.6f}", '      stem_direction: [0, 1, 0]', '    Leaves:']
    for i, l in enumerate(rec['leaves']):
        lines += [f'      - Leaf Index: {i + 1}', f"        starting_point: {float(l['starting_point']):.6f}", f"        length: {float(l['length']):.6f}",
                  f"        roll_angle: {float(l['roll_angle']):.6f}", f"        branching_angle: {float(l['branching_angle']):.6f}", f"        width: {float(l['width']):.6f}"]
    path.write_text('\n'.join(lines) + '\n')


def write_maize(rec, path, note, fit):
    a, b0, b1, n = fit
    out = ['<?xml version="1.0" encoding="UTF-8"?>', f'<!-- {note} -->',
           f'<!-- Predicted recipe (CropCode). bendAngleDeg and bladeLength are not predicted: filled as bladeLength = {a:.4f} x leafLength, '
           f'bendAngleDeg = {b0:.3f} + {b1:.4f} x droopiness (fits on {n} true leaves). -->',
           f'<plant height="{rec["height"]:.6f}">', '    <Tiller>', f'        <leaves number="{len(rec["leaves"])}">']
    for i, l in enumerate(rec['leaves']):
        az = (180.0 * i + float(l['azJitterDeg'])) % 360.0; dr = float(l['droopiness'])
        out.append(f'            <leaf id="{i}" distance="{float(l["distance"]):.6f}" leafLength="{float(l["leafLength"]):.6f}" leafWidth="{float(l["leafWidth"]):.6f}" '
                   f'leafAzimuthDeg="{az:.4f}" leafAngle="{float(l["leafAngle"]):.4f}" droopiness="{dr:.4f}" avgCurvature="{float(l["avgCurvature"]):.4f}" '
                   f'bendAngleDeg="{b0 + b1 * dr:.4f}" bladeLength="{a * float(l["leafLength"]):.6f}"/>')
    out += ['        </leaves>', '    </Tiller>', '</plant>']
    path.write_text('\n'.join(out) + '\n')


def emit(fig, sp, plant, tag, preds, true_rec, fit):
    """preds: {name: (rec, description)}"""
    d = OUT / fig / sp / (plant + (f'_{tag}' if tag else '')); d.mkdir(parents=True, exist_ok=True)
    src = find_true(sp, plant); shutil.copy(src, d / ('true' + src.suffix))
    ext = '.yml' if sp == 'sorghum' else '.xml'
    entry = dict(species=sp, plant=plant, view=tag, true_file=str((d / ('true' + src.suffix)).relative_to(OUT)), true=jsonable(true_rec), predicted={})
    for name, (rec, desc) in preds.items():
        note = f'{fig}: {sp} {plant} {tag or ""} -- {desc}'.replace('--', '-')
        (write_sorghum if sp == 'sorghum' else lambda r, p, n: write_maize(r, p, n, fit))(rec, d / f'predicted_{name}{ext}', note)
        entry['predicted'][name] = dict(description=desc, file=str((d / f'predicted_{name}{ext}').relative_to(OUT)), recipe=jsonable(rec))
    ALL.setdefault(fig, []).append(entry)


def jsonable(rec):
    return dict(height=float(rec['height']), n_leaves=int(rec.get('n', len(rec['leaves']))),
                leaves=[{k: float(v) for k, v in l.items()} for l in rec['leaves']])


if __name__ == '__main__':
    if OUT.exists(): shutil.rmtree(OUT)
    OUT.mkdir(parents=True); fit = fits(); print('fits', fit, flush=True)
    # Fig. 1
    T = pickle.load(open(TEASER, 'rb'))
    for sp in ('sorghum', 'maize'):
        d = T[sp]; rt = V.decode_recipe(sp, d['par'], d['tv']); rp = V.decode_recipe(sp, d['gen_par'], d['tv'], predicted=True)
        emit('fig1_teaser', sp, d['plant'], 'row_yaw000', {'row_finetune': (rp, 'row fine-tuned model on the occluded test row (photo, depth, scene cloud), yaw 0')}, rt, fit)
    # Fig. 2
    D = K.load_dump(); d = D['stage']['sorghum']['Sorghum_10036_00']; r = D['rows']['sorghum']['Sorghum_10113']
    emit('fig2_stages', 'sorghum', 'Sorghum_10036', 'view00_stage2',
         {'before_distillation': (d['one']['rgb:before']['recipe'], 'Stage 2 before: headline model from the photograph alone'),
          'after_distillation': (d['one']['rgb:after']['recipe'], 'Stage 2 after: distilled generalist from the photograph alone')}, d['recipe_true'], fit)
    emit('fig2_stages', 'sorghum', 'Sorghum_10113', 'row_stage3',
         {'before_row_finetune': (r['pred']['head:all']['recipe'], 'Stage 3 before: headline model zero-shot on the row, all streams'),
          'after_row_finetune': (r['pred']['ft:all']['recipe'], 'Stage 3 after: row fine-tuned model on the row, all streams')}, r['recipe_true'], fit)
    # Fig. 5 + supplement
    REC = pickle.load(open(RG.OUT.with_name('row_gallery_recipes.pkl'), 'rb')); G = pickle.load(open(RG.OUT, 'rb'))
    picks = {'sorghum': [('Sorghum_10756', 90), ('Sorghum_14816', 0), ('Sorghum_4815', 0)], 'maize': [('plant_0320', 90), ('plant_6665', 0), ('plant_4604', 0)]}
    for fig, sel in (('fig5_rows', lambda sp: picks[sp]), ('supp_rows_galleries', lambda sp: [(p, th) for p, _ in G[sp]['more'] for th in RG.THETAS])):
        for sp in ('sorghum', 'maize'):
            for pl, th in sel(sp):
                x = REC[sp][(pl, th)]
                emit(fig, sp, pl, f'row_yaw{th:03d}',
                     {'row_finetune': (x['ft'], f"row fine-tuned model on the test row, yaw {th}, all streams; leaf-parameter MAE {x['ft_leaf_mae']:.3f}"),
                      'isolated_control': (x['iso'], f"isolated-trained control on the same row, yaw {th}, all streams; leaf-parameter MAE {x['iso_leaf_mae']:.3f}")}, x['true'], fit)
    json.dump(ALL, open(OUT / 'recipes.json', 'w'), indent=1)
    n = sum(len(v) for v in ALL.values())
    (OUT / 'README.md').write_text(f"""# CropCode procedural models for external rendering

{n} plants, the ones the paper's figures draw. Each folder `<figure>/<species>/<plant>_<view>/` holds:

- `true.yml` (sorghum) or `true.xml` (maize): the generator's own file for that plant, copied unchanged.
- `predicted_<model>.yml` / `.xml`: the recipe a CropCode model writes, in the same layout.

Sorghum (EvoEngine spline layout): `stem_length` (m), and per leaf `starting_point` (fraction of the stem),
`length` (m), `roll_angle` and `branching_angle` (degrees), `width` (m). The predicted files carry no
Center/Left/Right Points (those are generator output) and no Seed / Generator Path (they would regenerate the TRUE plant).

Maize (generator XML): `height`, leaf count, and per leaf `distance`, `leafLength`, `leafWidth`, `leafAzimuthDeg`,
`leafAngle`, `droopiness`, `avgCurvature`. `bendAngleDeg` and `bladeLength` are not predicted by the model; they are
filled from linear fits on the true files: bladeLength = {fit[0]:.4f} x leafLength, bendAngleDeg = {fit[1]:.3f} + {fit[2]:.4f} x
droopiness ({fit[3]} leaves).

`recipes.json` holds every recipe decoded (true and predicted) per figure.

| folder | figure | predicted files |
|---|---|---|
| fig1_teaser | Fig. 1 | row fine-tune on the occluded test row (yaw 0) |
| fig2_stages | Fig. 2 | Stage 2: before / after distillation, photo only; Stage 3: before / after row fine-tuning |
| fig5_rows | Fig. 5 | row fine-tune and isolated control, on the test row |
| supp_rows_galleries | Figs. 28-29 | the same, ten plants per species at yaw 0 and 90 |
""")
    shutil.make_archive(str(OUT), 'zip', OUT.parent, OUT.name)
    print('wrote', OUT, 'and', str(OUT) + '.zip', n, 'plants', flush=True)
