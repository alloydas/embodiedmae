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
README_EXTRA = '''
## Fig. 3: the plants' data

`fig3_dataset/<species>/<plant>_<role>/` also holds the dataset files the figure is drawn from (as stored in the dataset):

- `gallery_1..5`, `gallery_highlighted` (the 6 gallery plants per species): for each view the figure draws,
  `viewVV_image.png` (photograph), `viewVV_depth.png`, `viewVV_point_cloud_cam.ply` (camera frame) and
  `viewVV_camera_pose.json`. The gallery view is the side view (sorghum camera 4; maize the camera nearest the horizon);
  the highlighted plant also has camera 0 (panel b) and the five cameras of panel c. Sorghum: `point_cloud_complete_world.ply`
  is the whole plant in the generator frame (each view cloud is the same plant moved rigidly into that camera's frame);
  maize view clouds are single-camera captures.
- `row_scene` (panel d, the test row seen broadside, shot 0): `row_image.png`, `row_depth.png`,
  `row_scene_point_cloud_cam.ply` (target + neighbours), `target_image.png`, `target_depth.png`,
  `target_point_cloud_cam.ply` (the target alone, same camera), `camera_pose.json`, `row.json` (the row layout).

## E10 (real plants)

Real plants have no generator file, so each stop folder `<layout>_row<row>_stop<stop>/` holds `truth.json` / `.yml` /
`.xml` (true height, maize leaf count, and for the 3,000-row maize the per-leaf CSV values) and one `predicted_<model>`
file per model the figure shows (all three inputs): `headline_zeroshot`, `synthetic_row_finetune`, `real_row_finetune`.
`recipes_e10.json` collects them. Folders: e10_paper_maize / e10_paper_sorghum (paper E10 figures, first upload),
supp_e10_gallery_maize / _sorghum (supplement E10 galleries), e10_rows3000_examples (the 3,000-row upload examples).

## Both formats

Every model file is given twice, with the same fields and values: sorghum `<name>.yml` (EvoEngine spline layout) and
`<name>.xml` (`<sorghums><sorghum><parameters .../><leaves><leaf .../>`; the true file's generator geometry kept as
`x y z; ...` text); maize `<name>.xml` (generator XML) and `<name>.yml` (`plant: {height, leaves: [...]}`). Header
comments (which model wrote the file, how the unpredicted maize fields were filled) are carried into both.
'''
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


def true_recipe(sp, plant):
    """the plant's true recipe decoded from its generator file (for recipes.json)"""
    if sp == 'sorghum':
        import embodied_mae_4m as S
        tv, par = S.load_spline_params(str(find_true(sp, plant)))[:2]
    else:
        import embodied_mae_4m_maize as M
        tv, par = M.load_spline_params(str(find_true(sp, plant)), schema='v2')[:2]
    return V.decode_recipe(sp, np.asarray(par), np.asarray(tv))


def jsonable(rec):
    return dict(height=float(rec['height']), n_leaves=int(rec.get('n', len(rec['leaves']))),
                leaves=[{k: float(v) for k, v in l.items()} for l in rec['leaves']])


if __name__ == '__main__':
    if OUT.exists():                                   # keep the E10 folders of export_e10_recipes.py
        for x in OUT.iterdir():
            if not (x.name.startswith(('e10_', 'supp_e10_')) or x.name == 'recipes_e10.json'):
                shutil.rmtree(x) if x.is_dir() else x.unlink()
    OUT.mkdir(parents=True, exist_ok=True); fit = fits(); print('fits', fit, flush=True)
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
    # Fig. 3 (the dataset): true plants only. The gallery is make_dataset_fig_v2.gallery_plants, reproduced here (that
    # module renders the figure on import): 5 val plants drawn with seed 3, sorted by height, plus the highlighted plant.
    import pandas as pd
    PL3 = {'sorghum': 'Sorghum_10001', 'maize': 'plant_0003'}; RP3 = {'sorghum': 'Sorghum_10007', 'maize': 'plant_0004'}
    ROWROOT = {'sorghum': pathlib.Path('/work/mech-ai/alloydas/sorghum_row/test'), 'maize': pathlib.Path('/work/mech-ai/alloydas/maize_row/test')}
    for sp in ('sorghum', 'maize'):
        rng = np.random.default_rng(3)
        if sp == 'sorghum':
            t = pd.read_csv(V.ROOTS[sp] / 'assignment.csv').merge(pd.read_csv(V.ROOTS[sp] / 'features.csv'), on='plant')
            t = t[t.split == 'val']; t['name'] = 'Sorghum_' + t.plant.astype(str); t['h'] = t.stem_length
        else:
            t = pd.read_csv(V.ROOTS[sp] / 'plant_scores.csv'); t = t[t.split == 'val']; t['name'] = t.plant; t['h'] = t.stem_internodeSum
        t = t[t.name != PL3[sp]]
        pick = list(t.sample(5, random_state=int(rng.integers(1e6))).name); hh = dict(zip(t.name, t.h))
        # the plants' DATA too (user, 2026-11-16: "for fig 3 share point cloud and procedural parameters", "Share 6 plants
        # data"): per view the figure draws, the photo, depth image, point cloud and camera pose as stored in the dataset
        el = pd.read_csv(R / 'data_split/maize_view_elevation.csv.gz') if sp == 'maize' else None
        def side_view(pl):                              # = make_dataset_fig_v2.side_view
            if sp == 'sorghum': return 4
            t_ = el[(el.split == 'val') & (el.plant == pl)]; return int(t_.iloc[np.argmin(np.abs(t_.elev_deg.values))]['view'])
        def camera_views(pl):                           # panel (c): every other camera by elevation
            if sp == 'sorghum': return list(range(10))[::2]
            return el[(el.split == 'val') & (el.plant == pl)].sort_values('elev_deg', ascending=False)['view'].tolist()[::2]
        def add_view(pl, v, d):
            f = V.folder(sp, 'val', pl, v); cloud = (f / f'{pl}_nc_cam.ply') if sp == 'sorghum' else (f / 'pointcloud_cam.ply')
            for src, nm in ((f / V.RGB_FILE[sp], f'view{v:02d}_image.png'), (f / 'depth.png', f'view{v:02d}_depth.png'),
                            (cloud, f'view{v:02d}_point_cloud_cam.ply'), (f / 'camera_pose.json', f'view{v:02d}_camera_pose.json')):
                if src.exists(): shutil.copy(src, d / nm)
            if sp == 'sorghum' and (f / f'{pl}_nc.ply').exists() and not (d / 'point_cloud_complete_world.ply').exists():
                shutil.copy(f / f'{pl}_nc.ply', d / 'point_cloud_complete_world.ply')     # the whole plant, generator frame
        for i, pl in enumerate(sorted(pick, key=lambda n: hh[n]) + [PL3[sp]]):
            tag = 'gallery' + ('_highlighted' if pl == PL3[sp] else f'_{i + 1}')
            emit('fig3_dataset', sp, pl, tag, {}, true_recipe(sp, pl), fit)
            d = OUT / 'fig3_dataset' / sp / f'{pl}_{tag}'
            for v in sorted({side_view(pl)} | ({0} | set(camera_views(pl)) if pl == PL3[sp] else set())):
                add_view(pl, v, d)
        emit('fig3_dataset', sp, RP3[sp], 'row_scene', {}, true_recipe(sp, RP3[sp]), fit)
        d = OUT / 'fig3_dataset' / sp / f'{RP3[sp]}_row_scene'
        shutil.copy(ROWROOT[sp] / RP3[sp] / 'row.json', d / 'row.json')
        rs = V.row_shot_dir(sp, 'test', RP3[sp], 0)                                     # panel (d): broadside, shot 0
        for src, nm in (('rgba.png', 'row_image.png'), ('depth.png', 'row_depth.png'), ('pointcloud_cam.ply', 'row_scene_point_cloud_cam.ply'),
                        ('gt_rgba.png', 'target_image.png'), ('gt_depth.png', 'target_depth.png'),
                        ('gt_pointcloud_cam.ply', 'target_point_cloud_cam.ply'), ('camera_pose.json', 'camera_pose.json')):
            if (rs / src).exists(): shutil.copy(rs / src, d / nm)
    # Fig. 4 (any input to all; maize in the paper, sorghum in the supplement): the distilled generalist from each stream alone
    import make_any2all_fig as A
    D4 = pickle.load(open(A.OUT4, 'rb'))
    desc = {'rgb': 'distilled generalist from the photograph alone', 'depth': 'distilled generalist from the depth image alone',
            'pc': 'distilled generalist from the point cloud alone',
            'text': 'distilled generalist given the true parameters as its only input (its reproduction of them)'}
    for sp, nm, fig in (('maize', 'plant_1875_09', 'fig4_any2all_maize'), ('sorghum', 'Sorghum_7712_05', 'supp_any2all_sorghum')):
        e = D4[sp][nm]; pl, view = nm.rsplit('_', 1)
        emit(fig, sp, pl, f'view{view}', {f'from_{ {"pc": "cloud", "text": "parameters"}.get(k, k) }': (e['out'][k]['rec'], desc[k]) for k in ('rgb', 'depth', 'pc')}, e['rec_true'], fit)   # the 'text' row shows the true parameters as its input
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
| fig3_dataset | Fig. 3 | none: the dataset figure shows true plants only (gallery of 6 per species and the crop-row scene plant, with row.json) |
| fig2_stages | Fig. 2 | Stage 2: before / after distillation, photo only; Stage 3: before / after row fine-tuning |
| fig4_any2all_maize / supp_any2all_sorghum | Fig. 4 and its sorghum twin in the supplement | the distilled generalist from each sensor alone: photo, depth, cloud (the figure's parameters row shows the true parameters as input) |
| fig5_rows | Fig. 5 | row fine-tune and isolated control, on the test row |
| supp_rows_galleries | Figs. 28-29 | the same, ten plants per species at yaw 0 and 90 |
""" + README_EXTRA)
    import procedural_formats; procedural_formats.convert(OUT)          # every model file as .yml AND .xml
    shutil.make_archive(str(OUT), 'zip', OUT.parent, OUT.name)
    print('wrote', OUT, 'and', str(OUT) + '.zip', n, 'plants', flush=True)
