"""Every file of the Fig. 5 plants (user, 2026-11-16: "share fig 5 data also", after the Fig. 3 data).

Fig. 5 (crop-row examples): sorghum Sorghum_10756 / 14816 / 4815 and maize plant_0320 / 6665 / 4604, held-out test rows.
Per plant:
  yawYYY/ for all 10 row orientations (test split, shot 0): row_image.png, row_depth.png, row_scene_point_cloud_cam.ply
          (target + neighbours), target_image.png, target_depth.png, target_point_cloud_cam.ply (the target alone, same
          camera), camera_pose.json
  row.json                        the row layout
  true.yml / true.xml             the generator's procedural parameters (and their other-format copy)
  predicted_{row_finetune,isolated_control}_yaw{000,090}.{yml,xml}
                                  what the row-trained model (Fig. 5) and the isolated control write from the row at
                                  broadside and end-on (all streams; the recipes of make_rows_recipe_dump.py, the
                                  figure's own predictions; header comment states the model and its leaf-parameter MAE)
Zipped so that every zip stays well under GitHub's limits.  python figures/export_fig5_data.py
"""
import sys, pathlib, shutil, zipfile, pickle
R = pathlib.Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R), str(R / 'figures'), str(R / 'eval')]
import _plantviz as V
import export_procedural_models as EP
import make_row_gallery_v2 as RG
import procedural_formats as PF

OUT = pathlib.Path('/work/mech-ai/alloydas/embodiedmae_outputs/fig5_plants_data')
PICKS = {'sorghum': [('Sorghum_10756', 90), ('Sorghum_14816', 0), ('Sorghum_4815', 0)],      # = Fig. 5 (make_ugg_figs --fig5)
         'maize': [('plant_0320', 90), ('plant_6665', 0), ('plant_4604', 0)]}
ROWROOT = {'sorghum': pathlib.Path('/work/mech-ai/alloydas/sorghum_row/test'), 'maize': pathlib.Path('/work/mech-ai/alloydas/maize_row/test')}

README = """# Fig. 5 plants: every view and their procedural models

The six held-out crop-row plants of the paper's Fig. 5 (sorghum Sorghum_10756, Sorghum_14816, Sorghum_4815; maize
plant_0320, plant_6665, plant_4604). The folder name gives the yaw the figure shows.

- `yawYYY/` for all 10 row orientations (0 = broadside, 90 = end-on): `row_image.png`, `row_depth.png`,
  `row_scene_point_cloud_cam.ply` (the target and its neighbours, camera frame), `target_image.png`, `target_depth.png`,
  `target_point_cloud_cam.ply` (the target plant alone, same camera), `camera_pose.json`.
- `row.json`: the row layout.
- `true.yml` + `true.xml`: the target's procedural parameters (sorghum: EvoEngine spline file and its XML; maize:
  generator XML and its YAML).
- `predicted_row_finetune_yaw{000,090}.{yml,xml}`: what the row-trained model writes from the row (the purple plant of
  Fig. 5) at broadside and end-on, all three input streams; `predicted_isolated_control_yaw*` the same for the
  isolated-trained control. Each file's header comment names the model and its leaf-parameter MAE.
"""


def copy(src, dst):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy(src, dst); return 1
    return 0


if __name__ == '__main__':
    fit = EP.fits(); REC = pickle.load(open(RG.OUT.with_name('row_gallery_recipes.pkl'), 'rb'))
    if OUT.exists(): shutil.rmtree(OUT)
    OUT.mkdir(parents=True); (OUT / 'README.md').write_text(README); n = 0
    for sp, picks in PICKS.items():
        for pl, th_fig in picks:
            d = OUT / sp / f'{pl}_fig5_yaw{th_fig:03d}'; d.mkdir(parents=True)
            src = EP.find_true(sp, pl); n += copy(src, d / ('true' + src.suffix))
            n += copy(ROWROOT[sp] / pl / 'row.json', d / 'row.json')
            for th in (0, 90):
                x = REC[sp].get((pl, th))
                if x is None: continue
                for tag, nm in (('ft', 'row_finetune'), ('iso', 'isolated_control')):
                    note = f"fig5: {sp} {pl} yaw {th} - {'row fine-tuned model' if tag == 'ft' else 'isolated-trained control'} on the test row, all streams; leaf-parameter MAE {x[tag + '_leaf_mae']:.3f}"
                    path = d / (f'predicted_{nm}_yaw{th:03d}' + ('.yml' if sp == 'sorghum' else '.xml'))
                    (EP.write_sorghum(x[tag], path, note) if sp == 'sorghum' else EP.write_maize(x[tag], path, note, fit)); n += 1
            for th in range(0, 100, 10):
                rs = V.row_shot_dir(sp, 'test', pl, th); y = d / f'yaw{th:03d}'
                for s_, nm in (('rgba.png', 'row_image.png'), ('depth.png', 'row_depth.png'), ('pointcloud_cam.ply', 'row_scene_point_cloud_cam.ply'),
                               ('gt_rgba.png', 'target_image.png'), ('gt_depth.png', 'target_depth.png'),
                               ('gt_pointcloud_cam.ply', 'target_point_cloud_cam.ply'), ('camera_pose.json', 'camera_pose.json')):
                    n += copy(rs / s_, y / nm)
            print(sp, d.name, sum(f.stat().st_size for f in d.rglob('*') if f.is_file()) // 1_000_000, 'MB', flush=True)
    n += PF.convert(OUT)                                          # every model file as .yml AND .xml
    for z in OUT.parent.glob('fig5_plants_data_*.zip'): z.unlink()
    zips = []
    for sp_dir in sorted(p for p in OUT.iterdir() if p.is_dir()):
        groups, cur, size = [], [], 0
        for p in sorted(sp_dir.iterdir()):
            s = sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
            if cur and size + s > 55e6: groups.append(cur); cur, size = [], 0
            cur.append(p); size += s
        if cur: groups.append(cur)
        for gi, g in enumerate(groups, 1):
            zp = OUT.parent / f"fig5_plants_data_{sp_dir.name}{'' if len(groups) == 1 else f'_{gi}'}.zip"
            with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.write(OUT / 'README.md', 'fig5_plants_data/README.md')
                for p in g:
                    for f in sorted(p.rglob('*')):
                        if f.is_file(): zf.write(f, 'fig5_plants_data/' + str(f.relative_to(OUT)))
            zips.append((zp.name, zp.stat().st_size // 1_000_000, [p.name for p in g])); print('wrote', *zips[-1], flush=True)
    print('files', n)
