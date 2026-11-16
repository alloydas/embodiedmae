"""Every file of the Fig. 3 plants (user, 2026-11-16: "for fig 3 share me the point cloud and each and every rgb, depth,
point cloud and procedural parameters").

The plants are those of fig3_dataset in the procedural-model export (export_procedural_models.py: make_dataset_fig_v2's
6 gallery plants per species, the highlighted one among them, and the crop-row plant of panel d). For each:
  gallery plants   all 10 camera views (val split): viewVV/{image.png, depth.png, point_cloud_cam.ply, camera_pose.json}
                   + procedural parameters (true.yml / true.xml) + sorghum point_cloud_complete_world.ply
  row-scene plant  all 10 row orientations (test split, shot 0): yawYYY/{row_image.png, row_depth.png,
                   row_scene_point_cloud_cam.ply, target_image.png, target_depth.png, target_point_cloud_cam.ply,
                   camera_pose.json} + row.json + procedural parameters
Written to OUT/<species>/<plant>_<role>/ and zipped per part so that every zip stays under GitHub's 100 MB limit.
  python figures/export_fig3_data.py
"""
import sys, pathlib, shutil, zipfile
R = pathlib.Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R), str(R / 'figures')]
import _plantviz as V

SRC = pathlib.Path('/work/mech-ai/alloydas/embodiedmae_outputs/procedural_models_for_viz/fig3_dataset')
OUT = pathlib.Path('/work/mech-ai/alloydas/embodiedmae_outputs/fig3_plants_data')
LIMIT = 90e6                                                    # bytes per zip (GitHub refuses files over 100 MB)

README = """# Fig. 3 plants: every view

The plants of the paper's Fig. 3 (the dataset figure): per species the 6 gallery plants (panel a; `gallery_highlighted`
is the plant of panels b and c) and the crop-row plant of panel d.

- `<species>/<plant>_gallery_*/`: `true.yml` and `true.xml` (procedural parameters: sorghum EvoEngine spline file and
  its XML; maize generator XML and its YAML), and for each of the 10 cameras `viewVV/`: `image.png` (sorghum:
  rgb_nobg.png, RGBA with transparent background; maize: rgba.png), `depth.png` (the renderer's packed RGBA depth),
  `point_cloud_cam.ply` (camera frame) and `camera_pose.json`. Sorghum also has `point_cloud_complete_world.ply`, the whole
  plant in the generator frame (each sorghum view cloud is this plant moved rigidly into that camera's frame; maize view
  clouds are what that one camera sees).
- `<species>/<plant>_row_scene/`: `true.yml` / `true.xml`, `row.json` (row layout) and for each of the 10 row
  orientations `yawYYY/` (0 = broadside): `row_image.png`, `row_depth.png`, `row_scene_point_cloud_cam.ply` (target plus
  neighbours), `target_image.png`, `target_depth.png`, `target_point_cloud_cam.ply` (the target alone, same camera) and
  `camera_pose.json`.
"""


def copy(src, dst):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy(src, dst); return 1
    return 0


if __name__ == '__main__':
    if OUT.exists(): shutil.rmtree(OUT)
    OUT.mkdir(parents=True); (OUT / 'README.md').write_text(README); n = 0
    for sp_dir in sorted(SRC.iterdir()):
        sp = sp_dir.name
        for pdir in sorted(sp_dir.iterdir()):
            row = pdir.name.endswith('_row_scene')
            plant = pdir.name[:-len('_row_scene')] if row else pdir.name.split('_gallery')[0]
            d = OUT / sp / pdir.name; d.mkdir(parents=True)
            for f in pdir.iterdir():                             # procedural parameters (+ row.json)
                if f.name.startswith('true.') or f.name == 'row.json': n += copy(f, d / f.name)
            if not row:
                for v in range(10):
                    f = V.folder(sp, 'val', plant, v); o = d / f'view{v:02d}'
                    cloud = (f / f'{plant}_nc_cam.ply') if sp == 'sorghum' else (f / 'pointcloud_cam.ply')
                    n += copy(f / V.RGB_FILE[sp], o / 'image.png') + copy(f / 'depth.png', o / 'depth.png')
                    n += copy(cloud, o / 'point_cloud_cam.ply') + copy(f / 'camera_pose.json', o / 'camera_pose.json')
                    if sp == 'sorghum' and v == 0: n += copy(f / f'{plant}_nc.ply', d / 'point_cloud_complete_world.ply')
            else:
                for th in range(0, 100, 10):
                    rs = V.row_shot_dir(sp, 'test', plant, th); y = d / f'yaw{th:03d}'
                    for src, nm in (('rgba.png', 'row_image.png'), ('depth.png', 'row_depth.png'), ('pointcloud_cam.ply', 'row_scene_point_cloud_cam.ply'),
                                    ('gt_rgba.png', 'target_image.png'), ('gt_depth.png', 'target_depth.png'),
                                    ('gt_pointcloud_cam.ply', 'target_point_cloud_cam.ply'), ('camera_pose.json', 'camera_pose.json')):
                        n += copy(rs / src, y / nm)
            print(sp, pdir.name, sum(f.stat().st_size for f in d.rglob('*') if f.is_file()) // 1_000_000, 'MB', flush=True)
    for z in OUT.parent.glob('fig3_plants_data_*.zip'): z.unlink()
    zips = []
    for sp_dir in sorted(p for p in OUT.iterdir() if p.is_dir()):
        for part, pick in (('gallery', lambda p: '_gallery' in p.name), ('row_scene', lambda p: p.name.endswith('row_scene'))):
            plants = [p for p in sorted(sp_dir.iterdir()) if pick(p)]
            groups, cur, size = [], [], 0
            for p in plants:                                     # split a part when its zip could pass the limit
                s = sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
                if cur and size + s > 55e6: groups.append(cur); cur, size = [], 0     # raw size: the sorghum PLYs barely compress
                cur.append(p); size += s
            if cur: groups.append(cur)
            for gi, g in enumerate(groups, 1):
                zp = OUT.parent / f"fig3_plants_data_{sp_dir.name}_{part}{'' if len(groups) == 1 else f'_{gi}'}.zip"
                with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as zf:
                    zf.write(OUT / 'README.md', 'fig3_plants_data/README.md')
                    for p in g:
                        for f in sorted(p.rglob('*')):
                            if f.is_file(): zf.write(f, 'fig3_plants_data/' + str(f.relative_to(OUT)))
                zips.append((zp.name, zp.stat().st_size // 1_000_000)); print('wrote', zp.name, zips[-1][1], 'MB', flush=True)
    print('files', n, '; zips', zips)
