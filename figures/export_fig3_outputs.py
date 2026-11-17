"""Model outputs for the Fig. 3 plants (user, 2026-11-16: "for fig 3 you did not send the outputs").

Fig. 3 itself shows dataset plants only; these are what CropCode writes for the same plants and views, added to the
folders of figures/export_fig3_data.py (OUT/<species>/<plant>_<role>/) and re-zipped:
  gallery plants (val split, every camera)  the distilled generalist (_plantviz CKPT 'generalist', the model of Fig. 4)
      given ONE input stream at a time, the recipe never an input (Fig. 4's protocol, seed 0):
      viewVV/predicted_from_{image,depth,point_cloud}/point_cloud_cam.ply   predicted cloud, camera frame, metres
                                                     procedural_parameters.{yml,xml}
  row-scene plant (test rows, every yaw)    the row fine-tuned model (Fig. 1 / Fig. 5), all three streams of the row:
      yawYYY/predicted_row_finetune/target_point_cloud_cam.ply (the target plant, row camera frame, metres)
                                    procedural_parameters.{yml,xml}
  outputs_summary.csv                       per output: Chamfer to the true cloud (normalised and m^2), height and
                                            leaf count written vs true
Clouds: the models work in the loaders' centred unit-sphere frame; gallery clouds are moved back with the centroid and
radius of the view's stored cloud (checked: the stored cloud in that frame matches the loader's input), row clouds with
the RowDataset's own centre / scale.
  sbatch --wait --partition=nova --account=mech-ai --qos=normal --cpus-per-task=8 --mem=32G --time=02:00:00 \
         --output=logs/%x_%j.out --wrap "python figures/export_fig3_outputs.py"
"""
import sys, pathlib, csv, shutil, zipfile
R = pathlib.Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R), str(R / 'figures'), str(R / 'eval')]
import numpy as np, torch
from scipy.spatial import cKDTree
import _plantviz as V
import export_procedural_models as EP
import export_fig3_data as F3
import procedural_formats as PF
import eval_row as ER
from row_dataset import RowDataset

SRC_NAME = {'rgb': 'image', 'depth': 'depth', 'pc': 'point_cloud'}


def write_ply(P, path):
    P = np.asarray(P, np.float32); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as fh:
        fh.write(f'ply\nformat binary_little_endian 1.0\nelement vertex {len(P)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n'.encode())
        fh.write(P.astype('<f4').tobytes())


def loader_frame(raw, pn):
    """centroid c and radius s with pn = (subset of raw - c) / s: the loader centres the SUBSAMPLED cloud, so start
    from the full cloud's values, match each loader point to its stored point and solve the scale and shift exactly"""
    from scipy.spatial import cKDTree
    c = raw.mean(0); s = np.linalg.norm(raw - c, axis=1).max(); tree = cKDTree(raw)
    for _ in range(200):                                           # scaled ICP onto the stored points; exact at convergence
        q = raw[tree.query(pn * s + c)[1]]
        A = np.zeros((len(pn) * 3, 4)); A[:, 0] = pn.reshape(-1); A[0::3, 1] = A[1::3, 2] = A[2::3, 3] = 1
        sol = np.linalg.lstsq(A, q.reshape(-1), rcond=None)[0]; s, c = float(sol[0]), sol[1:]
        if np.abs(q - (pn * s + c)).max() < 1e-6: break
    return c, s


def write_recipe(sp, rec, path_stem, note, fit):
    if sp == 'sorghum': EP.write_sorghum(rec, path_stem.with_suffix('.yml'), note)
    else: EP.write_maize(rec, path_stem.with_suffix('.xml'), note, fit)


if __name__ == '__main__':
    torch.set_num_threads(8); fit = EP.fits(); rows = []
    OUT = F3.OUT
    for sp in ('sorghum', 'maize'):
        # ---------------------------------------------------------------- gallery plants: generalist, one stream at a time
        m, cfg, ep = V.load_model(sp, 'generalist'); ds = V.dataset(sp, 'val')
        for d in sorted((OUT / sp).glob('*_gallery*')):
            plant = d.name.split('_gallery')[0]
            for v in range(10):
                nm = f'{plant}_{v:02d}'; it = V.item_by_name(ds, nm, seed=0)
                raw = V.read_cloud(d / f'view{v:02d}' / 'point_cloud_cam.ply')
                c, s = loader_frame(raw, it[2].numpy())
                fit_cd = float(cKDTree(raw).query(it[2].numpy() * s + c)[0].max())   # every loader point back on a stored point (m)
                assert fit_cd < 1e-5, (sp, nm, fit_cd)
                true = V.decode_recipe(sp, it[3].numpy(), it[4].numpy())
                for src in ('rgb', 'depth', 'pc'):
                    g = V.generate(m, it, [src], seed=0)
                    rec = V.decode_recipe(sp, g['par'].numpy(), it[4].numpy(), predicted=True)
                    o = d / f'view{v:02d}' / f'predicted_from_{SRC_NAME[src]}'
                    write_ply(g['pc'] * s + c, o / 'point_cloud_cam.ply')
                    cd = V.chamfer(g['pc'], it[2])
                    note = (f"fig3: {sp} {plant} view {v:02d} - distilled generalist ({V.CKPT[sp]['generalist'].parent.name}, epoch {ep}) "
                            f"from the {SRC_NAME[src].replace('_', ' ')} alone; recipe never an input")
                    write_recipe(sp, rec, o / 'procedural_parameters', note, fit)
                    rows.append(dict(species=sp, plant=plant, role=d.name.split(plant + '_')[1], view_or_yaw=f'view{v:02d}',
                                     model='distilled_generalist', input=SRC_NAME[src], chamfer_norm=cd, chamfer_m2=cd * s * s,
                                     height_pred=rec['height'], height_true=true['height'], leaves_pred=rec['n'], leaves_true=true['n']))
                print(sp, nm, 'frame fit', f'{fit_cd:.1e}', [(r['input'], round(r['chamfer_norm'] * 1e3, 2), round(r['height_pred'], 2), r['leaves_pred'])
                                                            for r in rows[-3:]], 'true', round(true['height'], 2), true['n'], flush=True)
        # ---------------------------------------------------------------- row-scene plant: row fine-tune, all three streams
        root, ft_run, _ = ('/work/mech-ai/alloydas/sorghum_row', 'sorghum_row_ft', None) if sp == 'sorghum' else ('/work/mech-ai/alloydas/maize_row', 'maize_row_ft_e1', None)
        ft, rcfg, _ = ER.build_model(R / 'outputs' / ft_run, 'checkpoints/checkpoint_epoch_100.pth', 'cpu', sp)
        info_sp = ER.species_info(sp); gain = float(rcfg.get('rgb_brighten_gain', 1.0)) if sp == 'sorghum' else None
        d = next((OUT / sp).glob('*_row_scene')); plant = d.name[:-len('_row_scene')]
        rds = RowDataset(root, 'test', sp, 'row', rcfg.get('img_size', 224), rcfg.get('num_points', info_sp['num_points']), info_sp['max_leaves'],
                         thetas=list(range(0, 100, 10)), shots=[0], plants=[plant], rgb_gain=gain,
                         gt_points=rcfg.get('gt_points', rcfg.get('num_points', info_sp['num_points'])), occlusion_stats=True)
        mr = float(rcfg.get('mask_ratio', 0.8))
        for i in range(len(rds)):
            pl, theta, shot, folder = rds.items[rds._resolve(i)]
            seed = int(pl.split('_')[-1]) * 7 + theta; np.random.seed(seed)
            rgb, depth, pc, params, valid, _, _, gt_pc, inf = rds[i]; b = lambda x: x[None]
            with torch.no_grad():
                pred, par = ER.predict(ft, 'all', b(rgb), b(depth), b(pc), b(params), mr, seed)
            scale, centre = float(inf['scale']), np.asarray(inf['centre'], np.float64).reshape(3)
            _, cg_m2, _, _ = ER.score_cloud(pred, b(gt_pc), torch.tensor([scale]))
            raw_t = V.read_cloud(d / f'yaw{theta:03d}' / 'target_point_cloud_cam.ply')
            back = gt_pc.numpy() * scale + centre                         # the frame check: the target cloud moved back
            fit_cd = V.chamfer(back, raw_t)
            assert fit_cd < 1e-6, (sp, pl, theta, fit_cd)
            rec = V.decode_recipe(sp, par[0].clamp(0, 1).numpy(), valid.numpy(), predicted=True)
            true = V.decode_recipe(sp, params.numpy(), valid.numpy())
            o = d / f'yaw{theta:03d}' / 'predicted_row_finetune'
            write_ply(pred[0].numpy() * scale + centre, o / 'target_point_cloud_cam.ply')
            note = f"fig3: {sp} {pl} yaw {theta} - row fine-tuned model ({ft_run}, epoch 100) on the test row, all three streams; recipe never an input"
            write_recipe(sp, rec, o / 'procedural_parameters', note, fit)
            cd = float(cg_m2[0])
            rows.append(dict(species=sp, plant=pl, role='row_scene', view_or_yaw=f'yaw{theta:03d}', model='row_finetune', input='image+depth+point_cloud',
                             chamfer_norm=cd / scale ** 2, chamfer_m2=cd, height_pred=rec['height'], height_true=true['height'],
                             leaves_pred=rec['n'], leaves_true=true['n']))
            print(sp, pl, theta, 'frame fit', f'{fit_cd:.1e}', 'CD 1e-4 m2', round(cd * 1e4, 2), 'h', round(rec['height'], 3), round(true['height'], 3),
                  'n', rec['n'], true['n'], flush=True)
    PF.convert(OUT)
    with open(OUT / 'outputs_summary.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (OUT / 'README.md').write_text(F3.README + """
## Model outputs

What CropCode writes for the same plants and views (Fig. 3 itself shows the dataset only):

- gallery plants, `viewVV/predicted_from_{image,depth,point_cloud}/`: the distilled generalist (the model of Fig. 4)
  given that one input alone, the procedural parameters never an input: `point_cloud_cam.ply` (the predicted plant cloud,
  in the view's camera frame, metres, like the input cloud) and `procedural_parameters.yml` / `.xml`.
- row-scene plant, `yawYYY/predicted_row_finetune/`: the row fine-tuned model (Figs. 1 and 5) given the row image, depth
  and scene cloud: `target_point_cloud_cam.ply` (the predicted target plant, row camera frame, metres; compare with
  `target_point_cloud_cam.ply` one level up) and `procedural_parameters.yml` / `.xml`.
- `outputs_summary.csv`: per output, Chamfer to the true cloud (normalised frame and m^2), and written vs true height (m)
  and leaf count. Predicted leaf counts are read from the written parameters (maize: the plant token; sorghum: the stem
  length), as in the paper's tables.
""")
    # re-zip exactly as export_fig3_data.py does
    for z in OUT.parent.glob('fig3_plants_data_*.zip'): z.unlink()
    for sp_dir in sorted(p for p in OUT.iterdir() if p.is_dir()):
        for part, pick in (('gallery', lambda p: '_gallery' in p.name), ('row_scene', lambda p: p.name.endswith('row_scene'))):
            groups, cur, size = [], [], 0
            for p in [p for p in sorted(sp_dir.iterdir()) if pick(p)]:
                s_ = sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
                if cur and size + s_ > 55e6: groups.append(cur); cur, size = [], 0
                cur.append(p); size += s_
            if cur: groups.append(cur)
            for gi, g in enumerate(groups, 1):
                zp = OUT.parent / f"fig3_plants_data_{sp_dir.name}_{part}{'' if len(groups) == 1 else f'_{gi}'}.zip"
                with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as zf:
                    zf.write(OUT / 'README.md', 'fig3_plants_data/README.md'); zf.write(OUT / 'outputs_summary.csv', 'fig3_plants_data/outputs_summary.csv')
                    for p in g:
                        for f in sorted(p.rglob('*')):
                            if f.is_file(): zf.write(f, 'fig3_plants_data/' + str(f.relative_to(OUT)))
                print('wrote', zp.name, zp.stat().st_size // 1_000_000, 'MB', [p.name for p in g], flush=True)
