"""Recipes for the crop-row galleries (paper Fig. 5 and the supplement row galleries).

figures/make_row_gallery_v2.py --dump stored only the predicted CLOUDS. Since 2026-11-13 Fig. 5 shows the procedural
model instead (user: "Now we have to update fig 5 for the same", after Fig. 2's before/after panels moved from clouds to
procedural models). This reruns the same two row models on the same items with the SAME protocol and seeds as that dump
(eval/eval_row.py predict, all three streams in, recipe never an input, mask ratio from the run config, per-item seed
plant_id * 7 + yaw set before the dataset draw and passed to predict), so the recipe belongs to the very prediction whose
cloud the old figure drew: the recomputed Chamfer is checked against the stored one.

Per item it stores the true recipe, each model's written recipe (figures/_plantviz.decode_recipe; leaf count read from
the recipe itself) and the recipe error of eval/eval_row.recipe_error: plant MAE and leaf MAE (the leaf-parameter MAE of
the paper's Table 7), on the normalised fields of the species' current layout.
  sbatch --wait --partition=nova --account=mech-ai --qos=normal --cpus-per-task=8 --mem=32G --output=logs/%x_%j.out \
         --wrap "python figures/make_rows_recipe_dump.py"
-> /work/mech-ai/alloydas/embodiedmae_outputs/_fig12/row_gallery_recipes.pkl
"""
import sys, pathlib, pickle
R = pathlib.Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R), str(R / 'figures'), str(R / 'eval')]
import numpy as np, torch
import make_row_gallery_v2 as RG
import _plantviz as V
import eval_row as ER
from row_dataset import RowDataset

OUT = RG.OUT.with_name('row_gallery_recipes.pkl')

if __name__ == '__main__':
    torch.set_num_threads(8)
    D0 = pickle.load(open(RG.OUT, 'rb')); D = {}
    for sp in ('sorghum', 'maize'):
        root, ft_run, iso_run = RG.SPEC[sp]
        ft, cfg, _ = ER.build_model(R / 'outputs' / ft_run, 'checkpoints/checkpoint_epoch_100.pth', 'cpu', sp)
        iso, _, _ = ER.build_model(R / 'outputs' / iso_run, 'checkpoints/checkpoint_epoch_100.pth', 'cpu', sp)
        info_sp = ER.species_info(sp); gain = float(cfg.get('rgb_brighten_gain', 1.0)) if sp == 'sorghum' else None
        plants = sorted({p for p, _ in D0[sp]['items']})
        ds = RowDataset(root, 'test', sp, 'row', cfg.get('img_size', 224), cfg.get('num_points', info_sp['num_points']), info_sp['max_leaves'],
                        thetas=list(RG.THETAS), shots=[0], plants=plants, rgb_gain=gain,
                        gt_points=cfg.get('gt_points', cfg.get('num_points', info_sp['num_points'])), occlusion_stats=True)
        mr = float(cfg.get('mask_ratio', 0.8)); items = {}
        for i in range(len(ds)):
            plant, theta, shot, folder = ds.items[ds._resolve(i)]
            if (plant, theta) not in D0[sp]['items']: continue
            seed = int(plant.split('_')[-1]) * 7 + theta
            np.random.seed(seed)
            rgb, depth, pc, params, valid, _, _, gt_pc, inf = ds[i]
            b = lambda x: x[None]
            rec = dict(true=V.decode_recipe(sp, params.numpy(), valid.numpy()))
            for tag, m in (('ft', ft), ('iso', iso)):
                with torch.no_grad():
                    pred, par = ER.predict(m, 'all', b(rgb), b(depth), b(pc), b(params), mr, seed)
                _, cg_m2, _, _ = ER.score_cloud(pred, b(gt_pc), torch.tensor([float(inf['scale'])]))
                cd_old = D0[sp]['items'][(plant, theta)]['out'][tag][1]; cd = float(cg_m2[0]) * 1e4
                assert abs(cd - cd_old) <= 1e-3 * max(1.0, cd_old), (sp, plant, theta, tag, cd, cd_old)
                pm, lm = ER.recipe_error(par, b(params), b(valid), info_sp)
                rec[tag] = V.decode_recipe(sp, par[0].clamp(0, 1).numpy(), valid.numpy(), predicted=True)
                rec[tag + '_plant_mae'] = float(pm[0]); rec[tag + '_leaf_mae'] = float(lm[0])
            items[(plant, theta)] = rec
            print(sp, plant, theta, {k: round(v, 4) for k, v in rec.items() if k.endswith('_mae')}, flush=True)
        D[sp] = items
    pickle.dump(D, open(str(OUT) + '.tmp', 'wb')); pathlib.Path(str(OUT) + '.tmp').replace(OUT)
    print('wrote', OUT, flush=True)
