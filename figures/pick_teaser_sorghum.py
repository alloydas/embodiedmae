"""Pick the sorghum plant for the Fig. 1 flow teaser (user, 2026-11-14: "Pick one best sorghum plant where leave number
matches with ground truth"; the earlier one, Sorghum_10113, is written with 22 leaves against 21).

Candidates: the sorghum test rows at yaw 0 (row mode, all three streams) of the row fine-tune's test report, with
15-40 % of the target hidden (the occlusion should be visible), ranked by recipe error (plant MAE + leaf MAE); the best 60
are re-run with the teaser's protocol (RowDataset seed 0, eval_row.predict 'all', seed 0) and kept if the written leaf
count equals the true one and the height is within 2 %. Prints them ranked by leaf MAE and writes the list to
/work/mech-ai/alloydas/embodiedmae_outputs/_fig12/teaser_sorghum_candidates.json.
"""
import sys, pathlib, json, csv
R = pathlib.Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R), str(R / 'figures'), str(R / 'eval')]
import numpy as np, torch
import _plantviz as V
from row_dataset import RowDataset
from eval_row import build_model, predict, recipe_error, species_info, DEFAULT_ROOT

REP = R / 'reports/row_sorghum_sorghum_row_ft_checkpoint_epoch_100_test_17082628.csv'
OUT = pathlib.Path('/work/mech-ai/alloydas/embodiedmae_outputs/_fig12/teaser_sorghum_candidates.json')

if __name__ == '__main__':
    torch.set_num_threads(8)
    rows = [r for r in csv.DictReader(open(REP)) if r['mode'] == 'row' and r['source'] == 'all' and int(r['theta']) == 0
            and 0.15 <= float(r['occ_frac']) <= 0.40]
    rows.sort(key=lambda r: float(r['plant_mae']) + float(r['leaf_mae']))
    model, cfg, _ = build_model(R / 'outputs/sorghum_row_ft', 'checkpoints/checkpoint_epoch_100.pth', 'cpu', 'sorghum')
    gain = float(cfg.get('rgb_brighten_gain', 1.0)); info_sp = species_info('sorghum'); out = []
    for r in rows[:60]:
        ds = RowDataset(DEFAULT_ROOT['sorghum'], 'test', 'sorghum', 'row', cfg.get('img_size', 224), cfg['num_points'], cfg['max_leaves'],
                        thetas=[0], rgb_gain=gain, gt_points=cfg['gt_points'], plants=[r['plant']])
        np.random.seed(0)
        rgb, depth, pc, params, valid, g_rgb, g_depth, g_pc, info = ds[0]
        with torch.no_grad():
            _, ppar = predict(model, 'all', rgb[None], depth[None], pc[None], params[None], float(cfg.get('mask_ratio', 0.8)), 0)
        pm, lm = recipe_error(ppar, params[None], valid[None], info_sp)
        rt = V.decode_recipe('sorghum', params.numpy(), valid.numpy()); rp = V.decode_recipe('sorghum', ppar[0].clamp(0, 1).numpy(), valid.numpy(), predicted=True)
        e = dict(plant=r['plant'], occ=float(info['occ_frac']), n_true=rt['n'], n_pred=rp['n'], h_true=rt['height'], h_pred=rp['height'],
                 h_err=abs(rp['height'] - rt['height']) / rt['height'], leaf_mae=float(lm[0]), plant_mae=float(pm[0]))
        print(e, flush=True)
        if e['n_pred'] == e['n_true'] and e['h_err'] <= 0.02: out.append(e)
    out.sort(key=lambda e: e['leaf_mae'])
    json.dump(out, open(OUT, 'w'), indent=1)
    print('KEPT', len(out)); [print(e) for e in out[:10]]
