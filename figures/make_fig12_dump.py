"""Raw arrays for Figures 1 and 2 (second version), dumped once so the layout can be iterated without a model.

Writes <out>/v2_dump.pkl (default /work/mech-ai/alloydas/embodiedmae_outputs/_fig12/), a dict:
  rows[species][plant]   one held-out TEST row, seen broadside (yaw 0): the row as the model reads it (image, depth
                         image, scene point cloud), the center plant alone (image, depth, cloud, recipe), the pixel
                         masks (target / neighbor / hidden target), and the center plant written by
                           ft:   the row fine-tune from all three streams (80 % of tokens hidden, the Regime III
                                 protocol of eval/eval_row.py) and from the image, the depth image or the cloud ALONE,
                           iso:  the isolated-trained control from all three streams of the row,
                           head: the headline model, no row fine-tuning, from all three streams,
                         each with its cloud, recipe and Chamfer against the true center plant (x1e-4 m^2).
  e10[key]               held-out real-plant row stops (eval/real_rows.py): inputs, the true center plant, and the
                         prediction of the headline model, the row fine-tune and the real-row fine-tune.
  stage[species]         one held-out single plant: the four streams, the headline model's pretraining pass at mask
                         ratio 0.8 (token masks, the point-cloud token centers, reconstructions), and the cloud /
                         depth / recipe generated from ONE stream alone before distillation (headline model) and
                         after it (the distilled generalist).
CPU, about 10 minutes.   usage: python figures/make_fig12_dump.py [--out DIR] [--only rows,e10,stage]
"""
import sys, pathlib; R = pathlib.Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R), str(R / 'figures'), str(R / 'eval')]
import argparse, csv, json, pickle, time
import numpy as np, torch
from PIL import Image
import _plantviz as V

ap = argparse.ArgumentParser()
ap.add_argument('--out', default='/work/mech-ai/alloydas/embodiedmae_outputs/_fig12')
ap.add_argument('--only', default='rows,e10,stage')
ap.add_argument('--alternates', type=int, default=3, help='extra row plants per species near the median Chamfer')
a = ap.parse_args()
OUT = pathlib.Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(8)
T0 = time.time()
log = lambda *s: print(f'[{time.time() - T0:6.0f}s]', *s, flush=True)
npf = lambda t: t.detach().float().cpu().numpy()

ROW = {'sorghum': dict(default='Sorghum_10113', ft='sorghum_row_ft', iso='sorghum_row_iso', head='4m_pretrain_15k_d2_sink',
                       head_ckpt='checkpoints/checkpoint_epoch_1000.pth',
                       report='reports/row_sorghum_sorghum_row_ft_checkpoint_epoch_100_test_17082628.csv'),
       'maize': dict(default='plant_0106', ft='maize_row_ft_e1', iso='maize_row_iso_e1', head='maize_4m_pretrain_15k_d2',
                     head_ckpt='checkpoints/checkpoint_epoch_1000.pth',
                     report='reports/row_maize_maize_row_ft_e1_checkpoint_epoch_100_test_17189700.csv')}
SINGLE = {'sorghum': ['Sorghum_10113_04', 'Sorghum_10113_05', 'Sorghum_10113_03', 'Sorghum_10036_04', 'Sorghum_10036_00'],   # the row plants seen alone
          'maize': ['plant_0106_00', 'plant_0106_01', 'plant_0106_02', 'plant_0106_03', 'plant_0106_04', 'plant_0043_00']}
E10 = dict(report='reports/e10_real_ft_test_17383192.csv', real='maize_real_ft:best_model.pth',
           row='maize_row_ft_e1:checkpoints/checkpoint_epoch_100.pth', head='maize_4m_pretrain_15k_d2:checkpoints/checkpoint_epoch_1000.pth',
           fixed=[('parallel', '0019', 17), ('parallel', '0148', 14), ('perpendicular', '0137', 4)])


def recipe_pack(sp, par, tv, predicted):
    rec = V.decode_recipe(sp, par, tv, predicted=predicted)
    return dict(height=float(rec['height']), n=int(rec['n']), leaves=[{k: float(v) for k, v in l.items()} for l in rec['leaves']])


def rgba448(path, size=448):
    return np.asarray(Image.open(path).convert('RGBA').resize((size, size), Image.LANCZOS))


# ====================================================================== rows (Regime III)
def dump_rows():
    from row_dataset import RowDataset
    from eval_row import build_model, predict, DEFAULT_ROOT
    out = {}
    for sp, c in ROW.items():
        rows = [r for r in csv.DictReader(open(R / c['report'])) if r['mode'] == 'row' and r['source'] == 'all' and int(r['theta']) == 0]
        ch = np.array([float(r['chamfer_gt_m2']) for r in rows]); occ = np.array([float(r['occ_frac']) for r in rows])
        med, omed = np.median(ch), np.median(occ)
        order = np.argsort(np.abs(ch - med))
        alts = [rows[i]['plant'] for i in order if occ[i] >= omed and rows[i]['plant'] != c['default']][:a.alternates]
        plants = [c['default']] + alts
        log(sp, 'row plants', plants, f'median Chamfer {med * 1e4:.1f}e-4 m2, median occlusion {omed:.2f}')
        models = {}
        for tag, run, ck in (('ft', c['ft'], 'checkpoints/checkpoint_epoch_100.pth'), ('iso', c['iso'], 'checkpoints/checkpoint_epoch_100.pth'),
                             ('head', c['head'], c['head_ckpt'])):
            models[tag] = build_model(R / 'outputs' / run, ck, 'cpu', sp); log(sp, 'loaded', tag, run)
        cfg = models['ft'][1]
        gain = float(cfg.get('rgb_brighten_gain', 1.0)) if sp == 'sorghum' else None
        out[sp] = {}
        for pl in plants:
            ds = RowDataset(DEFAULT_ROOT[sp], 'test', sp, 'row', cfg.get('img_size', 224), cfg['num_points'], cfg['max_leaves'],
                            thetas=[0], rgb_gain=gain, gt_points=cfg['gt_points'], plants=[pl])
            np.random.seed(0)                                   # the loader subsamples the clouds with np.random
            rgb, depth, pc, params, valid, g_rgb, g_depth, g_pc, info = ds[0]
            shot = V.row_shot_dir(sp, 'test', pl, 0)
            tgt, nbr, hid = V.row_masks(sp, shot, 448)
            d = dict(plant=pl, species=sp, occ=float(info['occ_frac']), nbr=float(info['nbr_frac']), scale=float(info['scale']),
                     rgb=V.tensor_rgb(rgb).astype(np.float32), depth=npf(depth[0]), scene=npf(pc),
                     gt_rgb=V.tensor_rgb(g_rgb).astype(np.float32), gt_depth=npf(g_depth[0]), gt_pc=npf(g_pc),
                     rgba=rgba448(shot / 'rgba.png'), gt_rgba=rgba448(shot / 'gt_rgba.png'),
                     depth448=V.decode_depth_png(shot / 'depth.png', 448).astype(np.float32),
                     gt_depth448=V.decode_depth_png(shot / 'gt_depth.png', 448).astype(np.float32),
                     mask_target=tgt, mask_neighbor=nbr, mask_hidden=hid, gain=gain,
                     recipe_true=recipe_pack(sp, npf(params), npf(valid), False), par_true=npf(params), valid=npf(valid), pred={})
            for tag, (model, mcfg, ep) in models.items():
                for src in (('all', 'rgb', 'depth', 'pc') if tag == 'ft' else ('all',)):
                    with torch.no_grad():
                        ppc, ppar = predict(model, src, rgb[None], depth[None], pc[None], params[None], float(mcfg.get('mask_ratio', 0.8)), 0)
                    P = ppc[0].float(); par_p = npf(ppar[0].clamp(0, 1))
                    ch_n = V.chamfer(P, g_pc)
                    d['pred'][f'{tag}:{src}'] = dict(pc=npf(P), par=par_p, chamfer=ch_n, chamfer_m2e4=ch_n * d['scale'] ** 2 * 1e4,
                                                    chamfer_in_m2e4=V.chamfer(P, pc) * d['scale'] ** 2 * 1e4,
                                                    recipe=recipe_pack(sp, par_p, npf(valid), True), epoch=ep)
            out[sp][pl] = d
            log(sp, pl, f"occ {d['occ']:.2f}", {k: round(v['chamfer_m2e4'], 1) for k, v in d['pred'].items()},
                'height', {k: round(v['recipe']['height'], 2) for k, v in d['pred'].items()}, 'true', round(d['recipe_true']['height'], 2),
                'leaves', {k: v['recipe']['n'] for k, v in d['pred'].items()}, 'true', d['recipe_true']['n'])
        del models
    return out


# ====================================================================== E10: real plants in rows
def dump_e10():
    from eval_row import build_model, predict
    from real_rows import RealRowDataset
    from embodied_mae_4m_maize import _PLANT_SCALE_V2, _PLANT_SHIFT_V2, PLANT_FIELDS_V2
    H_IDX, N_IDX = PLANT_FIELDS_V2.index('height'), 0
    rows = [r for r in csv.DictReader(open(R / E10['report'])) if r['source'] == 'all']
    by = {}
    for r in rows: by.setdefault((r['layout'], r['row'], int(r['stop'])), {})[r['model']] = r
    picks = list(E10['fixed'])
    for layout in ('parallel', 'perpendicular'):                # the stop closest to the median before AND after
        keys = [k for k in by if k[0] == layout and 1 < k[2] < 18 and len(by[k]) == 2]
        b = np.array([float(by[k]['maize_row_ft_e1']['chamfer_m2']) for k in keys]); f = np.array([float(by[k]['maize_real_ft']['chamfer_m2']) for k in keys])
        he = np.array([float(by[k]['maize_real_ft']['height_rel_err']) for k in keys])
        score = np.abs(b / np.median(b) - 1) + np.abs(f / np.median(f) - 1) + np.abs(he - np.median(he)) / max(np.median(he), 1e-6) * 0.5
        k = keys[int(np.argmin(score))]
        log('E10 median stop', layout, k, f'before {b[np.argmin(score)] * 1e4:.1f} after {f[np.argmin(score)] * 1e4:.1f} (medians {np.median(b) * 1e4:.1f} / {np.median(f) * 1e4:.1f})')
        if k not in picks: picks.append(k)
    models = {}
    for tag in ('head', 'row', 'real'):
        run, ck = E10[tag].split(':', 1); m, cfg, _ = build_model(R / 'outputs' / run, ck, 'cpu', 'maize')
        models[tag] = (m, float(cfg.get('mask_ratio', 0.8))); log('E10 loaded', tag, run)
    out = {}
    for i, (layout, row, stop) in enumerate(picks):
        ds = RealRowDataset(layouts=(layout,), rows=[row], stops={stop}, seed=0, species='maize')
        rgb, dep, pin, par, valid, pgt, info = ds[0]
        d = dict(layout=layout, row=row, stop=stop, info={k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in info.items()},
                 rgb=(npf(rgb).transpose(1, 2, 0) * V.IMNET_STD + V.IMNET_MEAN).clip(0, 1).astype(np.float32), depth=npf(dep[0]),
                 scene=npf(pin), gt_pc=npf(pgt), report={m: {k: by[(layout, row, stop)][m][k] for k in ('chamfer_m2', 'height_pred', 'leaves_pred', 'height_rel_err')}
                                                       for m in by.get((layout, row, stop), {})}, pred={})
        for tag, (m, mr) in models.items():
            with torch.no_grad():
                pp, ppar = predict(m, 'all', rgb[None], dep[None], pin[None], par[None], mr, 1000003 * i)
            P = pp[0].float(); p0 = npf(ppar[0, 0].clamp(0, 1))
            par_p = npf(ppar[0].clamp(0, 1))
            d['pred'][tag] = dict(pc=npf(P), chamfer_m2e4=V.chamfer(P, pgt) * info['scale'] ** 2 * 1e4, par=par_p,
                                  recipe=recipe_pack('maize', par_p, None, True),
                                  height=float(p0[H_IDX] * _PLANT_SCALE_V2[H_IDX] - _PLANT_SHIFT_V2[H_IDX]),
                                  leaves=float(p0[N_IDX] * _PLANT_SCALE_V2[N_IDX] - _PLANT_SHIFT_V2[N_IDX]))
        out[f'{layout}:{row}:{stop}'] = d
        log('E10', layout, row, stop, f"true height {info['height']:.2f} leaves {info['leaves']} occ {info['occ_frac']:.2f}",
            {k: (round(v['chamfer_m2e4'], 1), round(v['height'], 2), round(v['leaves'], 1)) for k, v in d['pred'].items()})
    return out


# ====================================================================== single plants: Stage 1 and Stage 2
def dump_stage():
    from train_sorghum_4m import unpatchify, _unnorm_pix
    out = {}
    for sp, names in SINGLE.items():
      ds = V.dataset(sp, 'test'); out[sp] = {}
      head, cfg, ep = V.load_model(sp, 'headline'); log(sp, 'loaded headline', ep)
      gen, gcfg, gep = V.load_model(sp, 'generalist'); log(sp, 'loaded generalist', gep)
      for name in names:
        item = V.item_by_name(ds, name, seed=0)
        rgb, depth, pc, par, tv = (item[k][None] for k in range(5))
        rec = {}
        orig_fps = head.pc_embed.fps
        def fps(xyz, n, _o=orig_fps, _r=rec):
            idx = _o(xyz, n); _r['idx'] = idx.detach().clone(); return idx
        head.pc_embed.fps = fps
        torch.manual_seed(3)
        with torch.no_grad():
            _, _, (pr, pd_, ppc, ppar), (mr, md, mp, mt) = head(rgb, depth, pc, par, tv, mask_ratio=0.8)
        cent = npf(pc[0])[rec['idx'][0].numpy()]
        d = dict(plant=name, species=sp, rgb=V.tensor_rgb(rgb[0]).astype(np.float32), depth=npf(depth[0, 0]), pc=npf(pc[0]),
                 par=npf(par[0]), tv=npf(tv[0]), recipe_true=recipe_pack(sp, npf(par[0]), npf(tv[0]), False),
                 mask=dict(rgb=npf(mr[0]), depth=npf(md[0]), pc=npf(mp[0]), text=npf(mt[0])), pc_centers=cent,
                 rec=dict(rgb=V.tensor_rgb(unpatchify(_unnorm_pix(head, pr, rgb), head.patch_size, 3, head.img_size)[0]).astype(np.float32),
                          depth=npf(unpatchify(pd_, head.patch_size, 1, head.img_size)[0, 0]), pc=npf(ppc[0]), par=npf(ppar[0].clamp(0, 1)),
                          chamfer=V.chamfer(ppc[0], pc[0])),
                 visible={k: int((1 - m[0]).sum()) for k, m in (('rgb', mr), ('depth', md), ('pc', mp), ('text', mt))}, epoch=ep)
        d['rec']['recipe'] = recipe_pack(sp, d['rec']['par'], d['tv'], False)       # the real leaf slots, hidden ones predicted
        head.pc_embed.fps = orig_fps
        # the teacher's view in distillation: all four streams (sorghum: every token; maize: half of each stream)
        torch.manual_seed(5)
        with torch.no_grad():
            _, _, _, (tr, td, tp, tt) = head(rgb, depth, pc, par, tv, visible=['rgb', 'depth', 'pc', 'text'], source_mask_ratio=0.0 if sp == 'sorghum' else 0.5)
        d['teacher_mask'] = dict(rgb=npf(tr[0]), depth=npf(td[0]), pc=npf(tp[0]), text=npf(tt[0]))
        d['one'] = {}
        for src in ('rgb', 'depth', 'pc', 'text'):
            for tag, m in (('before', head), ('after', gen)):
                g = V.generate(m, item, [src], seed=0)
                par_g = np.asarray(g['par'])
                d['one'][f'{src}:{tag}'] = dict(pc=np.asarray(g['pc'], dtype=np.float32), depth=np.asarray(g['depth'], dtype=np.float32),
                                               rgb=np.asarray(g['rgb'], dtype=np.float32), par=par_g, chamfer=V.chamfer(g['pc'], pc[0]),
                                               recipe=recipe_pack(sp, par_g, d['tv'], True))
        out[sp][name] = d
        log(sp, name, 'visible', d['visible'], f"rec chamfer {d['rec']['chamfer'] * 1e3:.2f}e-3",
            {k: round(v['chamfer'] * 1e3, 2) for k, v in d['one'].items()})
      del head, gen
    return out


if __name__ == '__main__':
    path = OUT / 'v2_dump.pkl'
    D = pickle.load(open(path, 'rb')) if path.exists() else {}
    for part, fn in (('rows', dump_rows), ('e10', dump_e10), ('stage', dump_stage)):
        if part in a.only.split(','):
            D[part] = fn()
            pickle.dump(D, open(str(path) + '.tmp', 'wb')); pathlib.Path(str(path) + '.tmp').replace(path)
            log('saved', part, f'{path.stat().st_size / 1e6:.0f} MB')
    log('done', path)
