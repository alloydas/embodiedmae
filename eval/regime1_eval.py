"""Regime I table (paper tab:regime1): isolated plants, one view, final data releases.

For each species, the distilled generalist (best_model.pth) given ONE input alone (rgb / depth / pc,
the other streams and the recipe masked), and the headline model (E1, epoch 1000) given rgb+depth+pc
(the teacher row, scored in its masked regime: 80 % of its visible tokens hidden), on the val plants, view 00 (deterministic, as the probe). Per input it reports:
  chamfer   symmetric squared Chamfer of the generated cloud vs the plant cloud, unit-sphere frame (x1e-3)
  f1        F1 at tau 0.01 (precision: predicted points within tau of the cloud; recall: the reverse)
  height    relative error of plant height decoded from the predicted recipe's plant token
            (sorghum: stem_length; maize: height)
  leaves    accuracy within +-1 leaf of the leaf count (maize: the plant token's leafCount; sorghum has no
            leaf-count field, so it is derived from the predicted stem length through the generator's
            rule stem_length = 0.05 n - 0.001, R^2 0.988 on the data)
  skill     mean over leaf fields of 1 - MAE_model / MAE_mean, on real leaf tokens (MAE_mean: predicting
            each field's val mean)
Writes reports/regime1_<job>.csv. CPU smoke: --limit 4 --device cpu.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import argparse, json, os
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
import embodied_mae_4m as S
import embodied_mae_4m_maize as M
from sorghum_dataset_4m import SorghumDataset4M
from maize_dataset_4m import MaizeDataset4M

ap = argparse.ArgumentParser()
ap.add_argument('--species', nargs='+', default=['sorghum', 'maize'])
ap.add_argument('--split', default='val'); ap.add_argument('--limit', type=int, default=None)
ap.add_argument('--device', default='cuda'); ap.add_argument('--batch-size', type=int, default=16)
ap.add_argument('--num-workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--tau', type=float, default=0.01)
ap.add_argument('--out', default=None)
a = ap.parse_args()
O = pathlib.Path('outputs')
SPEC = {
    'sorghum': dict(root='/work/mech-ai-scratch/alloy/shorgum_data/new_data_50K/Sorghum_15K', build=S.embodied_mae_4m_base, pts=8196,
                    gen=O / '4m_distill_15k_all_d2/best_model.pth', teacher=O / '4m_pretrain_15k_d2_sink/checkpoints/checkpoint_epoch_1000.pth',
                    pscale=S._PLANT_SCALE, pshift=S._PLANT_SHIFT, h_idx=0, n_leaf_fields=len(S.LEAF_FIELDS)),
    'maize': dict(root='/work/mech-ai-scratch/alloy/Maize_1', build=M.embodied_mae_4m_maize_base, pts=8192,
                  gen=O / 'maize_distill_e1_d2/best_model.pth', teacher=O / 'maize_4m_pretrain_15k_d2/checkpoints/checkpoint_epoch_1000.pth',
                  pscale=M._PLANT_SCALE_V2, pshift=M._PLANT_SHIFT_V2, h_idx=2, n_leaf_fields=len(M.LEAF_FIELDS_V2)),
}
ROWS = [('gen', 'rgb', ['rgb']), ('gen', 'depth', ['depth']), ('gen', 'pc', ['pc']), ('teacher', 'rgb+depth+pc', ['rgb', 'depth', 'pc'])]


def load(spec, ck):
    m = spec['build'](target_points=spec['pts']).to(a.device).eval()
    st = torch.load(ck, map_location='cpu', weights_only=False)
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in st.get('model_state_dict', st).items()}
    miss, unexp = m.load_state_dict(sd, strict=False); assert not miss and not unexp, (ck, len(miss), len(unexp))
    return m


def cfg_of(ck):
    for d in (ck.parent, ck.parent.parent):
        if (d / 'config.json').exists(): return json.load(open(d / 'config.json'))
    return {}


def pr(pred, gt, tau):
    """per-sample chamfer (squared, symmetric), precision, recall"""
    out = []
    for p, g in zip(pred, gt):
        d = torch.cdist(p[None].float(), g[None].float())[0]
        dp, dg = d.min(1).values, d.min(0).values
        out.append((float(dp.pow(2).mean() + dg.pow(2).mean()), float((dp < tau).float().mean()), float((dg < tau).float().mean())))
    return out


rows = []
for sp in a.species:
    s = SPEC[sp]; cfg = cfg_of(s['gen'])
    if sp == 'sorghum':
        ds = SorghumDataset4M(s['root'], split=a.split, num_points=s['pts'], view_sampling=True, deterministic_view=True,
                              rgb_file=cfg.get('rgb_file', 'rgb_nobg.png'), rgb_gain=cfg.get('rgb_brighten_gain'))
    else:
        ds = MaizeDataset4M(s['root'], split=a.split, num_points=s['pts'], view_sampling=True, deterministic_view=True,
                            rgb_file='rgba.png', param_schema='v2')
    if a.limit: ds.plant_views = ds.plant_views[:a.limit]
    np.random.seed(a.seed)
    models = {'gen': load(s, s['gen']), 'teacher': load(s, s['teacher'])}
    acc = {r[1]: dict(ch=[], p=[], r=[], h_err=[], n_ok=[], lf_err=[]) for r in ROWS}
    lf_gt_all, lf_valid_all = [], []
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers)
    for bi, (rgb, depth, pc, par, tv, names) in enumerate(dl):
        rgb, depth, pc, par, tv = (t.to(a.device) for t in (rgb, depth, pc, par, tv))
        lf_gt_all.append(par[:, 1:, :s['n_leaf_fields']].cpu()); lf_valid_all.append(tv[:, 1:].cpu())
        h_gt = (par[:, 0, s['h_idx']].cpu().numpy() * s['pscale'][s['h_idx']] - s['pshift'][s['h_idx']])
        if sp == 'maize': n_gt = np.rint(par[:, 0, 0].cpu().numpy() * s['pscale'][0])
        else: n_gt = tv[:, 1:].sum(1).cpu().numpy()       # real leaf tokens (plants above 24 leaves are capped at 24)
        for key, lab, vis in ROWS:
            torch.manual_seed(a.seed + bi)
            with torch.no_grad():
                # the headline model is scored in its own masked regime (80 % of the visible tokens hidden,
                # as pc_view_eval / pc_fscore): with every token visible its decoder is off-distribution
                # and the cloud collapses (Chamfer ~5e-3, checked on the smoke). Students: source fully visible.
                _, _, (_, _, ppc, ppar), _ = models[key](rgb, depth, pc, torch.zeros_like(par), tv, visible=vis,
                                                         source_mask_ratio=0.8 if key == 'teacher' else 0.0)
            for c, p_, r_ in pr(ppc, pc, a.tau):
                acc[lab]['ch'].append(c); acc[lab]['p'].append(p_); acc[lab]['r'].append(r_)
            ppar = ppar.clamp(0, 1).float().cpu()
            h = ppar[:, 0, s['h_idx']].numpy() * s['pscale'][s['h_idx']] - s['pshift'][s['h_idx']]
            acc[lab]['h_err'] += list(np.abs(h - h_gt) / np.abs(h_gt))
            if sp == 'maize': n = np.rint(ppar[:, 0, 0].numpy() * s['pscale'][0])
            else: n = np.rint((h + 0.001) / 0.05).clip(max=24)    # generator rule, capped at the 24 leaf slots
            acc[lab]['n_ok'] += list(np.abs(n - n_gt) <= 1)
            acc[lab]['lf_err'].append((ppar[:, 1:, :s['n_leaf_fields']] - par[:, 1:, :s['n_leaf_fields']].cpu()).abs())
        if bi % 20 == 0: print(f'  {sp} {min((bi + 1) * a.batch_size, len(ds))}/{len(ds)}', flush=True)
    G = torch.cat(lf_gt_all); V = torch.cat(lf_valid_all).bool()
    base = []
    for f in range(s['n_leaf_fields']):
        g = G[..., f][V]; base.append(float((g - g.mean()).abs().mean()))
    for key, lab, vis in ROWS:
        E = torch.cat(acc[lab]['lf_err'])
        skills = [1 - float(E[..., f][V].mean()) / base[f] for f in range(s['n_leaf_fields'])]
        P, R = np.mean(acc[lab]['p']), np.mean(acc[lab]['r'])
        f1 = np.mean([2 * p * r / (p + r) if p + r > 0 else 0 for p, r in zip(acc[lab]['p'], acc[lab]['r'])])
        rows.append(dict(species=sp, model=key, input=lab, n=len(acc[lab]['ch']), chamfer_e3=1e3 * np.mean(acc[lab]['ch']),
                         f1=f1, precision=P, recall=R, height_rel_err=np.median(acc[lab]['h_err']),
                         leaf_count_acc1=np.mean(acc[lab]['n_ok']), leaf_skill_mean=np.mean(skills),
                         leaf_skill_fields=' '.join(f'{x:.3f}' for x in skills)))
        print(rows[-1])
out = a.out or f"reports/regime1_{os.environ.get('SLURM_JOB_ID', 'manual')}.csv"
pd.DataFrame(rows).to_csv(out, index=False); print('wrote', out)
