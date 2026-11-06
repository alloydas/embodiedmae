"""Regime I table (paper tab:regime1), the two extra rows, final data, both species, val, all plants.

  rgb3       the distilled generalist reads the photographs of views 00, 01 and 02 separately and the
             three predicted recipes are averaged (normalised space). Recipe columns only: each view's
             cloud is in its own camera frame and the model has no multi-view cloud fusion.
  two_stage  completion then fit: the generalist generates a cloud from the view-00 photograph, and
             that generated cloud is fed back as the point-cloud source to read the recipe. Recipe
             columns from the second stage; Chamfer/F1 of the first-stage cloud (= the RGB row).
Recipe metrics exactly as eval/regime1_eval.py (height relative error, leaf count +-1, mean leaf skill).
Writes reports/regime1_extra_<job>.csv. CPU smoke: --limit 4 --device cpu.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import argparse, json, os, re, collections
import numpy as np, pandas as pd, torch
import embodied_mae_4m as S
import embodied_mae_4m_maize as M
from sorghum_dataset_4m import SorghumDataset4M
from maize_dataset_4m import MaizeDataset4M

ap = argparse.ArgumentParser()
ap.add_argument('--species', nargs='+', default=['sorghum', 'maize'])
ap.add_argument('--split', default='val'); ap.add_argument('--limit', type=int, default=None)
ap.add_argument('--device', default='cuda'); ap.add_argument('--batch-size', type=int, default=16)
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--tau', type=float, default=0.01)
ap.add_argument('--out', default=None)
a = ap.parse_args()
O = pathlib.Path('outputs')
SPEC = {
    'sorghum': dict(root='/work/mech-ai-scratch/alloy/shorgum_data/new_data_50K/Sorghum_15K', build=S.embodied_mae_4m_base, pts=8196,
                    gen=O / '4m_distill_15k_all_d2/best_model.pth', pscale=S._PLANT_SCALE, pshift=S._PLANT_SHIFT, h_idx=0,
                    nlf=len(S.LEAF_FIELDS), plant=lambda n: re.match(r'(Sorghum_\d+)_(\d+)$', n)),
    'maize': dict(root='/work/mech-ai-scratch/alloy/Maize_1', build=M.embodied_mae_4m_maize_base, pts=8192,
                  gen=O / 'maize_distill_e1_d2/best_model.pth', pscale=M._PLANT_SCALE_V2, pshift=M._PLANT_SHIFT_V2, h_idx=2,
                  nlf=len(M.LEAF_FIELDS_V2), plant=lambda n: re.match(r'(plant_\d+)_(\d+)$', n)),
}


def load(spec):
    m = spec['build'](target_points=spec['pts']).to(a.device).eval()
    st = torch.load(spec['gen'], map_location='cpu', weights_only=False)
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in st.get('model_state_dict', st).items()}
    miss, unexp = m.load_state_dict(sd, strict=False); assert not miss and not unexp
    return m


def cfg_of(ck):
    for d in (ck.parent, ck.parent.parent):
        if (d / 'config.json').exists(): return json.load(open(d / 'config.json'))
    return {}


@torch.no_grad()
def run(m, rgb, depth, pc, par, tv, vis, seed):
    torch.manual_seed(seed)
    _, _, (_, _, ppc, ppar), _ = m(rgb, depth, pc, torch.zeros_like(par), tv, visible=vis)
    return ppc.float(), ppar.clamp(0, 1).float()


rows = []
for sp in a.species:
    s = SPEC[sp]; cfg = cfg_of(s['gen'])
    if sp == 'sorghum':
        ds = SorghumDataset4M(s['root'], split=a.split, num_points=s['pts'], view_sampling=False,
                              rgb_file=cfg.get('rgb_file', 'rgb_nobg.png'), rgb_gain=cfg.get('rgb_brighten_gain'))
    else:
        ds = MaizeDataset4M(s['root'], split=a.split, num_points=s['pts'], view_sampling=False, rgb_file='rgba.png', param_schema='v2')
    by = collections.defaultdict(dict)
    for i, f in enumerate(ds.samples):
        mm = s['plant'](f.name)
        if mm: by[mm.group(1)][int(mm.group(2))] = i
    plants = sorted(p for p in by if all(v in by[p] for v in (0, 1, 2)))
    if a.limit: plants = plants[:a.limit]
    m = load(s); np.random.seed(a.seed)
    acc = {k: dict(h=[], n=[], lf=[], ch=[], p=[], r=[]) for k in ('rgb3', 'two_stage')}
    G, V = [], []
    for bi in range(0, len(plants), a.batch_size):
        chunk = plants[bi:bi + a.batch_size]
        items = {v: [ds[by[p][v]] for p in chunk] for v in (0, 1, 2)}
        T = lambda v, k: torch.stack([it[k] for it in items[v]]).to(a.device)
        rgb0, depth0, pc0, par, tv = T(0, 0), T(0, 1), T(0, 2), T(0, 3), T(0, 4)
        G.append(par[:, 1:, :s['nlf']].cpu()); V.append(tv[:, 1:].cpu())
        h_gt = par[:, 0, s['h_idx']].cpu().numpy() * s['pscale'][s['h_idx']] - s['pshift'][s['h_idx']]
        n_gt = np.rint(par[:, 0, 0].cpu().numpy() * s['pscale'][0]) if sp == 'maize' else tv[:, 1:].sum(1).cpu().numpy()
        # rgb3: average the recipe predicted from each of three photographs
        pars = [run(m, T(v, 0), T(v, 1), T(v, 2), T(v, 3), T(v, 4), ['rgb'], a.seed + bi)[1] for v in (0, 1, 2)]
        out = {'rgb3': torch.stack(pars).mean(0)}
        # two-stage: photograph -> cloud -> recipe from that generated cloud
        gpc, _ = run(m, rgb0, depth0, pc0, par, tv, ['rgb'], a.seed + bi)
        idx = torch.stack([torch.randperm(gpc.shape[1], device=gpc.device)[:pc0.shape[1]] if gpc.shape[1] >= pc0.shape[1]
                           else torch.randint(0, gpc.shape[1], (pc0.shape[1],), device=gpc.device) for _ in range(gpc.shape[0])])
        gin = torch.gather(gpc, 1, idx[..., None].expand(-1, -1, 3))
        out['two_stage'] = run(m, rgb0, depth0, gin, par, tv, ['pc'], a.seed + bi)[1]
        for p_, g_ in zip(gpc, pc0):
            d = torch.cdist(p_[None], g_[None].float())[0]; dp, dg = d.min(1).values, d.min(0).values
            acc['two_stage']['ch'].append(float(dp.pow(2).mean() + dg.pow(2).mean()))
            acc['two_stage']['p'].append(float((dp < a.tau).float().mean())); acc['two_stage']['r'].append(float((dg < a.tau).float().mean()))
        for k, pp in out.items():
            pp = pp.cpu()
            h = pp[:, 0, s['h_idx']].numpy() * s['pscale'][s['h_idx']] - s['pshift'][s['h_idx']]
            acc[k]['h'] += list(np.abs(h - h_gt) / np.abs(h_gt))
            n = np.rint(pp[:, 0, 0].numpy() * s['pscale'][0]) if sp == 'maize' else np.rint((h + 0.001) / 0.05).clip(max=24)
            acc[k]['n'] += list(np.abs(n - n_gt) <= 1)
            acc[k]['lf'].append((pp[:, 1:, :s['nlf']] - par[:, 1:, :s['nlf']].cpu()).abs())
        print(f'  {sp} {min(bi + a.batch_size, len(plants))}/{len(plants)}', flush=True)
    Gc = torch.cat(G); Vc = torch.cat(V).bool()
    base = [float((Gc[..., f][Vc] - Gc[..., f][Vc].mean()).abs().mean()) for f in range(s['nlf'])]
    for k in acc:
        E = torch.cat(acc[k]['lf']); sk = [1 - float(E[..., f][Vc].mean()) / base[f] for f in range(s['nlf'])]
        row = dict(species=sp, input=k, n=len(acc[k]['h']), height_rel_err=float(np.median(acc[k]['h'])),
                   leaf_count_acc1=float(np.mean(acc[k]['n'])), leaf_skill_mean=float(np.mean(sk)), leaf_skill_fields=' '.join(f'{x:.3f}' for x in sk))
        if acc[k]['ch']:
            row.update(chamfer_e3=1e3 * np.mean(acc[k]['ch']),
                       f1=float(np.mean([2 * p * r / (p + r) if p + r > 0 else 0 for p, r in zip(acc[k]['p'], acc[k]['r'])])))
        rows.append(row); print(row)
out = a.out or f"reports/regime1_extra_{os.environ.get('SLURM_JOB_ID', 'manual')}.csv"
pd.DataFrame(rows).to_csv(out, index=False); print('wrote', out)
