"""Paper Fig. 3 (qualitative, Regime I): one held-out (val) plant per species, generated from the
PHOTOGRAPH ALONE. Columns: photograph | cloud from the headline model (before distillation) |
cloud from the distilled generalist (after) | ground-truth cloud | recovered recipe (per-leaf
length, normalised units, ground truth vs generalist). Chamfer in each cloud title.
Same loading conventions as eval_rgb2pc_quant.py (each run's rgb_file / brighten gain; maize
final data = Maize_1, rgba.png, schema v2). Writes paper/figs/fig_qual.{pdf,png} + Overleaf copy.
usage: python figures/make_qual_fig.py [--sorghum-plant Sorghum_10001] [--maize-plant plant_0003] [--device cpu]
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import argparse, json, shutil
import numpy as np, torch, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from embodied_mae_4m import embodied_mae_4m_base
from embodied_mae_4m_maize import embodied_mae_4m_maize_base
from sorghum_dataset_4m import SorghumDataset4M
from maize_dataset_4m import MaizeDataset4M

ap = argparse.ArgumentParser()
ap.add_argument('--sorghum-plant', default='Sorghum_10001'); ap.add_argument('--maize-plant', default='plant_0003')
ap.add_argument('--device', default='cpu'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--ugg', action='store_true', help='UGG-style PREVIEW only (figures/ugg_grids.py): writes '
                'paper/figs/ugg_preview/fig_qual_ugg.{pdf,png}, never the paper copy; the generated clouds are cached')
ap.add_argument('--refresh', action='store_true', help='with --ugg: regenerate the cached clouds')
a = ap.parse_args()
O = pathlib.Path('outputs')
SPEC = {
    'sorghum': dict(root='/work/mech-ai-scratch/alloy/shorgum_data/new_data_50K/Sorghum_15K', plant=a.sorghum_plant,
                    before=O / '4m_pretrain_15k_d2_sink/checkpoints/checkpoint_epoch_1000.pth',
                    after=O / '4m_distill_15k_all_d2/best_model.pth', build=embodied_mae_4m_base, pts=8196, len_idx=1),
    'maize': dict(root='/work/mech-ai-scratch/alloy/Maize_1', plant=a.maize_plant,
                  before=O / 'maize_4m_pretrain_15k_d2/checkpoints/checkpoint_epoch_1000.pth',
                  after=O / 'maize_distill_e1_d2/best_model.pth', build=embodied_mae_4m_maize_base, pts=8192, len_idx=1),
}


def run_cfg(ck):
    for d in (ck.parent, ck.parent.parent):
        if (d / 'config.json').exists(): return json.load(open(d / 'config.json'))
    return {}


def load(spec, ck):
    m = spec['build'](target_points=spec['pts']).to(a.device).eval()
    st = torch.load(ck, map_location='cpu', weights_only=False)
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in st.get('model_state_dict', st).items()}
    miss, unexp = m.load_state_dict(sd, strict=False); assert not miss and not unexp, (ck, len(miss), len(unexp))
    return m, int(st.get('epoch', -1))


@torch.no_grad()
def gen(m, it):
    rgb, depth, pc, par, tv = (it[k] for k in range(5))
    torch.manual_seed(a.seed)
    _, _, (_, _, ppc, ppar), _ = m(rgb[None].to(a.device), depth[None].to(a.device), pc[None].to(a.device),
                                   torch.zeros_like(par)[None].to(a.device), tv[None].to(a.device), visible=['rgb'])
    return ppc[0].float().cpu(), ppar[0].float().cpu()


def cham(p, g):
    d = torch.cdist(p[None], g[None])[0]
    return float(d.min(1).values.pow(2).mean() + d.min(0).values.pow(2).mean())


def scat(ax, P, col, title, lim):
    P = P.numpy(); c, h = lim
    ax.scatter(P[:, 0], P[:, 2], P[:, 1], s=0.5, c=col, alpha=0.8, linewidths=0)
    ax.view_init(elev=15, azim=-40); ax.set_title(title, fontsize=7.5)
    ax.set_xlim(c[0] - h, c[0] + h); ax.set_ylim(c[2] - h, c[2] + h); ax.set_zlim(c[1] - h, c[1] + h)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()


if a.ugg:                                             # preview variant; the paper figure below is untouched
    import pickle
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); import ugg_grids
    dump = ugg_grids.DUMPS / 'fig_qual_dump.pkl'
    D = pickle.load(open(dump, 'rb')) if dump.exists() and not a.refresh else None
    if D is None or D['plants'] != {sp: s['plant'] for sp, s in SPEC.items()} or D['seed'] != a.seed:
        MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
        D = dict(plants={sp: s['plant'] for sp, s in SPEC.items()}, seed=a.seed, species={})
        for sp, s in SPEC.items():                    # the same steps, in the same order, as the paper figure
            cfg = run_cfg(s['after'])
            if sp == 'sorghum':
                ds = SorghumDataset4M(s['root'], split='val', num_points=s['pts'], rgb_file=cfg.get('rgb_file', 'rgb_nobg.png'),
                                      rgb_gain=cfg.get('rgb_brighten_gain'))
            else:
                ds = MaizeDataset4M(s['root'], split='val', num_points=s['pts'], rgb_file='rgba.png', param_schema='v2')
            idx = next(i for i, f in enumerate(ds.samples) if f.name == f"{s['plant']}_00")
            np.random.seed(a.seed)
            it = ds[idx]
            gt, par, tv = it[2], it[3], it[4]
            mb, eb = load(s, s['before']); ma, ea = load(s, s['after'])
            pb, _ = gen(mb, it); pa, para = gen(ma, it)
            n = int(tv[1:].sum())
            pl, pal = par[1:1 + n, s['len_idx']], para[1:1 + n, s['len_idx']].clamp(0, 1)
            D['species'][sp] = dict(plant=s['plant'], img=np.clip(it[0].permute(1, 2, 0).numpy() * STD + MEAN, 0, 1),
                                    pb=pb.numpy(), pa=pa.numpy(), gt=gt.numpy(), ch_b=cham(pb, gt), ch_a=cham(pa, gt),
                                    par_len=pl.numpy(), para_len=pal.numpy(), mae=float((pal - pl).abs().mean()), epochs=(eb, ea))
            print(sp, s['plant'], 'before', round(D['species'][sp]['ch_b'] * 1e3, 2), 'after', round(D['species'][sp]['ch_a'] * 1e3, 2),
                  'leaf-length MAE', round(D['species'][sp]['mae'], 3), 'epochs', eb, ea, flush=True)
        dump.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(D, open(dump.with_suffix('.tmp'), 'wb')); dump.with_suffix('.tmp').replace(dump)
    ugg_grids.render_qual(D)
    sys.exit(0)

fig = plt.figure(figsize=(11, 4.6))
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
for r, (sp, s) in enumerate(SPEC.items()):
    cfg = run_cfg(s['after'])
    if sp == 'sorghum':
        ds = SorghumDataset4M(s['root'], split='val', num_points=s['pts'], rgb_file=cfg.get('rgb_file', 'rgb_nobg.png'),
                              rgb_gain=cfg.get('rgb_brighten_gain'))
    else:
        ds = MaizeDataset4M(s['root'], split='val', num_points=s['pts'], rgb_file='rgba.png', param_schema='v2')
    idx = next(i for i, f in enumerate(ds.samples) if f.name == f"{s['plant']}_00")
    np.random.seed(a.seed)          # the loader subsamples the cloud with np.random
    it = ds[idx]
    gt, par, tv = it[2], it[3], it[4]
    mb, eb = load(s, s['before']); ma, ea = load(s, s['after'])
    pb, _ = gen(mb, it); pa, para = gen(ma, it)
    lo, hi = gt.min(0).values.numpy(), gt.max(0).values.numpy(); lim = ((lo + hi) / 2, (hi - lo).max() / 2 * 1.05)
    img = np.clip(it[0].permute(1, 2, 0).numpy() * STD + MEAN, 0, 1)
    ax = fig.add_subplot(2, 5, 5 * r + 1); ax.imshow(img); ax.set_axis_off()
    ax.set_title(f'{sp} (val plant {s["plant"].split("_")[-1]})\ninput photograph', fontsize=7.5)
    scat(fig.add_subplot(2, 5, 5 * r + 2, projection='3d'), pb, '#9a6a00', f'headline model (before)\nChamfer {cham(pb, gt) * 1e3:.2f}e-3', lim)
    scat(fig.add_subplot(2, 5, 5 * r + 3, projection='3d'), pa, '#1f6f8b', f'distilled generalist (after)\nChamfer {cham(pa, gt) * 1e3:.2f}e-3', lim)
    scat(fig.add_subplot(2, 5, 5 * r + 4, projection='3d'), gt, '#2f6b2a', 'ground-truth cloud', lim)
    ax = fig.add_subplot(2, 5, 5 * r + 5)
    n = int(tv[1:].sum()); k = np.arange(1, n + 1)
    ax.plot(k, par[1:1 + n, s['len_idx']].numpy(), 'o-', ms=2.5, lw=1, color='#2f6b2a', label='ground truth')
    ax.plot(k, para[1:1 + n, s['len_idx']].clamp(0, 1).numpy(), 's--', ms=2.5, lw=1, color='#1f6f8b', label='from the photograph')
    mae = float((para[1:1 + n, s['len_idx']].clamp(0, 1) - par[1:1 + n, s['len_idx']]).abs().mean())
    ax.set_title(f'recipe: leaf length per leaf\n(normalised; MAE {mae:.3f})', fontsize=7.5)
    ax.set_xlabel('leaf index (base to top)', fontsize=7); ax.tick_params(labelsize=6); ax.legend(fontsize=6, frameon=False)
    print(sp, s['plant'], 'before', round(cham(pb, gt) * 1e3, 2), 'after', round(cham(pa, gt) * 1e3, 2), 'leaf-length MAE', round(mae, 3), 'epochs', eb, ea)
fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.09, wspace=0.08, hspace=0.55)
out = pathlib.Path('paper/figs')
for ext in ('pdf', 'png'): fig.savefig(out / f'fig_qual.{ext}', dpi=200)
shutil.copy(out / 'fig_qual.pdf', 'paper/overleaf/Figures/fig_qual.pdf')
print('wrote', out / 'fig_qual.pdf')
