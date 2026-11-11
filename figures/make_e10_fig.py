"""E10 figure: real maize rows (eval/real_rows.py) in, the centre plant out.

Rows of the figure are camera stops picked from the E10 report (eval/eval_real_rows.py): per layout, the stops of
two different real rows whose row fine-tune Chamfer (all streams in) is closest to that layout's median, so the
panels show a typical case, not the best one. Columns: the photograph and depth image the model reads (the
training camera's 40 deg window), the scene cloud it reads, the centre plant as the camera sees it (the target),
and the centre plant written by the row fine-tune and by the headline model without row fine-tuning. Clouds are
drawn front-on (the camera's view), leaf greens from dark at the base to light at the top, nearer points brighter.
Each prediction carries its Chamfer (x1e-4 m^2) and its predicted height against the true one.

usage: python figures/make_e10_fig.py --report reports/e10_real_rows_17363411.csv reports/e10_real_rows_17363434.csv
       writes paper/figs/fig_e10_real.{pdf,png} and copies the PDF to paper/overleaf/Figures/
"""
import sys, pathlib; sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[1]), str(pathlib.Path(__file__).resolve().parents[1] / 'eval')]
import argparse, shutil
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from eval_row import build_model, predict
from real_rows import RealRowDataset
from embodied_mae_4m_maize import _PLANT_SCALE_V2, _PLANT_SHIFT_V2, PLANT_FIELDS_V2

ap = argparse.ArgumentParser()
ap.add_argument('--report', nargs='+', required=True)
ap.add_argument('--per-layout', type=int, default=2)
ap.add_argument('--ft', default='maize_row_ft_e1:checkpoints/checkpoint_epoch_100.pth')
ap.add_argument('--headline', default='maize_4m_pretrain_15k_d2:checkpoints/checkpoint_epoch_1000.pth')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', default='paper/figs/fig_e10_real')
a = ap.parse_args()
torch.set_num_threads(8)

df = pd.concat([pd.read_csv(r, dtype={'row': str}) for r in a.report])
ft_run = a.ft.split(':')[0]
sel = df[(df.model == ft_run) & (df.source == 'all')]
picks = []
for layout in ('parallel', 'perpendicular'):
    g = sel[sel.layout == layout].copy()
    g['dist'] = (g.chamfer_m2 - g.chamfer_m2.median()).abs()
    g = g[(g.stop > 1) & (g.stop < g.stop.max())].sort_values('dist')          # stops with neighbours on both sides
    seen = set()
    for _, r in g.iterrows():
        if r.row in seen: continue
        seen.add(r.row); picks.append((layout, r.row, int(r.stop)))
        if len(seen) == a.per_layout: break
print('stops:', picks)

dev = torch.device('cpu')
H_IDX = PLANT_FIELDS_V2.index('height')
models = []
for label, spec in (('Row fine-tune', a.ft), ('Headline model, no row fine-tune', a.headline)):
    run, ckpt = spec.split(':', 1)
    m, cfg, ep = build_model(pathlib.Path('outputs') / run, ckpt, dev, 'maize')
    models.append((label, m, float(cfg.get('mask_ratio', 0.8))))

GREENS = matplotlib.colors.LinearSegmentedColormap.from_list('plant', ['#1f4d22', '#2f7d32', '#5aa83e', '#a8d26a'])
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def draw_cloud(ax, P, lim):
    P = np.asarray(P, dtype=np.float64)
    h = (P[:, 1] - P[:, 1].min()) / max(np.ptp(P[:, 1]), 1e-9)
    near = (P[:, 2] - P[:, 2].min()) / max(np.ptp(P[:, 2]), 1e-9)                   # OpenGL: larger z = nearer
    c = GREENS(0.15 + 0.85 * h)[:, :3] * (0.55 + 0.45 * near)[:, None]
    o = np.argsort(P[:, 2])
    ax.scatter(P[o, 0], P[o, 1], s=0.25, c=c[o], linewidths=0, rasterized=True)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')


def height_of(par):
    p0 = par[0, 0].clamp(0, 1).float().numpy()
    return float(p0[H_IDX] * _PLANT_SCALE_V2[H_IDX] - _PLANT_SHIFT_V2[H_IDX])


cols = ['Image', 'Depth image', 'Point cloud (scene)', 'Center plant (truth)'] + [m[0] for m in models]
fig, axes = plt.subplots(len(picks), len(cols), figsize=(1.75 * len(cols), 1.85 * len(picks)), squeeze=False)
for i, (layout, row, stop) in enumerate(picks):
    ds = RealRowDataset(layouts=(layout,), rows=[row], stops={stop}, seed=a.seed)
    rgb, dep, pin, par, valid, pgt, info = ds[0]
    B = lambda t: t[None].to(dev)
    img = (rgb.permute(1, 2, 0).numpy() * STD + MEAN).clip(0, 1)
    axes[i, 0].imshow(img)
    d = dep[0].numpy().copy(); d[d == 0] = np.nan
    axes[i, 1].imshow(d, cmap='viridis_r'); axes[i, 1].set_facecolor('black')
    lim = 1.02
    draw_cloud(axes[i, 2], pin.numpy(), lim)
    draw_cloud(axes[i, 3], pgt.numpy(), lim)
    axes[i, 3].text(0.03, 0.03, f'height {info["height"]:.2f} m', transform=axes[i, 3].transAxes, fontsize=6.5)
    for j, (label, m, mr) in enumerate(models):
        with torch.no_grad():
            pp, ppar = predict(m, 'all', B(rgb), B(dep), B(pin), B(par), mr, a.seed + 1000003 * i)
        P = pp[0].float()
        d2 = torch.cdist(P[None], pgt[None])[0]
        ch = float(d2.min(1).values.pow(2).mean() + d2.min(0).values.pow(2).mean()) * info['scale'] ** 2
        draw_cloud(axes[i, 4 + j], P.numpy(), lim)
        axes[i, 4 + j].text(0.03, 0.03, f'Chamfer {1e4 * ch:.0f}e-4 m$^2$\nheight {height_of(ppar):.2f} m',
                            transform=axes[i, 4 + j].transAxes, fontsize=6.5)
    axes[i, 0].set_ylabel(f'{layout}\nrow {row}, stop {stop}', fontsize=7)
for j, c in enumerate(cols):
    axes[0, j].set_title(c, fontsize=7.5)
for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_linewidth(0.4); s.set_color('#888888')
fig.tight_layout(pad=0.3, w_pad=0.2, h_pad=0.3)
out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(f'{out}.pdf', dpi=300); fig.savefig(f'{out}.png', dpi=200)
shutil.copy(f'{out}.pdf', pathlib.Path('paper/overleaf/Figures') / f'{out.name}.pdf')
print('wrote', f'{out}.pdf', f'{out}.png')
