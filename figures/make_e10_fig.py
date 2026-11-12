"""E10 figures: real plants in crop rows (eval/real_rows.py) in, the center plant out, one figure per species.

Rows of a figure are camera stops of HELD-OUT test plants (the 25 % that the real-row fine-tune never saw,
outputs/<species>_real_ft/split.csv), picked per layout at the real-row fine-tune's median Chamfer (all streams in)
in its test report, so the panels show a typical case. Columns: the camera photograph, the depth image and the scene
point cloud the model reads, the center plant as the camera sees it (the target), and the center plant written by
  - the headline model, no row fine-tuning,
  - the row fine-tune (synthetic rows only),
  - the real-row fine-tune (the row fine-tune further trained on 75 % of the real plants),
each with its Chamfer (x1e-4 m^2) and, for maize, its predicted height against the true one. Sorghum frames are tall
strips (7-17 deg wide, 61-76 deg high); its tiles keep that shape and show the original photograph.
Clouds are drawn front-on (the camera's view), leaf greens dark at the base to light at the top, nearer points brighter.

usage: python figures/make_e10_fig.py --species maize|sorghum
       writes paper/figs/fig_e10_real[_sorghum].{pdf,png} and copies the PDF to paper/overleaf/Figures/
"""
import sys, pathlib; sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[1]), str(pathlib.Path(__file__).resolve().parents[1] / 'eval')]
import argparse, csv, shutil
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from eval_row import build_model, predict
from real_rows import RealRowDataset, SPECIES
from embodied_mae_4m_maize import _PLANT_SCALE_V2, _PLANT_SHIFT_V2, PLANT_FIELDS_V2

ap = argparse.ArgumentParser()
ap.add_argument('--species', default='maize', choices=('maize', 'sorghum'))
ap.add_argument('--per-layout', type=int, default=None, help='stops per layout (maize 2, sorghum 1)')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', default=None)
a = ap.parse_args()
torch.set_num_threads(8)
SORG = a.species == 'sorghum'
CFG = {'maize': dict(report='reports/e10_real_ft_test_17383192.csv', real='maize_real_ft:best_model.pth',
                     row='maize_row_ft_e1:checkpoints/checkpoint_epoch_100.pth',
                     head='maize_4m_pretrain_15k_d2:checkpoints/checkpoint_epoch_1000.pth', out='paper/figs/fig_e10_real'),
       'sorghum': dict(report='reports/e10_real_ft_sorghum_test_17384357.csv', real='sorghum_real_ft:best_model.pth',
                       row='sorghum_row_ft:checkpoints/checkpoint_epoch_100.pth',
                       head='4m_pretrain_15k_d2_sink:checkpoints/checkpoint_epoch_1000.pth', out='paper/figs/fig_e10_real_sorghum')}[a.species]
per_layout = a.per_layout or (1 if SORG else 2)

df = pd.read_csv(CFG['report'], dtype={'row': str})
sel = df[(df.model == CFG['real'].split(':')[0]) & (df.source == 'all')]
picks = []
for layout in ('parallel', 'perpendicular'):
    g = sel[sel.layout == layout].copy()
    g['dist'] = (g.chamfer_m2 - g.chamfer_m2.median()).abs()
    g = g[(g.stop > 1) & (g.stop < g.stop.max())].sort_values('dist')
    seen = set()
    for _, r in g.iterrows():
        if r.row in seen: continue
        seen.add(r.row); picks.append((layout, r.row, int(r.stop)))
        if len(seen) == per_layout: break
print('stops:', picks, flush=True)

dev = torch.device('cpu')
models = []
for label, spec in (('Headline model,\nno row fine-tuning', CFG['head']), ('Row fine-tune\n(synthetic rows)', CFG['row']),
                    ('Real-row fine-tune\n(75 % of real plants)', CFG['real'])):
    run, ckpt = spec.split(':', 1)
    m, cfg, _ = build_model(pathlib.Path('outputs') / run, ckpt, dev, a.species)
    models.append((label, m, float(cfg.get('mask_ratio', 0.8)), float(cfg.get('rgb_brighten_gain', 1.0)) if SORG else None))
gain = models[-1][3]

GREENS = matplotlib.colors.LinearSegmentedColormap.from_list('plant', ['#1f4d22', '#2f7d32', '#5aa83e', '#a8d26a'])
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
H_IDX = PLANT_FIELDS_V2.index('height')


def draw_cloud(ax, P, xlim, ylim):
    P = np.asarray(P, dtype=np.float64)
    h = (P[:, 1] - P[:, 1].min()) / max(np.ptp(P[:, 1]), 1e-9)
    near = (P[:, 2] - P[:, 2].min()) / max(np.ptp(P[:, 2]), 1e-9)                   # OpenGL: larger z = nearer
    c = GREENS(0.15 + 0.85 * h)[:, :3] * (0.55 + 0.45 * near)[:, None]
    o = np.argsort(P[:, 2])
    ax.scatter(P[o, 0], P[o, 1], s=0.25 if not SORG else 0.18, c=c[o], linewidths=0, rasterized=True)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect('auto' if SORG else 'equal')   # sorghum: limits match the tall tile


def photo(layout, row, stop):
    """the original camera photograph of a stop (sorghum: shown as captured, a tall strip)"""
    d = next(pathlib.Path(x) for x in SPECIES[a.species]['layouts'][layout].glob(f'row_{row}_seed1'))
    f = list(csv.DictReader(open(d / f'row_{row}_seed1_frames.csv')))[stop - 1]
    im = Image.open(d / f['photo']).convert('RGB'); s = 900 / im.height
    return np.asarray(im.resize((max(1, round(im.width * s)), 900), Image.LANCZOS)) / 255.0


cols = (['Image', 'Depth\nimage', 'Point\ncloud\n(scene)', 'Center\nplant\n(truth)', 'Headline\nmodel, no\nrow FT', 'Row FT\n(synthetic\nrows)',
         'Real-row FT\n(75 % of\nreal plants)'] if SORG else
        ['Image', 'Depth image', 'Point cloud\n(scene)', 'Center plant\n(truth)'] + [m[0] for m in models])
tw, th = (0.85, 2.35) if SORG else (1.75, 1.85)
fig, axes = plt.subplots(len(picks), len(cols), figsize=(tw * len(cols), th * len(picks)), squeeze=False)
for i, (layout, row, stop) in enumerate(picks):
    ds = RealRowDataset(layouts=(layout,), rows=[row], stops={stop}, seed=a.seed, species=a.species, rgb_gain=gain)
    rgb, dep, pin, par, valid, pgt, info = ds[0]
    B = lambda t: t[None].to(dev)
    d = dep[0].numpy().copy(); d[d == 0] = np.nan
    if SORG:
        cs = np.where(np.isfinite(d).any(0))[0]; c0, c1 = max(0, cs.min() - 3), min(d.shape[1], cs.max() + 4)
        axes[i, 0].imshow(photo(layout, row, stop), aspect='auto')
        axes[i, 1].imshow(d[:, c0:c1], cmap='viridis_r', aspect='auto'); axes[i, 1].set_facecolor('black')
        xlim, ylim = (-0.37, 0.37), (-1.02, 1.02)                      # 0.74 / 2.04 = the tile's width / height
    else:
        img = (rgb.permute(1, 2, 0).numpy() * STD + MEAN).clip(0, 1)
        axes[i, 0].imshow(img); axes[i, 1].imshow(d, cmap='viridis_r'); axes[i, 1].set_facecolor('black')
        xlim = ylim = (-1.02, 1.02)
    draw_cloud(axes[i, 2], pin.numpy(), xlim, ylim)
    draw_cloud(axes[i, 3], pgt.numpy(), xlim, ylim)
    fs = 5.6 if SORG else 6.5
    axes[i, 3].text(0.03, 0.02, f'height\n{info["height"]:.2f} m' if SORG else f'height {info["height"]:.2f} m',
                    transform=axes[i, 3].transAxes, fontsize=fs)
    for j, (label, m, mr, _) in enumerate(models):
        with torch.no_grad():
            pp, ppar = predict(m, 'all', B(rgb), B(dep), B(pin), B(par), mr, a.seed + 1000003 * i)
        P = pp[0].float()
        d2 = torch.cdist(P[None], pgt[None])[0]
        ch = float(d2.min(1).values.pow(2).mean() + d2.min(0).values.pow(2).mean()) * info['scale'] ** 2
        draw_cloud(axes[i, 4 + j], P.numpy(), xlim, ylim)
        txt = f'Chamfer\n{1e4 * ch:.0f}e-4 m$^2$' if SORG else f'Chamfer {1e4 * ch:.0f}e-4 m$^2$'
        if not SORG:
            p0 = ppar[0, 0].clamp(0, 1).float().numpy()
            txt += f'\nheight {p0[H_IDX] * _PLANT_SCALE_V2[H_IDX] - _PLANT_SHIFT_V2[H_IDX]:.2f} m'
        axes[i, 4 + j].text(0.03, 0.02, txt, transform=axes[i, 4 + j].transAxes, fontsize=fs)
    axes[i, 0].set_ylabel(f'{layout}\nplant {row}, stop {stop}', fontsize=6.5 if SORG else 7)
for j, c in enumerate(cols):
    axes[0, j].set_title(c, fontsize=6.2 if SORG else 7.5)
for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_linewidth(0.4); s.set_color('#888888')
fig.tight_layout(pad=0.3, w_pad=0.2, h_pad=0.4)
out = pathlib.Path(a.out or CFG['out']); out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(f'{out}.pdf', dpi=300); fig.savefig(f'{out}.png', dpi=200)
shutil.copy(f'{out}.pdf', pathlib.Path('paper/overleaf/Figures') / f'{out.name}.pdf')
print('wrote', f'{out}.pdf', f'{out}.png')
