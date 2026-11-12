"""E10 gallery: many held-out real plants, before and after the real-row fine-tune, one PNG per camera stop.

Maize: 6 stops per layout on distinct held-out plants, at the 10/30/50/70/90/98th percentiles of the real-row
fine-tune's Chamfer (all streams in), so the gallery spans good and bad cases. Sorghum: one stop (the middle one) of
each of the 8 held-out plants, layouts alternating. Each tile: the camera image the model reads, the true center plant,
the center plant from the row fine-tune (synthetic rows) and from the real-row fine-tune, with Chamfer (x1e-4 m^2)
and, for maize, height. Writes paper/figs/e10_gallery/<species>_<layout>_<plant>_<stop>.png and index.json.
usage: python figures/make_e10_gallery.py [--species maize sorghum]
"""
import sys, pathlib; R = pathlib.Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R), str(R / 'eval')]
import argparse, csv, json
import numpy as np, pandas as pd, torch, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
from eval_row import build_model, predict
from real_rows import RealRowDataset, SPECIES
from embodied_mae_4m_maize import _PLANT_SCALE_V2, _PLANT_SHIFT_V2, PLANT_FIELDS_V2

ap = argparse.ArgumentParser(); ap.add_argument('--species', nargs='+', default=['maize', 'sorghum']); a = ap.parse_args()
torch.set_num_threads(8)
OUT = R / 'paper/figs/e10_gallery'; OUT.mkdir(parents=True, exist_ok=True)
CFG = {'maize': dict(report='reports/e10_real_ft_test_17383192.csv', real='maize_real_ft:best_model.pth',
                     row='maize_row_ft_e1:checkpoints/checkpoint_epoch_100.pth'),
       'sorghum': dict(report='reports/e10_real_ft_sorghum_test_17384357.csv', real='sorghum_real_ft:best_model.pth',
                       row='sorghum_row_ft:checkpoints/checkpoint_epoch_100.pth')}
GREENS = matplotlib.colors.LinearSegmentedColormap.from_list('plant', ['#1f4d22', '#2f7d32', '#5aa83e', '#a8d26a'])
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
H_IDX = PLANT_FIELDS_V2.index('height')
index = json.load(open(OUT / 'index.json')) if (OUT / 'index.json').exists() else []
index = [e for e in index if e['species'] not in a.species]


def cloud(ax, P, xlim, ylim, sorg):
    P = np.asarray(P, dtype=np.float64)
    h = (P[:, 1] - P[:, 1].min()) / max(np.ptp(P[:, 1]), 1e-9); near = (P[:, 2] - P[:, 2].min()) / max(np.ptp(P[:, 2]), 1e-9)
    c = GREENS(0.15 + 0.85 * h)[:, :3] * (0.55 + 0.45 * near)[:, None]; o = np.argsort(P[:, 2])
    ax.scatter(P[o, 0], P[o, 1], s=0.15 if sorg else 0.25, c=c[o], linewidths=0, rasterized=True)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect('auto' if sorg else 'equal')


for sp in a.species:
    sorg = sp == 'sorghum'; C = CFG[sp]
    df = pd.read_csv(R / C['report'], dtype={'row': str})
    sel = df[(df.model == C['real'].split(':')[0]) & (df.source == 'all')]
    picks = []
    if sorg:
        for k, row in enumerate(sorted(sel.row.unique())):
            lay = ('parallel', 'perpendicular')[k % 2]
            picks.append((lay, row, 9))
    else:
        for lay in ('parallel', 'perpendicular'):
            g = sel[(sel.layout == lay) & (sel.stop > 1) & (sel.stop < 18)]; used = set()
            for q in (0.10, 0.30, 0.50, 0.70, 0.90, 0.98):
                t = g.chamfer_m2.quantile(q); cand = g.assign(d=(g.chamfer_m2 - t).abs()).sort_values('d')
                r = next(r for _, r in cand.iterrows() if r.row not in used); used.add(r.row)
                picks.append((lay, r.row, int(r.stop)))
    models = []
    for lab, spec in (('Row fine-tune\n(synthetic rows)', C['row']), ('Real-row fine-tune\n(75 % of real plants)', C['real'])):
        run, ck = spec.split(':', 1); m, cfg, _ = build_model(R / 'outputs' / run, ck, torch.device('cpu'), sp)
        models.append((lab, m, float(cfg.get('mask_ratio', 0.8)), float(cfg.get('rgb_brighten_gain', 1.0)) if sorg else None))
    for i, (lay, row, stop) in enumerate(picks):
        ds = RealRowDataset(layouts=(lay,), rows=[row], stops={stop}, seed=0, species=sp, rgb_gain=models[-1][3])
        rgb, dep, pin, par, valid, pgt, info = ds[0]
        tw, th = (0.95, 2.6) if sorg else (1.6, 1.6)
        fig, ax = plt.subplots(1, 4, figsize=(4 * tw, th + 0.6))
        if sorg:
            d = next(pathlib.Path(x) for x in SPECIES[sp]['layouts'][lay].glob(f'row_{row}_seed1'))
            f = list(csv.DictReader(open(d / f'row_{row}_seed1_frames.csv')))[stop - 1]
            im = Image.open(d / f['photo']).convert('RGB'); s = 900 / im.height
            ax[0].imshow(np.asarray(im.resize((max(1, round(im.width * s)), 900))) / 255.0, aspect='auto')
            xlim, ylim = (-0.37, 0.37), (-1.02, 1.02)
        else:
            ax[0].imshow((rgb.permute(1, 2, 0).numpy() * STD + MEAN).clip(0, 1)); xlim = ylim = (-1.02, 1.02)
        cloud(ax[1], pgt.numpy(), xlim, ylim, sorg)
        ax[1].text(0.03, 0.02, f'true height\n{info["height"]:.2f} m', transform=ax[1].transAxes, fontsize=6.5)
        e = dict(species=sp, layout=lay, plant=row, stop=stop, height=float(info['height']), occ=float(info['occ_frac']))
        for j, (lab, m, mr, _) in enumerate(models):
            with torch.no_grad():
                pp, ppar = predict(m, 'all', rgb[None], dep[None], pin[None], par[None], mr, 1000003 * i)
            P = pp[0].float(); d2 = torch.cdist(P[None], pgt[None])[0]
            ch = 1e4 * float(d2.min(1).values.pow(2).mean() + d2.min(0).values.pow(2).mean()) * info['scale'] ** 2
            cloud(ax[2 + j], P.numpy(), xlim, ylim, sorg)
            txt = f'Chamfer {ch:.0f}'
            if not sorg:
                hp = float(ppar[0, 0, H_IDX].clamp(0, 1)) * _PLANT_SCALE_V2[H_IDX] - _PLANT_SHIFT_V2[H_IDX]
                txt += f'\nheight {hp:.2f} m'; e[('before' if j == 0 else 'after') + '_height'] = float(hp)
            e[('before' if j == 0 else 'after') + '_chamfer'] = float(ch)
            ax[2 + j].text(0.03, 0.02, txt, transform=ax[2 + j].transAxes, fontsize=6.5)
        for k, t in enumerate(['Image', 'Center plant (truth)'] + [m_[0] for m_ in models]):
            ax[k].set_title(t, fontsize=6.8)
        for x in ax:
            x.set_xticks([]); x.set_yticks([])
            for s_ in x.spines.values(): s_.set_linewidth(0.4); s_.set_color('#999999')
        fig.tight_layout(pad=0.25, w_pad=0.25)
        name = f'{sp}_{lay}_{row}_{stop:02d}.png'; fig.savefig(OUT / name, dpi=170, facecolor='white'); plt.close(fig)
        e['file'] = name; index.append(e); print(name, {k: round(v, 3) if isinstance(v, float) else v for k, v in e.items()}, flush=True)
json.dump(index, open(OUT / 'index.json', 'w'), indent=1)
print('done', len(index), 'tiles')
