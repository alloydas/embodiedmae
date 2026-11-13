"""E10 supplement galleries in the UGG style, as Fig. 5 (user, 2026-11-13: "do the same for the E10 galleries").

Same held-out stops, models and seeds as figures/make_e10_gallery.py (the picks are read from its
paper/figs/e10_gallery/index.json, in its order, so predict() sees the same seed 1000003 * i), drawn as ONE figure per
species instead of hand-assembled tiles:
  image (dashed blue box, the camera picture the model reads) | ground truth (grey box) | row fine-tune on synthetic
  rows (pink box) | real-row fine-tune on 75 % of the real plants (pink box; header in UGG's 'Ours' violet);
  'Ground truth' for 'Center plant (truth)'; each prediction labelled 'CD = <Chamfer, 1e-4 m^2>' (maize: and its height);
  headers once per column; maize: parallel / perpendicular rows side by side, one row per Chamfer percentile;
  sorghum: the eight held-out plants, two rows of four.
Each recomputed Chamfer is checked against index.json (the published tiles). Font: Liberation Sans (Fig. 5's).
Output: paper/figs/ugg_preview/fig_e10_gallery_{maize,sorghum}_ugg.{pdf,png}. CPU, ~5 min:
  sbatch --wait --partition=nova --account=mech-ai --qos=normal --cpus-per-task=8 --mem=48G --time=01:00:00 \
         --output=logs/%x_%j.out --wrap "python figures/make_e10_gallery_ugg.py"
"""
import sys, pathlib; R = pathlib.Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R), str(R / 'eval'), str(R / 'figures')]
import csv, json
import numpy as np, torch, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from PIL import Image
from eval_row import build_model, predict
from real_rows import RealRowDataset, SPECIES
from embodied_mae_4m_maize import _PLANT_SCALE_V2, _PLANT_SHIFT_V2, PLANT_FIELDS_V2
import _scene                                                   # Liberation Sans, as Fig. 5
import ugg_style as U

torch.set_num_threads(8)
OUT = R / 'paper/figs/ugg_preview'
IDX = json.load(open(R / 'paper/figs/e10_gallery/index.json'))
CFG = {'maize': dict(real='maize_real_ft:best_model.pth', row='maize_row_ft_e1:checkpoints/checkpoint_epoch_100.pth'),
       'sorghum': dict(real='sorghum_real_ft:best_model.pth', row='sorghum_row_ft:checkpoints/checkpoint_epoch_100.pth')}
GREENS = matplotlib.colors.LinearSegmentedColormap.from_list('plant', ['#1f4d22', '#2f7d32', '#5aa83e', '#a8d26a'])
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
H_IDX = PLANT_FIELDS_V2.index('height')
FS = 7.0
INKC, OURS = '#111111', U.OURS
HEAD = ['Image', 'Ground truth', 'Row fine-tune\n(synthetic)', 'Real-row\nfine-tune']
HEAD_S = ['Image', 'Ground\ntruth', 'Row\nfine-tune', 'Real-row\nfine-tune']
HEAD_C = [INKC, INKC, U.STREAM_INK['pc'], OURS]
HALO = [pe.withStroke(linewidth=2.0, foreground='white')]


def cloud(ax, P, xlim, ylim, sorg):
    P = np.asarray(P, dtype=np.float64)
    h = (P[:, 1] - P[:, 1].min()) / max(np.ptp(P[:, 1]), 1e-9); near = (P[:, 2] - P[:, 2].min()) / max(np.ptp(P[:, 2]), 1e-9)
    c = GREENS(0.15 + 0.85 * h)[:, :3] * (0.55 + 0.45 * near)[:, None]; o = np.argsort(P[:, 2])
    ax.scatter(P[o, 0], P[o, 1], s=0.10 if sorg else 0.18, c=c[o], linewidths=0, rasterized=True)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)


def compute(sp):
    """the published tiles' stops, recomputed: image, truth cloud, both predictions, Chamfer, heights"""
    picks = [e for e in IDX if e['species'] == sp]
    models = []
    for spec in (CFG[sp]['row'], CFG[sp]['real']):
        run, ck = spec.split(':', 1); m, cfg, _ = build_model(R / 'outputs' / run, ck, torch.device('cpu'), sp)
        models.append((m, float(cfg.get('mask_ratio', 0.8)), float(cfg.get('rgb_brighten_gain', 1.0)) if sp == 'sorghum' else None))
    out = []
    for i, e in enumerate(picks):
        lay, row, stop = e['layout'], e['plant'], e['stop']
        ds = RealRowDataset(layouts=(lay,), rows=[row], stops={stop}, seed=0, species=sp, rgb_gain=models[-1][2])
        rgb, dep, pin, par, valid, pgt, info = ds[0]
        if sp == 'sorghum':
            d = next(pathlib.Path(x) for x in SPECIES[sp]['layouts'][lay].glob(f'row_{row}_seed1'))
            f = list(csv.DictReader(open(d / f'row_{row}_seed1_frames.csv')))[stop - 1]
            im = Image.open(d / f['photo']).convert('RGB'); s = 900 / im.height
            img = np.asarray(im.resize((max(1, round(im.width * s)), 900))) / 255.0
        else:
            img = (rgb.permute(1, 2, 0).numpy() * STD + MEAN).clip(0, 1)
        r = dict(e, img=img, gt=pgt.numpy(), preds=[], ch=[], hp=[])
        for j, (m, mr, _) in enumerate(models):
            with torch.no_grad():
                pp, ppar = predict(m, 'all', rgb[None], dep[None], pin[None], par[None], mr, 1000003 * i)
            P = pp[0].float(); d2 = torch.cdist(P[None], pgt[None])[0]
            ch = 1e4 * float(d2.min(1).values.pow(2).mean() + d2.min(0).values.pow(2).mean()) * info['scale'] ** 2
            r['preds'].append(P.numpy()); r['ch'].append(ch)
            r['hp'].append(float(ppar[0, 0, H_IDX].clamp(0, 1)) * _PLANT_SCALE_V2[H_IDX] - _PLANT_SHIFT_V2[H_IDX] if sp == 'maize' else None)
        ref = (e['before_chamfer'], e['after_chamfer'])
        assert all(abs(a - b) <= 0.01 * max(1.0, b) for a, b in zip(r['ch'], ref)), (sp, row, stop, r['ch'], ref)
        out.append(r); print(sp, lay, row, stop, [round(c, 1) for c in r['ch']], 'published', [round(c, 1) for c in ref], flush=True)
    return out


def block(fig, W, H, x, y, t, th, r, sorg, xlim, ylim, fs):
    """one stop: four boxed tiles of width t, height th, lower left (x, y) in inches"""
    m = 0.025
    kinds = [('rgb', True), (None, False), ('pc', False), ('pc', False)]
    for k in range(4):
        xk = x + k * (t + 0.035)
        U.pastel_box(fig, xk, y, t, th, kinds[k][0], dashed=kinds[k][1], r=0.04, transform=fig.dpi_scale_trans, z=-1)
        ax = fig.add_axes([(xk + m) / W, (y + m) / H, (t - 2 * m) / W, (th - 2 * m) / H]); ax.set_axis_off(); ax.patch.set_alpha(0)
        if k == 0:
            ax.imshow(r['img'], aspect='auto')
        elif k == 1:
            cloud(ax, r['gt'], xlim, ylim, sorg)
            if not sorg:
                ax.text(0.04, 0.03, f"height {r['height']:.2f} m", transform=ax.transAxes, fontsize=fs, color=INKC, path_effects=HALO)
        else:
            j = k - 2; cloud(ax, r['preds'][j], xlim, ylim, sorg)
            c = U.STREAM_INK['pc'] if j == 0 else OURS
            if sorg:                                         # narrow tiles: the number goes under the box
                fig.text((xk + t / 2) / W, (y - 0.04) / H, f"CD =\n{r['ch'][j]:.0f}", fontsize=fs, color=c, ha='center', va='top', linespacing=1.05)
            else:
                ax.text(0.04, 0.03, f"CD = {r['ch'][j]:.0f}\nheight {r['hp'][j]:.2f} m", transform=ax.transAxes, fontsize=fs, color=c,
                        path_effects=HALO, linespacing=1.15)


def render_maize(D):
    W = 6.875; lab = 0.42; gb = 0.22; t = (W - lab - gb - 0.10 - 2 * 3 * 0.035) / 8; hdr = 0.52; gy = 0.06
    rows = {'parallel': [r for r in D if r['layout'] == 'parallel'], 'perpendicular': [r for r in D if r['layout'] == 'perpendicular']}
    n = max(len(v) for v in rows.values()); H = hdr + n * t + (n - 1) * gy + 0.03
    fig = plt.figure(figsize=(W, H))
    pct = ['10th', '30th', '50th', '70th', '90th', '98th']
    for b, lay in enumerate(('parallel', 'perpendicular')):
        xb = lab + b * (4 * t + 3 * 0.035 + gb)
        fig.text((xb + (4 * t + 3 * 0.035) / 2) / W, (H - 0.02) / H, f'{lay.capitalize()} rows', fontsize=FS + 0.5, fontweight='bold', ha='center', va='top')
        for k in range(4):
            fig.text((xb + k * (t + 0.035) + t / 2) / W, (H - 0.22) / H, HEAD[k], fontsize=FS - 0.6, ha='center', va='top', color=HEAD_C[k], linespacing=1.0)
        y = H - hdr
        for i, r in enumerate(rows[lay]):
            y -= t
            block(fig, W, H, xb, y, t, t, r, False, (-1.02, 1.02), (-1.02, 1.02), FS - 0.8)
            if b == 0:
                fig.text((lab - 0.06) / W, (y + t / 2) / H, pct[i], fontsize=FS, ha='right', va='center')
            y -= gy
    return fig


def render_sorghum(D):
    W = 6.875; gb = 0.16; t = (W - 0.10 - 3 * gb - 4 * 3 * 0.035) / 16; th = 2.7 * t; hdr = 0.40; gy = 0.52
    H = hdr + 2 * th + gy + 0.42
    x00 = 0.05
    fig = plt.figure(figsize=(W, H))
    for i, r in enumerate(D):
        rr, cc = divmod(i, 4)
        xb = x00 + cc * (4 * t + 3 * 0.035 + gb); yb = H - hdr - (rr + 1) * th - rr * gy
        if rr == 0:
            for k in range(4):
                fig.text((xb + k * (t + 0.035) + t / 2) / W, (H - 0.03) / H, HEAD_S[k], fontsize=FS - 0.8, ha='center', va='top', color=HEAD_C[k], linespacing=1.0)
        block(fig, W, H, xb, yb, t, th, r, True, (-0.37, 0.37), (-1.02, 1.02), FS - 1.2)
        fig.text((xb + (4 * t + 3 * 0.035) / 2) / W, (yb - 0.32) / H, f"{r['layout']} row, true height {r['height']:.2f} m", fontsize=FS - 1.0,
                 ha='center', va='top', color='#444444')
    return fig


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    for sp, fn in (('maize', render_maize), ('sorghum', render_sorghum)):
        fig = fn(compute(sp))
        for ext in ('pdf', 'png'):
            fig.savefig(OUT / f'fig_e10_gallery_{sp}_ugg.{ext}', dpi=300)
        plt.close(fig); print('wrote', OUT / f'fig_e10_gallery_{sp}_ugg.pdf', flush=True)
