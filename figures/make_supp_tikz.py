"""Supplement figures S1, S4, S5, S10, S11 as TikZ (user, 2026-11-20: "Convert them in tikz", after they were redrawn
with textured 3D procedural models).

Built on _scene.Scene like Fig. 5 and Figs. S8-S9: boxes, headers and labels are TikZ; each tile's content (photograph,
depth image, point cloud, 3D procedural model) is an embedded PNG. The data are the --proc runs of the matplotlib
scripts (same stops, models, seeds and numbers):
  S1  fig:dataset_stats      panel a rebuilt here (make_dataset_stats_fig.py's picks, true procedural models);
                             panels b-d embedded as one 600 dpi image of that script's --proc output
  S4/S5  fig:e10_real(_sorghum)   supp_fig_e10_real{,_sorghum}_proc.pkl  (figures/make_e10_fig.py --proc)
  S10/S11 fig:supp-e10-*     supp_fig_e10_gallery_{maize,sorghum}_ugg_proc.pkl  (figures/make_e10_gallery_ugg.py --proc)
Writes paper/overleaf/Figures/<name>_tikz.tex + Figures/<name>/ and a preview paper/figs/ugg_preview/<name>_tikz.{pdf,png}.
  python figures/make_supp_tikz.py [s1 s4 s5 s10 s11]
"""
import sys, pathlib, pickle, io
R = pathlib.Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R), str(R / 'figures'), str(R / 'eval')]
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from _scene import Scene, png_array, png_fig
import _procfig as PF
from ugg_style import STREAM, STREAM_INK, STREAM_FILL, OURS

D12 = pathlib.Path('/work/mech-ai/alloydas/embodiedmae_outputs/_fig12')
FIG = R / 'paper/overleaf/Figures'; PREV = R / 'paper/figs/ugg_preview'
GREENS = matplotlib.colors.LinearSegmentedColormap.from_list('plant', ['#1f4d22', '#2f7d32', '#5aa83e', '#a8d26a'])
INK, NEUTRAL_E, NEUTRAL_F = '#111111', '#9a9a9a', '#f4f4f5'
BOX = {k: (STREAM[k], STREAM_FILL[k]) for k in STREAM}; BOX[None] = (NEUTRAL_E, NEUTRAL_F)


def cloud_png(P, w, h, xlim, ylim, s=0.18):
    P = np.asarray(P, np.float64)
    hh = (P[:, 1] - P[:, 1].min()) / max(np.ptp(P[:, 1]), 1e-9); near = (P[:, 2] - P[:, 2].min()) / max(np.ptp(P[:, 2]), 1e-9)
    c = GREENS(0.15 + 0.85 * hh)[:, :3] * (0.55 + 0.45 * near)[:, None]; o = np.argsort(P[:, 2])
    fig = plt.figure(figsize=(w, h)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.scatter(P[o, 0], P[o, 1], s=s, c=c[o], linewidths=0); ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    return png_fig(fig, 500)


def depth_png(d, max_side=600):
    d = np.asarray(d, np.float64); ok = np.isfinite(d)
    rgb = np.zeros(d.shape + (3,))
    if ok.any():
        lo, hi = np.nanmin(d), np.nanmax(d); rgb[ok] = plt.get_cmap('viridis_r')((d[ok] - lo) / max(hi - lo, 1e-9))[:, :3]
    return png_array(rgb, max_side)


def fit_png(arr, w, h, side=700):
    """an image placed in a w x h box without distortion: padded with black to the box's shape"""
    a_ = np.asarray(arr, np.float64); H0, W0 = a_.shape[:2]; tgt = w / h
    if W0 / H0 < tgt:
        pad = int(round(H0 * tgt)) - W0; a_ = np.pad(a_, ((0, 0), (pad // 2, pad - pad // 2), (0, 0)))
    elif W0 / H0 > tgt:
        pad = int(round(W0 / tgt)) - H0; a_ = np.pad(a_, ((pad // 2, pad - pad // 2), (0, 0), (0, 0)))
    return png_array(a_, side)


def box(S, x, y, w, h, kind, dash=False, r=0.04):
    e, f = BOX[kind]; S.rect(x, y, w, h, e, lw=1.0, fill=f, r=r, dash=dash)


def proc_tiles(recs, sp, aspect):
    lim = PF.shared_lim(recs, sp)
    return [png_array(t, 700) for t in PF.trim_group([PF.tile([(rc, PF.P3.TEXTURE, 1.0)], sp, lim=lim, px=700) for rc in recs], keep_aspect=aspect)]


def save(S, name):
    S.render(PREV / f'{name}_tikz', drawio=False)
    S.render_tikz(FIG / f'{name}_tikz.tex', FIG / name, f'Figures/{name}'); print('wrote', FIG / f'{name}_tikz.tex', flush=True)


# ------------------------------------------------------------------ S10 / S11: real-row galleries
def gallery(sp):
    D = pickle.load(open(D12 / f'supp_fig_e10_gallery_{sp}_ugg_proc.pkl', 'rb'))
    W, m, g = 6.875, 0.025, 0.035
    head = [('Image', INK), ('Ground truth', INK), ('Row fine-tune\n(synthetic)', STREAM_INK['text']), ('Real-row\nfine-tune', OURS)]
    kinds = [('rgb', True), (None, False), ('text', False), ('text', False)]
    if sp == 'maize':
        F = 6.2; lab, gb = 0.42, 0.22; t = (W - lab - gb - 0.10 - 2 * 3 * g) / 8; th = t; hdr, gy = 0.52, 0.06
        rows = {L: [r for r in D if r['layout'] == L] for L in ('parallel', 'perpendicular')}
        n = max(len(v) for v in rows.values()); H = hdr + n * t + (n - 1) * gy + 0.03
        S = Scene(W, H, f'S10 real maize gallery'); pct = ['10th', '30th', '50th', '70th', '90th', '98th']
        for b, L in enumerate(('parallel', 'perpendicular')):
            xb = lab + b * (4 * t + 3 * g + gb)
            S.text(xb + (4 * t + 3 * g) / 2, H - 0.02, f'{L.capitalize()} rows', 7.5, ha='center', va='top', bold=True)
            for k, (s_, c) in enumerate(head):
                S.text(xb + k * (t + g) + t / 2, H - 0.22, s_, 6.4, color=c, ha='center', va='top', ls=1.05)
            y = H - hdr
            for i, r in enumerate(rows[L]):
                y -= t; inner = t - 2 * m
                pt = proc_tiles(r['recs'], 'maize', inner / (inner * 0.70))
                for k in range(4):
                    xk = xb + k * (t + g); box(S, xk, y, t, t, kinds[k][0], kinds[k][1])
                    if k == 0: S.image(xk + m, y + m, inner, inner, fit_png(r['img'], inner, inner, 500), f'm {L} {i} image')
                    elif k == 1:
                        S.image(xk + m, y + m, inner, inner, cloud_png(r['gt'], inner, inner, (-1.02, 1.02), (-1.02, 1.02)), f'm {L} {i} truth')
                        S.text(xk + m + 0.04 * inner, y + m + 0.03 * inner, f"height {r['height']:.2f} m", F, va='bottom')
                    else:
                        j = k - 2; S.image(xk + m, y + m + 0.30 * inner, inner, 0.70 * inner, pt[j], f'm {L} {i} pred {j}')
                        S.text(xk + m + 0.04 * inner, y + m + 0.03 * inner, f"CD = {r['ch'][j]:.0f}\nheight {r['hp'][j]:.2f} m", F,
                               color=STREAM_INK['text'] if j == 0 else OURS, va='bottom', ls=1.15)
                if b == 0: S.text(lab - 0.06, y + t / 2, pct[i], 7.0, ha='right', va='center')
                y -= gy
        return S, 'fig_e10_gallery_maize'
    F = 5.8; gb = 0.16; t = (W - 0.10 - 3 * gb - 4 * 3 * g) / 16; th = 2.7 * t; hdr, gy = 0.40, 0.52; x00 = 0.05
    H = hdr + 2 * th + gy + 0.42; S = Scene(W, H, 'S11 real sorghum gallery')
    head_s = [('Image', INK), ('Ground\ntruth', INK), ('Row\nfine-tune', STREAM_INK['text']), ('Real-row\nfine-tune', OURS)]
    for i, r in enumerate(D):
        rr, cc = divmod(i, 4); xb = x00 + cc * (4 * t + 3 * g + gb); yb = H - hdr - (rr + 1) * th - rr * gy
        iw, ih = t - 2 * m, th - 2 * m
        if rr == 0:
            for k, (s_, c) in enumerate(head_s):
                S.text(xb + k * (t + g) + t / 2, H - 0.03, s_, 6.2, color=c, ha='center', va='top', ls=1.0)
        pt = proc_tiles(r['recs'], 'sorghum', iw / ih)
        for k in range(4):
            xk = xb + k * (t + g); box(S, xk, yb, t, th, kinds[k][0], kinds[k][1])
            if k == 0: S.image(xk + m, yb + m, iw, ih, fit_png(r['img'], iw, ih), f's {i} image')
            elif k == 1: S.image(xk + m, yb + m, iw, ih, cloud_png(r['gt'], iw, ih, (-0.37, 0.37), (-1.02, 1.02), s=0.10), f's {i} truth')
            else:
                j = k - 2; S.image(xk + m, yb + m, iw, ih, pt[j], f's {i} pred {j}')
                S.text(xk + t / 2, yb - 0.04, f"CD =\n{r['ch'][j]:.0f}", F, color=STREAM_INK['text'] if j == 0 else OURS,
                       ha='center', va='top', ls=1.05, small_ok=True)
        S.text(xb + (4 * t + 3 * g) / 2, yb - 0.32, f"{r['layout']} row, true height {r['height']:.2f} m", 6.0, color='#444444',
               ha='center', va='top')
    return S, 'fig_e10_gallery_sorghum'


# ------------------------------------------------------------------ S4 / S5: real rows, one stop per row
def e10(sp):
    D = pickle.load(open(D12 / f"supp_fig_e10_real{'_sorghum' if sp == 'sorghum' else ''}_proc.pkl", 'rb'))
    sorg = sp == 'sorghum'; W, g, m, lab = (5.6 if sorg else 6.875), 0.04, 0.02, (0.95 if sorg else 0.62)
    heads = ['Image', 'Depth image', 'Point cloud\n(scene)', 'Center plant\n(truth)', 'Headline model,\nno row fine-tuning',
             'Row fine-tune\n(synthetic rows)', 'Real-row fine-tune\n(75 % of real plants)']
    if sorg: heads = ['Image', 'Depth\nimage', 'Point\ncloud\n(scene)', 'Center\nplant\n(truth)', 'Headline\nmodel, no\nrow FT',
                      'Row FT\n(synthetic\nrows)', 'Real-row FT\n(75 % of\nreal plants)']
    kinds = [('rgb', True), ('depth', True), ('pc', True), (None, False), ('text', False), ('text', False), ('text', False)]
    hc = [INK] * 4 + [STREAM_INK['text'], STREAM_INK['text'], OURS]
    t = (W - lab - 6 * g) / 7; th = 2.7 * t if sorg else t; hdr = 0.50 if sorg else 0.36; gy = 0.07
    n = len(D['rows']); H = hdr + n * th + (n - 1) * gy + 0.02
    S = Scene(W, H, f"S{'5' if sorg else '4'} real {sp} rows"); F = 6.0
    for k, (s_, c) in enumerate(zip(heads, hc)):
        S.text(lab + k * (t + g) + t / 2, H - 0.01, s_, 6.4, color=c, ha='center', va='top', ls=1.05)
    xl, yl = ((-0.37, 0.37), (-1.02, 1.02)) if sorg else ((-1.02, 1.02), (-1.02, 1.02))
    y = H - hdr
    for i, r in enumerate(D['rows']):
        y -= th; iw, ih = t - 2 * m, th - 2 * m; lo = 0.17 if sorg else 0.24
        pt = proc_tiles(r['recs'], sp, iw / (ih * (1 - lo)))
        for k in range(7):
            xk = lab + k * (t + g); box(S, xk, y, t, th, kinds[k][0], kinds[k][1], r=0.03)
            X, Y = xk + m, y + m
            if k == 0: S.image(X, Y, iw, ih, fit_png(r['img'], iw, ih), f'{sp} {i} image')
            elif k == 1: S.image(X, Y, iw, ih, depth_png(r['depth']), f'{sp} {i} depth')
            elif k in (2, 3):
                P = r['pin'] if k == 2 else r['pgt']
                S.image(X, Y, iw, ih, cloud_png(P, iw, ih, xl, yl, s=0.14 if sorg else 0.22), f'{sp} {i} cloud {k}')
                if k == 3: S.text(X + 0.04 * iw, Y + 0.02 * iw, f"height\n{r['height']:.2f} m" if sorg else f"height {r['height']:.2f} m", F, va='bottom', ls=1.1)
            else:
                j = k - 4; S.image(X, Y + lo * ih, iw, (1 - lo) * ih, pt[j], f'{sp} {i} pred {j}')
                tx = r['texts'][j].replace('Chamfer\n', 'CD = ').replace('Chamfer ', 'CD = ').replace('e-4 m$^2$', '')
                S.text(X + 0.04 * iw, Y + 0.02 * iw, tx, F, color=hc[k], va='bottom', ls=1.1)
        L, row, stop = r['label']
        S.text(lab - 0.08, y + th / 2, f'{L}\nplant {row}\nstop {stop}', 6.4, ha='right', va='center', ls=1.1)
        y -= gy
    return S, f"fig_e10_real{'_sorghum' if sorg else ''}"


# ------------------------------------------------------------------ S1: dataset statistics
def stats():
    import _plantviz as V, export_procedural_models as EP
    sf = pd.read_csv(V.ROOTS['sorghum'] / 'features.csv').merge(pd.read_csv(V.ROOTS['sorghum'] / 'assignment.csv')[['plant', 'split']], on='plant')
    sf['height'] = sf['stem_length']; sf['leaves'] = sf['n_leaves']; sf['name'] = 'Sorghum_' + sf['plant'].astype(str)
    mf = pd.read_csv(V.ROOTS['maize'] / 'plant_scores.csv'); mf['height'] = mf['stem_internodeSum']; mf['leaves'] = mf['leaf_count']; mf['name'] = mf['plant']
    W = 6.875; lab = 0.30; g = 0.05; t = (W - lab - 7 * g) / 8; th = 1.25 * t; hdr = 0.20; rowgap = 0.22
    # panels b-d: the matplotlib figure's lower part (same code, --proc run), embedded as one image
    full = Image.open(R / 'paper/figs/fig_dataset_stats_proc_600.png').convert('RGB'); w0, h0 = full.size
    cut = int(h0 * (1 - 0.445)); bd = full.crop((0, cut, w0, h0)); bdh = W * bd.height / bd.width
    H = hdr + 2 * (th + 0.17) + rowgap + bdh + 0.05
    S = Scene(W, H, 'S1 dataset statistics')
    S.text(0.0, H - 0.01, 'a', 9.0, bold=True, va='top')
    S.text(0.16, H - 0.025, 'The morphological range: eight val plants per species along the leaf-count range (sorghum 4–35 leaves, maize 3–22), true procedural models',
           6.4, va='top')
    y = H - hdr
    for sp, T in (('sorghum', sf), ('maize', mf)):
        t_ = T[T.split == 'val'].sort_values('leaves')
        picks = [t_.iloc[int(q * (len(t_) - 1))] for q in np.linspace(0.02, 0.98, 8)]
        recs = [EP.true_recipe(sp, p['name']) for p in picks]; lim = PF.shared_lim(recs, sp)
        tiles = [png_array(a, 700) for a in PF.trim_group([PF.tile([(r_, PF.P3.TEXTURE, 1.0)], sp, lim=lim, px=700) for r_ in recs], keep_aspect=t / th)]
        y -= 0.17 + th
        for k, (p, png) in enumerate(zip(picks, tiles)):
            x = lab + k * (t + g); box(S, x, y, t, th, 'text', r=0.03); S.image(x + 0.02, y + 0.02, t - 0.04, th - 0.04, png, f'{sp} {k}')
            S.text(x + t / 2, y + th + 0.03, f'{int(p["leaves"])} leaves, {p["height"]:.2f} m', 6.0, ha='center', va='bottom')
        S.text(0.10, y + th / 2, sp, 6.6, ha='center', va='center', rot=90)
        y -= rowgap / 2
    b = io.BytesIO(); bd.save(b, 'PNG', optimize=True); S.image(0, 0.02, W, bdh, b.getvalue(), 'panels b-d')
    return S, 'fig_dataset_stats'


# ------------------------------------------------------------------ field data (2026-11-20): S4/S5 and S10/S11 rebuilt on the
# field upload (figures/make_field_dump.py); both species in one layout, 3:4 tiles (the field photographs are 3:4)
FIELD_HC = {'row': STREAM_INK['text'], 'field': OURS}


def own_lims(P, w, h):
    """limits that frame the plant itself (its height fills the tile, the width follows the tile's shape)"""
    P = np.asarray(P); y0, y1 = P[:, 1].min(), P[:, 1].max(); cy = (y0 + y1) / 2; hh = (y1 - y0) / 2 * 1.06
    cx = float(np.median(P[:, 0])); hw = hh * w / h
    return (cx - hw, cx + hw), (cy - hh, cy + hh)


def field_gallery(sp):
    D = pickle.load(open(D12 / f'supp_field_gallery_{sp}.pkl', 'rb'))
    W, m, g = 6.875, 0.02, 0.03
    lab, gb = (0.70, 0.74) if sp == 'sorghum' else (0.48, 0.52)      # room for the plant ids left of each block
    t = (W - lab - gb - 6 * g) / 8; th = t * 4 / 3; hdr, gy = 0.42, 0.06
    blocks = [[r for r in D if r['block'] == b] for b in ('A', 'B')] if sp == 'maize' else [D[0::2], D[1::2]]
    n = max(len(b) for b in blocks); H = hdr + n * th + (n - 1) * gy + 0.02
    S = Scene(W, H, f'field gallery {sp}'); F = 6.0
    head = [('Image', INK), ('Ground truth', INK), ('Synthetic-row\nfine-tune', STREAM_INK['text']), ('Field\nfine-tune', OURS)]
    kinds = [('rgb', True), (None, False), ('text', False), ('text', False)]
    for b, rows in enumerate(blocks):
        xb = lab + b * (4 * t + 3 * g + gb)
        for k, (s_, c) in enumerate(head):
            S.text(xb + k * (t + g) + t / 2, H - 0.01, s_, 6.4, color=c, ha='center', va='top', ls=1.05)
        y = H - hdr
        for i, r in enumerate(rows):
            y -= th; iw, ih = t - 2 * m, th - 2 * m; lo = 0.33
            pt = proc_tiles(r['recs'], sp, iw / (ih * (1 - lo)))
            for k in range(4):
                xk = xb + k * (t + g); box(S, xk, y, t, th, kinds[k][0], kinds[k][1], r=0.03); X, Y = xk + m, y + m
                if k == 0: S.image(X, Y, iw, ih, fit_png(r['img'], iw, ih), f'{sp} g{b} {i} image')
                elif k == 1:
                    xl_, yl_ = own_lims(r['gt'], iw, (1 - lo) * ih)
                    S.image(X, Y + lo * ih, iw, (1 - lo) * ih, cloud_png(r['gt'], iw, (1 - lo) * ih, xl_, yl_, s=0.22), f'{sp} g{b} {i} truth')
                    S.text(X + 0.05 * iw, Y + 0.03 * iw, f"{r['height']:.2f} m\n{r['leaves']} leaves", F, va='bottom', ls=1.1)
                else:
                    j = k - 2; S.image(X, Y + lo * ih, iw, (1 - lo) * ih, pt[j], f'{sp} g{b} {i} pred {j}')
                    S.text(X + 0.05 * iw, Y + 0.03 * iw, f"CD = {r['ch'][j]:.0f}\n{r['hp'][j]:.2f} m\n{r['lv'][j]} leaves", F,
                           color=STREAM_INK['text'] if j == 0 else OURS, va='bottom', ls=1.1)
            S.text(xb - 0.06, y + th / 2, f"{r['label']}\nplant\n{r['plant']}", 6.0, ha='right', va='center', ls=1.1)
            y -= gy
    return S, f'fig_e10_gallery_{sp}'


def field_e10(sp):
    D = pickle.load(open(D12 / f'supp_field_real_{sp}.pkl', 'rb'))
    W, g, m, lab = 6.875, 0.035, 0.02, 0.78
    heads = ['Image', 'Depth image', 'Point cloud\n(scene)', 'Center plant\n(truth)', 'No row\ntraining', 'Synthetic-row\nfine-tune', 'Field\nfine-tune']
    kinds = [('rgb', True), ('depth', True), ('pc', True), (None, False), ('text', False), ('text', False), ('text', False)]
    hc = [INK] * 4 + [STREAM_INK['text'], STREAM_INK['text'], OURS]
    t = (W - lab - 6 * g) / 7; th = t * 4 / 3; hdr, gy = 0.36, 0.07
    n = len(D['rows']); H = hdr + n * th + (n - 1) * gy + 0.02
    S = Scene(W, H, f'field rows {sp}'); F = 5.8
    for k, (s_, c) in enumerate(zip(heads, hc)):
        S.text(lab + k * (t + g) + t / 2, H - 0.01, s_, 6.4, color=c, ha='center', va='top', ls=1.05)
    xl, yl = (-0.78, 0.78), (-1.02, 1.02)
    y = H - hdr
    for i, r in enumerate(D['rows']):
        y -= th; iw, ih = t - 2 * m, th - 2 * m; lo = 0.27
        pt = proc_tiles(r['recs'], sp, iw / (ih * (1 - lo)))
        for k in range(7):
            xk = lab + k * (t + g); box(S, xk, y, t, th, kinds[k][0], kinds[k][1], r=0.03); X, Y = xk + m, y + m
            if k == 0: S.image(X, Y, iw, ih, fit_png(r['img'], iw, ih), f'{sp} {i} image')
            elif k == 1:
                d = np.asarray(r['depth'], np.float64); ok = np.isfinite(d); rgb = np.zeros(d.shape + (3,))
                if ok.any(): lo_, hi_ = d[ok].min(), d[ok].max(); rgb[ok] = plt.get_cmap('viridis_r')((d[ok] - lo_) / max(hi_ - lo_, 1e-9))[:, :3]
                S.image(X, Y, iw, ih, fit_png(rgb, iw, ih, 500), f'{sp} {i} depth')
            elif k in (2, 3):
                P = r['pin'] if k == 2 else r['pgt']
                if k == 3:
                    xl_, yl_ = own_lims(P, iw, (1 - lo) * ih)
                    S.image(X, Y + lo * ih, iw, (1 - lo) * ih, cloud_png(P, iw, (1 - lo) * ih, xl_, yl_, s=0.24), f'{sp} {i} cloud {k}')
                    S.text(X + 0.05 * iw, Y + 0.03 * iw, f"{r['height']:.2f} m\n{r['leaves']} leaves", F, va='bottom', ls=1.1)
                else:
                    S.image(X, Y, iw, ih, cloud_png(P, iw, ih, xl, yl, s=0.12), f'{sp} {i} cloud {k}')
            else:
                j = k - 4; S.image(X, Y + lo * ih, iw, (1 - lo) * ih, pt[j], f'{sp} {i} pred {j}')
                S.text(X + 0.05 * iw, Y + 0.03 * iw, r['texts'][j].replace(' · ', ', '), F, color=hc[k], va='bottom', ls=1.1)
        pl, row, stop = r['label']
        S.text(lab - 0.08, y + th / 2, f'plant\n{pl}\nrow {row}\nstop {stop}', 6.2, ha='right', va='center', ls=1.1)
        y -= gy
    return S, f"fig_e10_real{'_sorghum' if sp == 'sorghum' else ''}"


if __name__ == '__main__':
    want = sys.argv[1:] or ['s1', 's4', 's5', 's10', 's11']
    jobs = {'s1': stats, 's4': lambda: field_e10('maize'), 's5': lambda: field_e10('sorghum'), 's10': lambda: field_gallery('maize'),
            's11': lambda: field_gallery('sorghum'),
            's4old': lambda: e10('maize'), 's5old': lambda: e10('sorghum'), 's10old': lambda: gallery('maize'), 's11old': lambda: gallery('sorghum')}
    for k in want:
        S, name = jobs[k](); save(S, name)
