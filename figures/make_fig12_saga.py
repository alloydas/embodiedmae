"""Figures 1 and 2 in the layout of SAGA (Kundu et al., CVPR 2026): a three-panel teaser and a staged method figure.

Figure 1 (paper/figs/fig1_overview.*): (a) overview: crop rows in (image, depth image, point cloud; one held-out
  test row per species), CropCode as one box, the center plant's procedural parameters, generated point cloud and
  traits out (the tiles and text of the current teaser, paper/figs/fig_teaser.drawio); (b) t-SNE of the headline
  models' latents, coloured by a trait; (c) trait-probe radar against the published encoders and the supervised
  reference (paper Table 10).
Figure 2 (paper/figs/fig2_stages.*): Stage 1 four-stream pretraining, Stage 2 cross-modal distillation, Stage 3
  row fine-tuning, stacked as rounded panels with warm-start arrows; a zoom-in on the network at the right; each stage
  ends in its result (reconstruction, Table 6, Table 5).
Each figure is drawn by figures/_scene.py to PDF / PNG (matplotlib) and to an editable .drawio file.
Assets: python figures/make_fig12_assets.py (CPU job) first.   usage: python figures/make_fig12_saga.py [--fig2-tikz]
--fig2-tikz writes Figure 2 only, plus paper/overleaf/Figures/fig2_stages_tikz.tex and its images in Figures/fig2/ (the paper's Fig. 2).
"""
import sys, pathlib; R = pathlib.Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R), str(R / 'figures')]
import html, io, json, re
import xml.etree.ElementTree as ET
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from _scene import Scene, png_file, text_size

A = R / 'paper/figs/fig12_assets'; TILES = R / 'paper/overleaf/Figures/teaser'
INK, MUTED, NAVY, BOXFILL, LOSS = '#1d221c', '#5f685c', '#2b3a67', '#eef1f7', '#c0392b'
CS = {'rgb': '#c0392b', 'depth': '#b8860b', 'pc': '#2a9d8f', 'text': '#30638e'}
STAGE = {1: ('#e9b949', '#fdf8ea'), 2: ('#6aa84f', '#f1f8ee'), 3: ('#4a78b5', '#eef3fa')}


def png_fig(fig, dpi=400):
    b = io.BytesIO(); fig.savefig(b, format='png', dpi=dpi, facecolor='white'); plt.close(fig); return b.getvalue()


# ---------------------------------------------------------------- teaser text (from the current teaser's draw.io)
def teaser_text():
    out = {'sorghum': {}, 'maize': {}}
    for c in ET.parse(R / 'paper/figs/fig_teaser.drawio').iter('mxCell'):
        v = c.get('value') or ''
        if not v or 'image=' in (c.get('style') or ''): continue
        s = html.unescape(re.sub(r'<br>', '\n', re.sub(r'</?div[^>]*>', '', v))).replace('\xa0', ' ')
        sp = 'sorghum' if float(c.find('mxGeometry').get('y')) < 380 else 'maize'
        if re.match(r'(stem|height) [\d.]+ m, \d+ leaves', s): out[sp]['recipe'] = s
        elif s.startswith('height\n'): out[sp]['traits'] = s
        elif 'hidden by its neighbors' in s: out[sp]['occ'] = s.split(' %')[0]
        elif s.startswith('Chamfer'): out[sp]['chamfer'] = s.split(' vs')[0]
    return out


# ---------------------------------------------------------------- radar at its final size
def radar_png(w, h):
    axes = ['S height', 'S leaf cnt', 'S branch', 'S roll', 'S leaf len', 'M height', 'M leaf angle', 'M curl', 'M stem rad']
    tab = {'DINOv2': [0.924, 0.928, 0.840, 0.573, 0.235, 0.561, 0.816, 0.072, 0.588],
           'MultiMAE': [0.922, 0.925, 0.833, 0.679, 0.232, 0.570, 0.783, 0.069, 0.442],
           'Point-MAE': [0.958, 0.963, 0.863, 0.716, 0.188, 0.509, 0.863, 0.199, 0.426],
           'EmbodiedMAE': [0.976, 0.979, 0.871, 0.765, 0.265, 0.528, 0.823, 0.139, 0.428],
           'Supervised': [0.973, 0.978, 0.924, 0.634, 0.271, 0.785, 0.935, 0.618, 0.860],
           'CropCode': [0.983, 0.988, 0.912, 0.798, 0.371, 0.747, 0.967, 0.739, 0.841]}
    sty = {'DINOv2': ('#9aa5b1', '-'), 'MultiMAE': ('#c49a6c', '-'), 'Point-MAE': ('#7fb08a', '-'), 'EmbodiedMAE': ('#b48ead', '-'),
           'Supervised': ('#555555', (0, (2.5, 1.5))), 'CropCode': (NAVY, '-')}
    M = np.array(list(tab.values())); lo, hi = M.min(0), M.max(0)
    ang = np.linspace(0, 2 * np.pi, len(axes), endpoint=False); angc = np.r_[ang, ang[:1]]
    fig = plt.figure(figsize=(w, h)); ax = fig.add_axes([0.25, 0.31, 0.50, 0.53], projection='polar')
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    for name, v in tab.items():
        r = 0.25 + 0.75 * (np.array(v) - lo) / np.maximum(hi - lo, 1e-9); r = np.r_[r, r[:1]]
        c, ls = sty[name]; ours = name == 'CropCode'
        ax.plot(angc, r, color=c, lw=1.5 if ours else 0.7, ls=ls, zorder=5 if ours else 2, label=name)
        if ours: ax.fill(angc, r, color=c, alpha=0.13, zorder=1)
    ax.set_xticks(ang); ax.set_xticklabels([f'{a}\n{v:.2f}' for a, v in zip(axes, tab['CropCode'])], fontsize=3.7, color=INK, linespacing=0.95)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels([]); ax.set_ylim(0, 1.04)
    ax.grid(color='#dddddd', lw=0.4); ax.spines['polar'].set_color('#cccccc'); ax.spines['polar'].set_linewidth(0.4); ax.tick_params(axis='x', pad=0)
    fig.legend(loc='lower center', ncol=3, fontsize=4.0, frameon=False, bbox_to_anchor=(0.5, -0.01), handlelength=1.5, columnspacing=0.6)
    return png_fig(fig)


def tsne_png(sp, w, h, label, cmap):
    z = np.load(A / f'tsne_{sp}.npz'); E, y = z['E'], z['y']
    fig = plt.figure(figsize=(w, h)); ax = fig.add_axes([0.02, 0.02, 0.68, 0.86])
    o = np.argsort(y); sc = ax.scatter(E[o, 0], E[o, 1], c=y[o], cmap=cmap, s=0.6, linewidths=0, vmin=np.percentile(y, 2), vmax=np.percentile(y, 98))
    ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
    ax.set_title(sp, fontsize=5.5, pad=1, color=INK)
    cax = fig.add_axes([0.72, 0.12, 0.05, 0.66]); cb = fig.colorbar(sc, cax=cax)
    cb.ax.tick_params(labelsize=4, length=1.2, width=0.3, pad=1); cb.outline.set_linewidth(0.3); cb.set_label(label, fontsize=4.4, labelpad=1)
    return png_fig(fig)


# ======================================================================= Figure 1
def figure1():
    W, H = 7.0, 2.42
    S = Scene(W, H, 'Fig 1 overview'); T = 0.50
    tx = teaser_text()
    # ---- (a) overview
    y_rows = {'sorghum': 1.32, 'maize': 0.52}
    X_IN = [0.22, 0.74, 1.26]; XB, WB = 1.88, 0.84; XR, WR = 2.86, 1.08; XG = 4.06; XT, WT = 4.62, 0.62
    hy = 1.88
    for lab, x, c in (('Image', X_IN[0], CS['rgb']), ('Depth\nimage', X_IN[1], CS['depth']), ('Point\ncloud', X_IN[2], CS['pc'])):
        S.text(x + T / 2, hy, lab, 5.2, color=c, ha='center', va='bottom', bold=True, ls=1.05)
    S.text(X_IN[0], hy + 0.26, 'Occluded crop rows, as a rover sees them', 6.2, bold=True)
    S.text(XR, hy + 0.26, 'The center plant, written out', 6.2, bold=True)
    S.text(XR + WR / 2, hy, 'Procedural\nparameters', 5.2, color=CS['text'], ha='center', va='bottom', bold=True, ls=1.05)
    S.text(XG + T / 2, hy, 'Generated\npoint cloud', 5.2, color=CS['pc'], ha='center', va='bottom', bold=True, ls=1.05)
    S.text(XT + WT / 2, hy, 'Traits', 5.2, ha='center', va='bottom', bold=True)
    # the model box, with SAGA-style stacked internals
    by0, by1 = y_rows['maize'], y_rows['sorghum'] + T
    S.rect(XB, by0 - 0.02, WB, by1 - by0 + 0.04, NAVY, lw=1.1, fill=BOXFILL, r=0.07)
    S.text(XB + WB / 2, by1 - 0.09, 'CropCode', 7, color=NAVY, ha='center', bold=True)
    inner = [('decoders: image, depth,\ncloud, parameters', '#fbe3c4'), ('shared ViT encoder', '#d9e2f3'),
             ('Dirichlet token masking', '#e4efe0'), ('tokenizers per stream', '#f3e0e0')]
    ih, gap = 0.19, 0.055; y0 = by0 + 0.06
    for k, (s, f) in enumerate(reversed(inner)):
        yy = y0 + k * (ih + gap)
        S.rect(XB + 0.07, yy, WB - 0.14, ih, NAVY, lw=0.5, fill=f, r=0.03)
        S.text(XB + WB / 2, yy + ih / 2, s, 3.7, color=NAVY, ha='center', ls=1.1)
        if k: S.arrow((XB + WB / 2, yy - gap + 0.005), (XB + WB / 2, yy - 0.005), color=NAVY, lw=0.5)
    for sp in ('sorghum', 'maize'):
        y = y_rows[sp]; p = 's' if sp == 'sorghum' else 'm'
        S.text(0.10, y + T / 2, sp, 6, ha='center', bold=True, rot=90)
        for k, (nm, c) in enumerate((('rgb', CS['rgb']), ('depth', CS['depth']), ('pc', CS['pc']))):
            S.image(X_IN[k], y, T, T, png_file(TILES / f'{p}_{nm}.png'), f'{sp} {nm}'); S.rect(X_IN[k], y, T, T, c, lw=0.8)
        S.text(X_IN[0], y - 0.05, f"{tx[sp]['occ']} % of the center plant hidden by neighbors", 4.2, color=MUTED, va='top')
        S.arrow((X_IN[2] + T + 0.03, y + T / 2), (XB - 0.03, y + T / 2))
        S.arrow((XB + WB + 0.03, y + T / 2), (XR - 0.03, y + T / 2), color=CS['text'])
        S.rect(XR, y, WR, T, CS['text'], lw=0.8)
        rec = '\n'.join(tx[sp]['recipe'].split('\n')[:5])
        S.text(XR + 0.03, y + T - 0.03, rec, 3.15, color=CS['text'], va='top', mono=True, ls=1.25)
        S.arrow((XR + WR + 0.03, y + T / 2), (XG - 0.03, y + T / 2), color=CS['pc'])
        S.image(XG, y, T, T, png_file(TILES / f'{p}_gen.png'), f'{sp} generated'); S.rect(XG, y, T, T, CS['pc'], lw=0.8)
        S.text(XG + T / 2, y - 0.05, tx[sp]['chamfer'], 4.2, color=MUTED, ha='center', va='top')
        S.rect(XT, y, WT, T, INK, lw=0.5)
        S.text(XT + 0.03, y + T - 0.03, tx[sp]['traits'], 3.7, va='top', ls=1.13)
    S.text(2.65, 0.07, '(a) One model: any field sensor in, the procedural plant out', 6, ha='center')
    # ---- (b) t-SNE  (c) radar
    XRt, WRt = 5.40, 1.60
    S.arrow((XRt - 0.10, 0.12), (XRt - 0.10, 2.32), color='#cccccc', lw=0.5, head=False)
    S.image(XRt, 1.58, 0.80, 0.76, tsne_png('sorghum', 0.80, 0.76, 'leaf count', 'Blues'), 't-SNE sorghum')
    S.image(XRt + 0.80, 1.58, 0.80, 0.76, tsne_png('maize', 0.80, 0.76, 'height (m)', 'Greens'), 't-SNE maize')
    S.text(XRt + WRt / 2, 1.50, '(b) Latents order plants by their traits', 5.0, ha='center')
    S.image(XRt, 0.15, WRt, 1.27, radar_png(WRt, 1.27), 'radar')
    S.text(XRt + WRt / 2, 0.07, '(c) Trait probe vs published encoders', 5.0, ha='center')
    return S


# ======================================================================= Figure 2
def bars_png(kind, w, h):
    fig, axs = plt.subplots(1, 2, figsize=(w, h))
    if kind == 'distill':        # paper Table 6: single-stream cloud Chamfer (x1e-3), before -> after distillation
        data = (('sorghum', [2.27, 0.90, 5.48], [0.88, 0.63, 1.39]), ('maize', [11.3, 1.26, 6.43], [0.77, 0.58, 0.83]))
        for ax, (sp, b, a_) in zip(axs, data):
            x = np.arange(3)
            ax.bar(x - 0.2, b, 0.38, color='#d9d4c7', label='before'); ax.bar(x + 0.2, a_, 0.38, color=NAVY, label='after')
            ax.set_xticks(x); ax.set_xticklabels(['RGB', 'D', 'PC'], fontsize=3.8); ax.set_yscale('log'); ax.set_ylim(0.3, 20)
            ax.set_title(sp, fontsize=4.4, pad=1)
        axs[0].set_ylabel('Chamfer ×1e-3', fontsize=3.8, labelpad=1); axs[1].legend(fontsize=3.4, frameon=False, loc='upper right', handlelength=0.8, borderpad=0.1)
    else:                        # paper Table 5: Chamfer on rows (x1e-4 m^2), all three streams in
        data = (('sorghum', [230.5, 133.8, 30.7]), ('maize', [29.6, 29.0, 4.0]))
        for ax, (sp, v) in zip(axs, data):
            ax.bar(np.arange(3), v, 0.62, color=['#d9d4c7', '#a7b3c9', NAVY])
            ax.set_xticks(range(3)); ax.set_xticklabels(['zero-\nshot', 'iso-\nlated', 'row\nFT'], fontsize=2.9, linespacing=0.95)
            for k, vv in enumerate(v): ax.text(k, vv, f'{vv:g}', ha='center', va='bottom', fontsize=3.3)
            ax.set_ylim(0, max(v) * 1.25); ax.set_title(sp, fontsize=4.4, pad=1)
        axs[0].set_ylabel('Chamfer on rows\n×1e-4 m²', fontsize=3.6, labelpad=1, linespacing=0.95)
    for ax in axs:
        ax.tick_params(labelsize=3.5, length=1.2, width=0.3, pad=1)
        for sp_ in ax.spines.values(): sp_.set_linewidth(0.3)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.subplots_adjust(left=0.27 if kind == 'rows' else 0.2, right=0.98, bottom=0.27 if kind == 'rows' else 0.2, top=0.86, wspace=0.5)
    return png_fig(fig)


def figure2():
    """Fig. 2 at the paper's text width (6.875 in, drawn at 100 %), every label 9 pt. Real tiles throughout: Stage 1 and
    Stage 2 use one held-out sorghum test plant (Sorghum_10036, view 00), Stage 3 one held-out test row (Sorghum_10113,
    broadside). Data: figures/make_fig12_dump.py. The procedural parameters are drawn as the stem-and-leaves schematic
    (insertion height, length, width and side of each leaf), not as text. Numbers: sorghum test means of Table
    tab:anysubset (photograph to cloud, x1e-3) and tab:rows (all streams, x1e-4 m^2)."""
    import _fig12_kit as K
    from _scene import cloud_frame
    F = 9.0                                                     # the one type size
    D = K.load_dump(); d = D['stage']['sorghum']['Sorghum_10036_00']; r = D['rows']['sorghum']['Sorghum_10113']
    W, H = 6.875, 4.42
    S = Scene(W, H, 'Fig 2 stages')
    V = dict(elev=10.0, azim=20.0)
    fr = cloud_frame(d['pc'], **V); frr = cloud_frame(r['gt_pc'], **V)
    hidden_c = '#d3d6cf'

    def glyph_png(rec, side, vis=None):
        """the parameter schematic; vis: per-token visibility (token 0 = plant, 1.. = leaves): hidden leaves light grey"""
        fig = plt.figure(figsize=(side, side), dpi=600); ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
        if vis is not None:
            K.glyph(ax, 'sorghum', rec, hidden_c, stem=hidden_c, lw_scale=0.55)
            keep = dict(rec, leaves=[l for i, l in enumerate(rec['leaves']) if vis[1 + i]])
            if keep['leaves'] or vis[0]: K.glyph(ax, 'sorghum', keep, CS['text'], stem='#4a3b2a' if vis[0] else hidden_c, lw_scale=0.55)
        else:
            K.glyph(ax, 'sorghum', rec, CS['text'], lw_scale=0.55)
        Hh = rec['height']; ax.set_xlim(-0.55 * Hh, 0.55 * Hh); ax.set_ylim(-0.02 * Hh, 1.08 * Hh); ax.set_aspect('equal'); ax.set_axis_off()
        return K.png_fig(fig, 600)

    def pred_depth(pn):
        z = pn * (d['depth'].max() - d['depth'].min()) + d['depth'].min(); z = z.copy(); z[pn < 0.05] = 0; return z

    def grid(x, y, t, tiles, g=0.03):
        """2 x 2 tiles: image, depth image / point cloud, parameters; each (png, colour, name)"""
        for i, (png, c, nm) in enumerate(tiles):
            K.tile(S, x + (i % 2) * (t + g), y + (1 - i // 2) * (t + g), t, png, c, lw=0.8, name=nm)
        return 2 * t + g

    # ---------------------------------------------------------------- stage panels
    XL, XR = 0.40, 4.98
    bands = {1: (2.72, 1.56), 2: (1.40, 1.16), 3: (0.04, 1.20)}
    titles = {1: 'Stage 1: masked pretraining', 2: 'Stage 2: cross-modal distillation', 3: 'Stage 3: row fine-tuning'}
    for k, (y, h) in bands.items():
        stroke, fill = K.STAGE[k]
        S.rect(XL, y, XR - XL, h, stroke, lw=1.2, fill=fill, r=0.09)
        tw = text_size(titles[k], F, bold=True)[0] + 0.18
        S.rect(XL + 0.14, y + h - 0.11, tw, 0.22, stroke, lw=0.9, fill='#ffffff', r=0.05, z=0)
        S.text(XL + 0.23, y + h, titles[k], F, bold=True)
    # warm start: Stages 2 and 3 begin from the Stage 1 weights
    S.arrow((XL, 3.40), (XL, 2.00), color=LOSS, lw=1.1, curve=0.22)
    S.arrow((XL, 3.20), (XL, 0.66), color=LOSS, lw=1.1, curve=0.11)
    S.text(0.10, 2.00, 'warm start', F, color=LOSS, rot=90, ha='center', bold=True)

    # ---------------------------------------------------------------- Stage 1
    t, y1 = 0.40, 3.03
    full = [(K.tile_img(d['rgb']), CS['rgb'], 'image'), (K.tile_depth(d['depth']), CS['depth'], 'depth image'),
            (K.tile_cloud(d['pc'], fr, t, s=0.35, **V), CS['pc'], 'point cloud'), (glyph_png(d['recipe_true'], t), CS['text'], 'parameters')]
    gw = grid(0.52, y1, t, full)
    vis_t = [m < 0.5 for m in d['mask']['text']]
    masked = [(K.tile_img(K.masked(d['rgb'], d['mask']['rgb'])), CS['rgb'], 'image masked'),
              (K.tile_img(K.masked(K.depth_rgb(d['depth']), d['mask']['depth'])), CS['depth'], 'depth masked'),
              (K.tile_masked_cloud(d['pc'], d['pc_centers'], d['mask']['pc'], fr, t, s=0.35, **V), CS['pc'], 'cloud masked'),
              (glyph_png(d['recipe_true'], t, vis_t), CS['text'], 'parameters masked')]
    grid(1.55, y1, t, masked)
    rec = [(K.tile_img(d['rec']['rgb']), CS['rgb'], 'image output'), (K.tile_depth(pred_depth(d['rec']['depth'])), CS['depth'], 'depth output'),
           (K.tile_cloud(d['rec']['pc'], fr, t, s=0.35, **V), CS['pc'], 'cloud output'), (glyph_png(d['rec']['recipe'], t), CS['text'], 'parameters output')]
    grid(4.06, y1, t, rec)
    yl = y1 + gw + 0.10
    S.text(0.52 + gw / 2, yl, 'Input', F, ha='center', va='bottom', bold=True)
    S.text(1.55 + gw / 2, yl, '80 % hidden', F, ha='center', va='bottom', bold=True)
    S.text(4.06 + gw / 2, yl, 'Output', F, ha='center', va='bottom', bold=True)
    S.arrow((0.52 + gw + 0.03, y1 + gw / 2), (1.55 - 0.03, y1 + gw / 2))
    S.arrow((1.55 + gw + 0.03, y1 + gw / 2), (2.62 - 0.03, y1 + gw / 2))
    K.model(S, 2.62, y1, 1.28, gw, size=F, names=('Encoder', 'Decoder'))
    K.icon(S, 'fire', 2.62 + 0.66, y1 + gw - 0.05, 0.17)
    S.arrow((2.62 + 1.28 + 0.03, y1 + gw / 2), (4.06 - 0.03, y1 + gw / 2))
    S.arrow((4.06 + gw / 2, y1 - 0.08), (0.52 + gw / 2, y1 - 0.08), color=LOSS, lw=0.9, dash=True)
    S.text(2.69, y1 - 0.13, 'loss on the hidden tokens', F, color=LOSS, ha='center', va='top')

    # ---------------------------------------------------------------- Stage 2
    ts, y2 = 0.26, 1.62
    S.rect(0.52, y2 - 0.07, 1.25, 0.72, '#888888', lw=0.9, fill='#ffffff', r=0.05)
    K.icon(S, 'snow', 0.68, y2 + 0.49, 0.17)
    S.text(0.80, y2 + 0.49, 'Teacher', F, bold=True)
    for i, (png, c, nm) in enumerate(full):
        K.tile(S, 0.58 + i * (ts + 0.04), y2, ts, png, c, lw=0.6, name=f'teacher {nm}')
    S.arrow((1.80, y2 + 0.26), (2.27, y2 + 0.26), color=LOSS, lw=0.9, dash=True)
    S.text(2.035, y2 + 0.31, 'features', F, color=LOSS, ha='center', va='bottom')
    S.rect(2.30, y2 - 0.07, 1.25, 0.72, NAVY, lw=1.0, fill='#ffffff', r=0.05)
    K.icon(S, 'fire', 2.46, y2 + 0.49, 0.17)
    S.text(2.58, y2 + 0.49, 'Student', F, color=NAVY, bold=True)
    K.tile(S, 2.36, y2, ts, full[0][0], CS['rgb'], lw=0.9, name='student image')
    for i, c in enumerate((CS['depth'], CS['pc'], CS['text'])):
        K.tile(S, 2.36 + (i + 1) * (ts + 0.04), y2, ts, K.tile_hidden(c), c, lw=0.6, dash=True, name=f'student hidden {i}')
    S.arrow((3.58, y2 + 0.29), (3.72, y2 + 0.29))
    to = 0.58; xo = (3.75, 4.37)
    for x, key, lab, c in ((xo[0], 'rgb:before', 'before', MUTED), (xo[1], 'rgb:after', 'after', NAVY)):
        K.tile(S, x, y2 + 0.02, to, K.tile_cloud(d['one'][key]['pc'], fr, to, s=0.4, **V), CS['pc'] if lab == 'after' else MUTED,
               lw=0.8, dash=lab == 'before', name=f'stage 2 {lab}')
        S.text(x + to / 2, y2 + 0.02 + to + 0.05, lab, F, color=c, ha='center', va='bottom', bold=lab == 'after')
    S.text((xo[0] + xo[1] + to) / 2, y2 - 0.04, 'Chamfer 2.27 → 0.88', F, ha='center', va='top')

    # ---------------------------------------------------------------- Stage 3
    tr, y3 = 0.40, 0.44
    for i, (png, c, nm) in enumerate(((K.tile_rgba(r['rgba'], r['gain']), CS['rgb'], 'row image'), (K.tile_depth(r['depth448']), CS['depth'], 'row depth'),
                                      (K.tile_cloud(r['scene'], cloud_frame(r['scene'], **V), tr, s=0.18, **V), CS['pc'], 'row cloud'))):
        K.tile(S, 0.52 + i * (tr + 0.04), y3, tr, png, c, lw=0.8, name=nm)
    S.text(0.52 + (3 * tr + 0.08) / 2, y3 + tr + 0.08, 'Crop row', F, ha='center', va='bottom', bold=True)
    S.arrow((1.85, y3 + tr / 2), (2.20, y3 + tr / 2))
    K.model(S, 2.23, y3 - 0.02, 1.20, tr + 0.04, size=F, names=('Enc', 'Dec'))
    K.icon(S, 'fire', 2.23 + 0.62, y3 + tr - 0.03, 0.17)
    S.text(2.83, y3 - 0.09, 'loss: center plant only', F, color=LOSS, ha='center', va='top')
    S.arrow((3.46, y3 + tr / 2), (3.72, y3 + tr / 2))
    for x, key, lab, c in ((xo[0], 'head:all', 'before', MUTED), (xo[1], 'ft:all', 'after', NAVY)):
        K.tile(S, x, y3 - 0.06, to, K.tile_cloud(r['pred'][key]['pc'], frr, to, s=0.4, **V), CS['pc'] if lab == 'after' else MUTED,
               lw=0.8, dash=lab == 'before', name=f'stage 3 {lab}')
        S.text(x + to / 2, y3 - 0.06 + to + 0.05, lab, F, color=c, ha='center', va='bottom', bold=lab == 'after')
    S.text((xo[0] + xo[1] + to) / 2, y3 - 0.12, 'Chamfer 230 → 31', F, ha='center', va='top')

    # ---------------------------------------------------------------- Inside CropCode
    XZ, WZ = 5.10, W - 0.02 - 5.10
    S.rect(XZ, 0.04, WZ, 4.24, '#8a8a8a', lw=1.3, fill='#f4f4f2', r=0.09)
    S.text(XZ + WZ / 2, 4.06, 'Inside CropCode', F, ha='center', bold=True)
    a, gp = 0.12, 0.035
    for i, (nm, c, hid) in enumerate((('Image  196', CS['rgb'], (1, 3)), ('Depth  196', CS['depth'], (0, 2, 3)),
                                      ('Cloud  196', CS['pc'], (2,)), ('Params  1+K', CS['text'], (1, 2)))):
        yy = 3.65 - i * 0.30
        K.tok_row(S, XZ + 0.14, yy, 4, a=a, gap=gp, color=c, hidden=hid, lw=0.6)
        S.text(XZ + 0.14 + 4 * a + 3 * gp + 0.10, yy + a / 2, nm, F, color=c, va='center', bold=True)
    S.text(XZ + 0.14, 2.52, 'tokens (hollow = hidden)', F, color=MUTED, va='center')
    S.text(XZ + 0.14, 2.20, 'Dirichlet split', F, va='center', bold=True)
    x = XZ + 0.14; tot = sum(d['visible'].values()); bw = WZ - 0.28
    for kk in ('rgb', 'depth', 'pc', 'text'):
        w_ = bw * d['visible'][kk] / tot; S.rect(x, 1.92, w_, 0.13, CS[kk], lw=0.3, fill=CS[kk], r=0.01, z=0); x += w_
    S.text(XZ + 0.14, 1.76, 'of the visible tokens', F, color=MUTED, va='center')
    for (lab, sub, fill, y0) in (('Encoder', '12 blocks', K.ENC_FILL, 1.06), ('Decoder', '8 blocks', K.DEC_FILL, 0.50)):
        S.rect(XZ + 0.14, y0, WZ - 0.28, 0.42, NAVY, lw=0.8, fill=fill, r=0.04)
        S.text(XZ + WZ / 2, y0 + 0.21, f'{lab}, {sub}', F, color=NAVY, ha='center', bold=True)
    S.arrow((XZ + WZ / 2, 1.62), (XZ + WZ / 2, 1.49), color=NAVY, lw=0.8)
    S.arrow((XZ + WZ / 2, 1.05), (XZ + WZ / 2, 0.93), color=NAVY, lw=0.8)
    for i, (lab, c) in enumerate((('pixels', CS['rgb']), ('depth', CS['depth']), ('points', CS['pc']), ('params', CS['text']))):
        wq = (WZ - 0.28 - 0.09) / 4; xx = XZ + 0.14 + i * (wq + 0.03)
        S.rect(xx, 0.13, wq, 0.10, c, lw=0.5, fill=c, r=0.02, z=0)
    S.arrow((XZ + WZ / 2, 0.49), (XZ + WZ / 2, 0.42), color=NAVY, lw=0.8)
    S.text(XZ + WZ / 2, 0.32, 'one head per stream', F, color=NAVY, ha='center')
    return S


if __name__ == '__main__':
    out = R / 'paper/figs'
    if '--fig2-tikz' in sys.argv:               # Figure 2 only, as native TikZ for the paper (+ the PDF / PNG / draw.io copies)
        S = figure2(); S.render(out / 'fig2_stages')
        S.render_tikz(R / 'paper/overleaf/Figures/fig2_stages_tikz.tex', R / 'paper/overleaf/Figures/fig2', 'Figures/fig2')
    else:
        figure1().render(out / 'fig1_overview')
        figure2().render(out / 'fig2_stages')
