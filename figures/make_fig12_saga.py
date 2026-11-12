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
from _scene import Scene, png_file

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
    W, H = 7.0, 3.82
    S = Scene(W, H, 'Fig 2 stages'); te = json.load(open(A / 'text.json'))
    XL, XS1 = 0.42, 5.30                   # stage panels span x XL..XS1
    bands = {1: (2.44, 1.18), 2: (1.27, 1.02), 3: (0.10, 1.02)}   # y, height
    titles = {1: 'Stage 1: four-stream pretraining on single plants', 2: 'Stage 2: cross-modal distillation (any one stream)',
              3: 'Stage 3: row fine-tuning under occlusion'}
    for k, (y, h) in bands.items():
        stroke, fill = STAGE[k]
        S.rect(XL, y, XS1 - XL, h, stroke, lw=1.3, fill=fill, r=0.08)
        S.rect(XL + 0.12, y + h - 0.075, 2.40, 0.15, stroke, lw=0.9, fill='#ffffff', r=0.04, z=0)
        S.text(XL + 0.19, y + h + 0.0, titles[k], 5.6, color=INK, bold=True)
    # warm start: Stage 2 (teacher and student) and Stage 3 both start from the Stage 1 model
    S.arrow((XL, 3.00), (XL, 1.84), color=LOSS, lw=1.2, curve=0.20)
    S.arrow((XL, 2.80), (XL, 0.62), color=LOSS, lw=1.2, curve=0.10)
    S.text(0.07, 1.75, 'warm start from Stage 1', 4.6, color=LOSS, rot=90, ha='center', bold=True)
    t, g = 0.36, 0.03

    def grid(x, y, names):
        cols = [CS['rgb'], CS['depth'], CS['pc'], CS['text']]
        for i, nm in enumerate(names):
            xx, yy = x + (i % 2) * (t + g), y + (1 - i // 2) * (t + g)
            if nm.endswith('.png'): S.image(xx, yy, t, t, png_file(A / nm), nm)
            else:
                S.rect(xx, yy, t, t, cols[i], lw=0.6, fill='#ffffff', z=0)
                S.text(xx + 0.02, yy + t - 0.02, '\n'.join(l.replace('stem ', '')[:21] for l in nm.split('\n')), 2.4, color=cols[i], va='top', mono=True, ls=1.15)
            S.rect(xx, yy, t, t, cols[i], lw=0.6)
    # ---- Stage 1
    y1 = 2.66; gw = 2 * t + g
    grid(0.58, y1, ['s1_rgb.png', 's1_depth.png', 's1_pc.png', '\n'.join(te['recipe_true'][:4])])
    S.text(0.58 + gw / 2, y1 + gw + 0.05, 'four streams', 4.6, ha='center', bold=True)
    S.arrow((0.58 + gw + 0.03, y1 + gw / 2), (1.55 - 0.03, y1 + gw / 2))
    v = te['visible']
    grid(1.55, y1, ['s1_rgb_masked.png', 's1_depth_masked.png', 's1_pc_masked.png', '(hidden)'])
    S.text(1.55 + gw / 2, y1 + gw + 0.05, '80 % of tokens hidden', 4.6, ha='center', bold=True)
    S.text(1.55 + gw / 2, y1 - 0.02, f"visible tokens {v['rgb']} / {v['depth']} / {v['pc']} / {v['text']}", 3.4, color=MUTED, ha='center', va='top')
    S.arrow((1.55 + gw + 0.03, y1 + gw / 2), (2.52 - 0.03, y1 + gw / 2))
    S.rect(2.52, y1 + 0.06, 0.62, 0.63, NAVY, lw=0.9, fill='#d9e2f3', r=0.05)
    S.text(2.83, y1 + gw / 2, 'shared\nViT-B\nencoder', 5, color=NAVY, ha='center', bold=True)
    S.arrow((3.17, y1 + gw / 2), (3.31, y1 + gw / 2))
    for i, (nm, c) in enumerate((('image', CS['rgb']), ('depth', CS['depth']), ('cloud', CS['pc']), ('parameters', CS['text']))):
        S.rect(3.34, y1 + 0.58 - i * 0.18, 0.52, 0.14, c, lw=0.6, fill='#ffffff', r=0.03, z=0)
        S.text(3.60, y1 + 0.65 - i * 0.18, f'{nm} head', 3.6, color=c, ha='center')
    S.arrow((3.89, y1 + gw / 2), (4.03, y1 + gw / 2))
    grid(4.06, y1, ['s1_rec_rgb.png', 's1_rec_depth.png', 's1_rec_pc.png', '\n'.join(te['recipe_rec'][:4])])
    S.text(4.06 + gw / 2, y1 + gw + 0.05, 'reconstruction', 4.6, ha='center', bold=True)
    S.arrow((4.06 + gw / 2, y1 - 0.10), (0.58 + gw / 2, y1 - 0.10), color=LOSS, lw=0.8, dash=True)
    S.text(2.82, y1 - 0.13, 'losses: image and depth MSE and parameter Smooth-L1 on hidden tokens; QAL + Sinkhorn on the whole cloud',
           3.6, color=LOSS, ha='center', va='top')
    # ---- Stage 2
    y2 = 1.36
    S.rect(0.58, y2 + 0.04, 1.18, 0.70, '#888888', lw=0.8, fill='#ffffff', r=0.05)
    S.text(1.17, y2 + 0.62, '❄ teacher (frozen)', 4.8, ha='center', bold=True)
    S.text(1.17, y2 + 0.38, 'reads all four\nstreams', 3.9, color=MUTED, ha='center')
    for i, c in enumerate(CS.values()):
        S.rect(0.68 + i * 0.25, y2 + 0.10, 0.2, 0.08, c, lw=0.4, fill=c, r=0.01, z=0)
    S.rect(2.10, y2 + 0.04, 1.18, 0.70, NAVY, lw=0.9, fill='#ffffff', r=0.05)
    S.text(2.69, y2 + 0.62, 'student', 4.8, color=NAVY, ha='center', bold=True)
    S.text(2.69, y2 + 0.42, 'reads one stream,\nthe rest hidden', 3.9, color=MUTED, ha='center')
    S.image(2.42, y2 + 0.08, 0.2, 0.2, png_file(A / 's1_rgb.png'), 'student input'); S.rect(2.42, y2 + 0.08, 0.2, 0.2, CS['rgb'], lw=0.5)
    S.text(2.66, y2 + 0.18, 'e.g. the image', 3.7, color=CS['rgb'])
    S.arrow((1.79, y2 + 0.44), (2.07, y2 + 0.44), color=LOSS, lw=0.8, dash=True)
    S.text(1.93, y2 + 0.50, 'features\n+ latent', 3.5, color=LOSS, ha='center', va='bottom', ls=1.05)
    S.arrow((3.31, y2 + 0.39), (3.50, y2 + 0.39))
    S.image(3.52, y2 + 0.04, 0.72, 0.72, png_file(A / 's2_gen_pc.png'), 'cloud from image'); S.rect(3.52, y2 + 0.04, 0.72, 0.72, CS['pc'], lw=0.6)
    S.text(3.88, y2 + 0.01, 'cloud from the image alone', 3.5, color=MUTED, ha='center', va='top')
    S.image(4.30, y2 + 0.02, 0.95, 0.76, bars_png('distill', 0.95, 0.76), 'Table 6')
    # ---- Stage 3
    y3 = 0.19
    for i, (nm, c) in enumerate((('s_rgb', CS['rgb']), ('s_depth', CS['depth']), ('s_pc', CS['pc']))):
        S.image(0.58 + i * 0.41, y3 + 0.18, 0.37, 0.37, png_file(TILES / f'{nm}.png'), nm); S.rect(0.58 + i * 0.41, y3 + 0.18, 0.37, 0.37, c, lw=0.6)
    S.text(1.17, y3 + 0.63, 'a crop row (three streams)', 4.6, ha='center', bold=True)
    S.text(1.17, y3 + 0.13, 'nothing marks the center plant', 3.6, color=MUTED, ha='center', va='top')
    S.arrow((1.81, y3 + 0.37), (2.07, y3 + 0.37))
    S.rect(2.10, y3 + 0.05, 1.18, 0.66, NAVY, lw=0.9, fill='#ffffff', r=0.05)
    S.text(2.69, y3 + 0.52, 'row fine-tune', 4.8, color=NAVY, ha='center', bold=True)
    S.text(2.69, y3 + 0.28, 'loss against the\ncenter plant alone', 3.9, color=MUTED, ha='center')
    S.arrow((3.31, y3 + 0.37), (3.50, y3 + 0.37))
    S.image(3.52, y3 + 0.02, 0.72, 0.72, png_file(TILES / 's_gen.png'), 'center plant'); S.rect(3.52, y3 + 0.02, 0.72, 0.72, CS['pc'], lw=0.6)
    S.text(3.88, y3 - 0.01, 'the center plant, written out', 3.5, color=MUTED, ha='center', va='top')
    S.image(4.30, y3 + 0.0, 0.95, 0.76, bars_png('rows', 0.95, 0.76), 'Table 5')
    # ---- zoom-in on the network
    XZ, WZ = 5.42, 1.52
    S.rect(XZ, 0.10, WZ, 3.52, '#8a8a8a', lw=1.6, fill='#f4f4f4', r=0.1)
    S.text(XZ + WZ / 2, 3.47, 'Inside CropCode', 6, ha='center', bold=True)
    rows = [('image: 16 px patches → 196 tokens', CS['rgb']), ('depth: 16 px patches → 196 tokens', CS['depth']),
            ('cloud: FPS + kNN groups,\nPointNet → 196 tokens', CS['pc']), ('parameters: 1 plant + K leaf\ntokens (character embedding)', CS['text'])]
    yy = 3.28
    for s_, c in rows:
        hh = 0.27 if '\n' in s_ else 0.17
        S.rect(XZ + 0.08, yy - hh, WZ - 0.16, hh, c, lw=0.6, fill='#ffffff', r=0.03, z=0)
        S.text(XZ + 0.13, yy - hh / 2, s_, 3.8, color=c, ls=1.1); yy -= hh + 0.05
    S.text(XZ + WZ / 2, yy - 0.02, '+ position and stream embeddings', 3.7, color=MUTED, ha='center', va='top'); yy -= 0.24
    S.text(XZ + 0.10, yy, 'Dirichlet split of the visible budget', 3.9, va='top', bold=True); yy -= 0.17
    x = XZ + 0.10
    for frac, c in zip((0.35, 0.15, 0.30, 0.20), CS.values()):
        S.rect(x, yy - 0.10, (WZ - 0.20) * frac, 0.09, c, lw=0.3, fill=c, r=0.005, z=0); x += (WZ - 0.20) * frac
    S.text(XZ + WZ / 2, yy - 0.14, 'one draw per batch; every stream\nkeeps at least 25 % hidden', 3.4, color=MUTED, ha='center', va='top', ls=1.1)
    yy -= 0.44
    for s_, f, hh in (('ViT-B encoder: 12 blocks, 768-d,\n[CLS] + visible tokens only', '#d9e2f3', 0.30),
                      ('decoder: 8 blocks, 512-d, mask\ntokens back at their positions', '#fbe3c4', 0.30)):
        S.rect(XZ + 0.08, yy - hh, WZ - 0.16, hh, NAVY, lw=0.6, fill=f, r=0.03, z=0)
        S.text(XZ + WZ / 2, yy - hh / 2, s_, 3.7, color=NAVY, ha='center', ls=1.1); yy -= hh + 0.09
        S.arrow((XZ + WZ / 2, yy + 0.085), (XZ + WZ / 2, yy + 0.005), color=NAVY, lw=0.5)
    heads = [('patch pixels', CS['rgb']), ('patch depth', CS['depth']), ('folding → 8,192 pts', CS['pc']), ('9 / 14 floats per token', CS['text'])]
    for i, (s_, c) in enumerate(heads):
        S.rect(XZ + 0.08 + (i % 2) * 0.69, yy - 0.17 - (i // 2) * 0.21, 0.66, 0.17, c, lw=0.6, fill='#ffffff', r=0.03, z=0)
        S.text(XZ + 0.41 + (i % 2) * 0.69, yy - 0.085 - (i // 2) * 0.21, s_, 3.4, color=c, ha='center')
    return S


if __name__ == '__main__':
    out = R / 'paper/figs'
    if '--fig2-tikz' in sys.argv:               # Figure 2 only, as native TikZ for the paper (+ the PDF / PNG / draw.io copies)
        S = figure2(); S.render(out / 'fig2_stages')
        S.render_tikz(R / 'paper/overleaf/Figures/fig2_stages_tikz.tex', R / 'paper/overleaf/Figures/fig2', 'Figures/fig2')
    else:
        figure1().render(out / 'fig1_overview')
        figure2().render(out / 'fig2_stages')
