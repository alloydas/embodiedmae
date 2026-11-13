"""Fig. 3 (the dataset), second design. User, 2026-11-13: "look into the 1st and second figure of both of the paper
[Slice-100K, arXiv:2407.04180; FlowBench, arXiv:2409.18032] and like that try to generate Figure 3 of our paper",
"in UGG style".

What it takes from them:
  FlowBench Figs. 1-2  a gallery of many samples in rows labelled by group (here: species), and grey leader lines
                       from one sample of the gallery to its close-up;
  Slice-100K Fig. 2    one sample laid out as its data formats side by side, each named underneath;
  UGG (ugg_style.py)   one pastel colour per stream (image blue, depth yellow, point cloud pink, parameters purple)
                       in rounded boxes; the parameters drawn as the stem-and-leaf schematic of the any-to-all figure.
Panels: (a) the plants (6 per species, val split, side view, ordered by height); (b) one view of the highlighted
plant as its four streams; (c) five of the ten cameras of that plant; (d) one crop-row scene per species (the photograph with
neighbour pixels red and hidden target pixels orange, and the scene cloud with the target's points in green).
Same plants as figures/make_dataset_fig.py (Sorghum_10001 / plant_0003; rows Sorghum_10007 / plant_0004); the font is
matplotlib's default, as in that figure. Output: paper/figs/ugg_preview/fig_dataset_v2.{pdf,png} (matplotlib preview) and the native TikZ version
paper/overleaf/Figures/fig_dataset_tikz.tex + its tiles in Figures/fig3/ (user: "make it in tikz"). CPU, ~2 min:
  sbatch --wait --partition=nova --account=mech-ai --qos=normal --cpus-per-task=4 --mem=24G --time=00:30:00 \
         --output=logs/%x_%j.out --wrap "python figures/make_dataset_fig_v2.py"
"""
import sys, pathlib
R = pathlib.Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R), str(R / 'figures')]
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.spatial import cKDTree
import _plantviz as V
import _scene
import make_any2all_fig as A
import ugg_style as U

OUT = R / 'paper/figs/ugg_preview'
PL = {'sorghum': 'Sorghum_10001', 'maize': 'plant_0003'}
RP = {'sorghum': 'Sorghum_10007', 'maize': 'plant_0004'}
N_GAL, SEED, GCOLS = 6, 3, 3                          # user: fewer samples in (a) and (c)
FS, FL = 6.2, 9.0                                    # make_dataset_fig.py's sizes: labels 6.2 pt, panel letters 9 pt
W, H = 6.875, 3.86
A.C_OUT = U.STREAM_INK['text']
elev_tab = pd.read_csv(R / 'data_split/maize_view_elevation.csv.gz')

TIKZ_TEX = R / 'paper/overleaf/Figures/fig_dataset_tikz.tex'
TIKZ_IMG = R / 'paper/overleaf/Figures/fig3'                 # the tiles; included as Figures/fig3/<name>.png
TIKZ_IMG.mkdir(parents=True, exist_ok=True)
TZ, COLS = [], {}

fig = plt.figure(figsize=(W, H))
ov = fig.add_axes([0, 0, 1, 1]); ov.set_xlim(0, W); ov.set_ylim(0, H); ov.set_axis_off(); ov.patch.set_alpha(0)


def col(c):
    h = matplotlib.colors.to_hex(c).lstrip('#').upper(); COLS[h] = f'c{h}'; return f'c{h}'


def tex(s):
    return s.replace('%', '\\%').replace('°', '$^{\\circ}$').replace('−', '$-$')   # pdfLaTeX-safe


def box(x, y, w, h, stream=None, r=0.05, lw=1.0, ec=None, fc=None, dashed=False):
    """a rounded box: matplotlib preview + TikZ (stream pastel by default)"""
    fc_, ec_ = (U.STREAM_FILL[stream], U.STREAM[stream]) if stream else ('#f4f4f5', '#9a9a9a')
    fc_ = fc if fc is not None else fc_; ec_ = ec or ec_
    ov.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc_ if fc_ != 'none' else 'none',
                                ec=ec_, lw=lw, ls=(0, (3.2, 2.0)) if dashed else '-', zorder=6 if fc_ == 'none' else 0))
    fill = '' if fc_ == 'none' else f', fill={col(fc_)}'
    TZ.append(f'\\draw[draw={col(ec_)}, line width={lw:.2f}pt, rounded corners={r * 72:.2f}pt{fill}'
              f'{", dash pattern=on 3.2pt off 2pt" if dashed else ""}] ({x:.4f},{y:.4f}) rectangle ({x + w:.4f},{y + h:.4f});')


def line(p, q, color='#9a9a9a', lw=0.6):
    ov.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, zorder=4)
    TZ.append(f'\\draw[draw={col(color)}, line width={lw:.2f}pt] ({p[0]:.4f},{p[1]:.4f}) -- ({q[0]:.4f},{q[1]:.4f});')


def txt(x, y, s, size, ha='left', va='baseline', color='#111111', bold=False, rot=0):
    ov.text(x, y, s, fontsize=size, ha=ha, va=va, color=color, fontweight='bold' if bold else 'normal', rotation=rot)
    a = {'top': 'north', 'center': '', 'bottom': 'south', 'baseline': 'base'}[va]
    h = {'left': 'west', 'center': '', 'right': 'east'}[ha]
    anchor = (f'{a} {h}'.strip() or 'center') if not rot else {'center': 'center', 'bottom': 'south', 'top': 'north'}.get(va, 'center')
    TZ.append(f'\\node[anchor={anchor}, inner sep=0, text={col(color)}, rotate={rot}, font={{\\fontsize{{{size:.1f}}}{{{size * 1.2:.1f}}}\\selectfont\\sffamily'
              f'{chr(92) + "bfseries" if bold else ""}}}] at ({x:.4f},{y:.4f}) {{{tex(s)}}};')


def img(name, x, y, w, h, arr=None, draw=None, proj=None, dpi=600):
    """a picture tile: an array, or draw(ax) on its own transparent figure (clouds, schematic); saved as a PNG for TikZ"""
    from PIL import Image
    path = TIKZ_IMG / f'{name}.png'
    if arr is not None:
        Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(path)
    else:
        f2 = plt.figure(figsize=(w, h), dpi=dpi); a2 = f2.add_axes([0, 0, 1, 1], **({'projection': proj} if proj else {}))
        a2.set_axis_off(); a2.patch.set_alpha(0); draw(a2)
        f2.savefig(path, dpi=dpi, transparent=True); plt.close(f2)
    ax = fig.add_axes([x / W, y / H, w / W, h / H]); ax.set_axis_off(); ax.patch.set_alpha(0)
    ax.imshow(np.asarray(Image.open(path)), extent=(0, 1, 0, 1)); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    TZ.append(f'\\node[anchor=south west, inner sep=0] at ({x:.4f},{y:.4f}) {{\\includegraphics[width={w:.4f}in, height={h:.4f}in]{{Figures/fig3/{name}.png}}}};')


def crop_alpha(path, pad=0.04, out=256):
    """the plant's own bounding square from its alpha, on white, bottom-aligned (the plant stands on the tile floor)"""
    from PIL import Image
    im = Image.open(path).convert('RGBA'); a = np.asarray(im)[..., 3] > 0
    ys, xs = np.nonzero(a); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    s = int(max(y1 - y0, x1 - x0) * (1 + 2 * pad))
    canvas = Image.new('RGBA', (s, s), (255, 255, 255, 0))
    canvas.paste(im.crop((x0, y0, x1, y1)), ((s - (x1 - x0)) // 2, s - (y1 - y0) - int(pad * s)))
    rgba = np.asarray(canvas.resize((out, out), Image.LANCZOS)).astype(np.float32) / 255
    return rgba[..., :3] * rgba[..., 3:4] + (1 - rgba[..., 3:4])


def side_view(sp, plant):
    if sp == 'sorghum':
        return 4                                      # +6 deg: the horizon end of the fixed ladder
    t = elev_tab[(elev_tab.split == 'val') & (elev_tab.plant == plant)]
    return int(t.iloc[np.argmin(np.abs(t.elev_deg.values))]['view'])


def gallery_plants(sp):
    rng = np.random.default_rng(SEED)
    if sp == 'sorghum':
        t = pd.read_csv(V.ROOTS[sp] / 'assignment.csv').merge(pd.read_csv(V.ROOTS[sp] / 'features.csv'), on='plant')
        t = t[t.split == 'val']; t['name'] = 'Sorghum_' + t.plant.astype(str); t['h'] = t.stem_length
    else:
        t = pd.read_csv(V.ROOTS[sp] / 'plant_scores.csv'); t = t[t.split == 'val']; t['name'] = t.plant; t['h'] = t.stem_internodeSum
    t = t[t.name != PL[sp]]
    pick = list(t.sample(N_GAL - 1, random_state=int(rng.integers(1e6))).name)
    hh = dict(zip(t.name, t.h))
    return sorted(pick, key=lambda n: hh[n]) + [PL[sp]]      # the highlighted plant last: bottom right, next to (b)


def letter(x, y, k, title):
    txt(x, y, k, FL, bold=True); txt(x + 0.16, y, title, FS)


# ------------------------------------------------------------------ layout (inches, y up)
GT, GG = 0.44, 0.035                                  # gallery tile, gap
GX0 = 0.30; GW = GCOLS * GT + (GCOLS - 1) * GG
ST = 0.84
SX0 = GX0 + GW + 0.75
SG = (W - 0.07 - SX0 - 0.07 - 4 * ST) / 3                # stream gap: (b) spans to the right margin
ROW_Y = {'sorghum': 2.47, 'maize': 1.37}             # bottom of each species band (top tier)
BAND_H = 2 * GT + GG
letter(0.02, 3.72, 'a', f'The plants: 15,000 per species ({N_GAL} shown)')
letter(SX0 - 0.06, 3.72, 'b', 'One view of the boxed plant as its four streams')

for sp in ('sorghum', 'maize'):
    y0 = ROW_Y[sp]
    # ---- (a) gallery: one image-blue box per species, the plants in two rows
    box(GX0 - 0.06, y0 - 0.06, GW + 0.12, BAND_H + 0.12, 'rgb', r=0.06)
    txt(GX0 - 0.15, y0 + BAND_H / 2, sp, FS + 0.8, ha='center', va='center', bold=True, rot=90)
    names = gallery_plants(sp); hl = None
    for i, n in enumerate(names):
        r_, c_ = divmod(i, GCOLS)
        x = GX0 + c_ * (GT + GG); y = y0 + (1 - r_) * (GT + GG)
        img(f'{sp}_gallery_{i}', x, y, GT, GT, arr=crop_alpha(V.folder(sp, 'val', n, side_view(sp, n)) / V.RGB_FILE[sp]))
        if n == PL[sp]:
            hl = (x, y)
    x, y = hl                                          # the highlighted plant, and FlowBench's leader lines to (b)
    box(x - 0.015, y - 0.015, GT + 0.03, GT + 0.03, r=0.03, lw=1.0, ec='#333333', fc='none')
    yb = y0 + BAND_H / 2 - ST / 2
    bx0, by0, by1 = SX0 - 0.07, yb - 0.07, yb + ST + 0.07
    for (px, py), (qx, qy) in (((x + GT + 0.015, y + GT + 0.015), (bx0, by1)), ((x + GT + 0.015, y - 0.015), (bx0, by0))):
        line((px, py), (qx, qy))

    # ---- (b) the highlighted plant's view 0 as its four streams (Slice-100K Fig. 2), names underneath
    f = V.folder(sp, 'val', PL[sp], 0)
    ds = V.dataset(sp); it = V.item_by_name(ds, f'{PL[sp]}_00')
    dep = V.decode_depth_png(f / 'depth.png'); fg = dep > 0
    box(bx0, by0, 4 * ST + 3 * SG + 0.14, by1 - by0, None, r=0.06)          # one neutral frame per view
    for j, (st, nm) in enumerate((('rgb', 'Image'), ('depth', 'Depth image'), ('pc', 'Point cloud'), ('text', 'Procedural parameters'))):
        x = SX0 + j * (ST + SG)
        box(x - 0.03, yb - 0.03, ST + 0.06, ST + 0.06, st, r=0.05)
        nm_ = f'{sp}_{st}'
        if st == 'rgb':
            img(nm_, x, yb, ST, ST, arr=A.crop_square(V.rgba_on_white(f / V.RGB_FILE[sp]), fg))
        elif st == 'depth':
            img(nm_, x, yb, ST, ST, arr=A.crop_square(V.depth_rgb(dep), fg))
        elif st == 'pc':
            P = it[2].numpy(); img(nm_, x, yb, ST, ST, draw=lambda a, P=P: A._cloud(a, P, P, _scene.cloud_frame(P, **A.VIEW), under=False, s=0.25))
        else:
            rec = V.decode_recipe(sp, it[3].numpy(), it[4].numpy())
            img(nm_, x, yb, ST, ST, draw=lambda a, rec=rec: A._glyph(a, sp, rec, color=U.STREAM_INK['text']))
        if sp == 'sorghum':                            # names once, above the columns (Slice-100K names them once)
            txt(x + ST / 2, by1 + 0.075, nm, FS, ha='center', va='baseline', color=U.STREAM_INK[st], bold=True)

# ------------------------------------------------------------------ bottom tier: (c) cameras, (d) rows
letter(0.02, 1.13, 'c', 'Ten cameras per plant (five shown), by elevation')
CT, CG = 0.34, 0.035
for r_, sp in enumerate(('sorghum', 'maize')):
    if sp == 'sorghum':
        order, el = list(range(10)), V.SORGHUM_ELEV
    else:
        t = elev_tab[(elev_tab.split == 'val') & (elev_tab.plant == PL[sp])].sort_values('elev_deg', ascending=False)
        order, el = t['view'].tolist(), t.elev_deg.values
    order, el = list(order)[::2], list(el)[::2]          # every other camera, top to bottom
    y = 0.60 - r_ * 0.53
    box(GX0 - 0.04, y - 0.04, 5 * CT + 4 * CG + 0.08, CT + 0.08, 'rgb', r=0.04)
    txt(GX0 - 0.15, y + CT / 2, sp, FS - 0.4, ha='center', va='center', bold=True, rot=90)
    for k, (v, e) in enumerate(zip(order, el)):
        x = GX0 + k * (CT + CG)
        img(f'{sp}_camera_{k}', x, y, CT, CT, arr=crop_alpha(V.folder(sp, 'val', PL[sp], v) / V.RGB_FILE[sp], pad=0.08))
        txt(x + CT / 2, y + CT + 0.05, f'{e:+.0f}°'.replace('-', '−'), FS - 0.8, ha='center', va='bottom', color='#444444')

XD = 3.02; RT = 0.70; RG = 0.08
letter(XD - 0.18, 1.13, 'd', 'Crop rows: the target plant between neighbors')
txt(XD - 0.02, 1.025, 'red: neighbor pixels;  orange: target pixels a neighbor hides (% hidden);  green: target points', FS - 0.8, color='#555555')
for j, sp in enumerate(('sorghum', 'maize')):
    d = V.row_shot_dir(sp, 'test', RP[sp], 0)
    rowimg, st = V.row_overlay(sp, d)
    scene = V.read_cloud(d / 'pointcloud_cam.ply'); tgt = V.read_cloud(d / 'gt_pointcloud_cam.ply')
    is_t = cKDTree(tgt).query(scene, workers=1)[0] < 3e-3
    x = XD + j * (2 * RT + RG + 0.30)
    txt(x + RT + RG / 2, 0.935, sp, FS, ha='center', va='baseline', bold=True)
    box(x - 0.03, 0.13, RT + 0.06, RT + 0.06, 'rgb', r=0.05)
    img(f'{sp}_row_image', x, 0.16, RT, RT, arr=rowimg)
    txt(x + RT / 2, 0.10, f'{st["occ"] * 100:.0f} % hidden', FS - 0.6, ha='center', va='top')
    box(x + RT + RG - 0.03, 0.13, RT + 0.06, RT + 0.06, 'pc', r=0.05)
    cc = np.where(is_t[:, None], np.array([[0.18, 0.42, 0.16]]), np.array([[0.70, 0.70, 0.70]]))
    img(f'{sp}_row_cloud', x + RT + RG, 0.16, RT, RT, draw=lambda a, scene=scene, cc=cc: V.scatter_cloud(a, scene, c=cc, s=0.3, elev=10, azim=-30), proj='3d')
    txt(x + RT + RG + RT / 2, 0.10, f'{100 * (~is_t).mean():.0f} % neighbors', FS - 0.6, ha='center', va='top')

OUT.mkdir(parents=True, exist_ok=True)
for ext in ('pdf', 'png'):
    fig.savefig(OUT / f'fig_dataset_v2.{ext}', dpi=300)
print('wrote', OUT / 'fig_dataset_v2.pdf')
hdr = ['% Fig. 3 (the dataset) as native TikZ, written by figures/make_dataset_fig_v2.py (do not edit by hand: re-run it).',
       '% Tiles in Figures/fig3/; boxes, leader lines and text are TikZ, text in \\sffamily (Helvetica under cvpr/times).',
       f'% Natural size {W} x {H} in; include with \\resizebox{{\\textwidth}}{{!}}{{\\input{{Figures/fig_dataset_tikz}}}}.']
defs = [f'\\definecolor{{{n}}}{{HTML}}{{{h}}}%' for h, n in sorted(COLS.items())]   # '%': no stray spaces inside \\resizebox
TIKZ_TEX.write_text('\n'.join(hdr + defs + ['\\begin{tikzpicture}[x=1in, y=1in]', f'\\useasboundingbox (0,0) rectangle ({W},{H});'] + TZ + ['\\end{tikzpicture}', '']))
print('wrote', TIKZ_TEX, len(TZ), 'TikZ commands,', len(list(TIKZ_IMG.glob('*.png'))), 'tiles')
