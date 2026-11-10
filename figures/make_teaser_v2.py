"""Fig. 1 (teaser), final data: any field sensor in, the procedural plant out.
Two rows (sorghum, maize; one held-out TEST plant each). Left: the three sensor streams of the view
(photograph, depth frame, point cloud), any one of which is an input. Middle: the CropCode encoder.
Right: the recipe the model writes from the PHOTOGRAPH ALONE (the distilled generalist), and what the
recipe gives: the re-rendered plant (the generated cloud, Chamfer against the true cloud) and the
traits read from it, each beside its true value.

The figure is described once, as a list of primitives in inches (x from the left, y from the bottom),
and drawn by two renderers from that list:
  matplotlib -> paper/figs/fig_teaser.{pdf,png} (+ the Overleaf copy of the PDF)
  draw.io    -> paper/figs/fig_teaser.drawio: every box, label, arrow and the recipe and trait text
                are native, editable draw.io cells; the photographs, depth frames and clouds are PNGs
                embedded as data URIs, so the file is self-contained. Open it in app.diagrams.net or
                the draw.io desktop app; File > Export as > PDF for the paper.
usage: python figures/make_teaser_v2.py [--sorghum Sorghum_10036] [--maize plant_0043] [--cache <pkl>]
CPU, ~3 min for the two model passes (skipped when --cache exists).
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import argparse, base64, html, io, pickle, shutil
from xml.sax.saxutils import escape as xml_escape
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from PIL import Image
import _plantviz as V

ap = argparse.ArgumentParser()
ap.add_argument('--sorghum', default='Sorghum_10036'); ap.add_argument('--maize', default='plant_0043')
ap.add_argument('--split', default='test'); ap.add_argument('--device', default='cpu'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--cache', default=None, help='pickle of the model outputs: read if it exists, else written')
ap.add_argument('--out', default='paper/figs/fig_teaser', help='path without extension')
a = ap.parse_args()
PL = {'sorghum': a.sorghum, 'maize': a.maize}
FIELDS = {'sorghum': None, 'maize': ('dist', 'len', 'wid', 'angle', 'droop')}
TRAITS = {'sorghum': [('height', lambda r: r['height'], 'm'), ('leaves', lambda r: r['n'], ''),
                      ('leaf length', lambda r: np.mean([l['length'] for l in r['leaves']]), 'm'),
                      ('leaf width', lambda r: np.mean([l['width'] for l in r['leaves']]), 'm')],
          'maize': [('height', lambda r: r['height'], 'm'), ('leaves', lambda r: r['n'], ''),
                    ('leaf length', lambda r: np.mean([l['leafLength'] for l in r['leaves']]), 'm'),
                    ('leaf angle', lambda r: np.mean([l['leafAngle'] for l in r['leaves']]), '°')]}
INK, MUTED, NAVY, BOXFILL = '#1d221c', '#5f685c', '#2b3a67', '#eef1f7'
CS = {'rgb': '#c0392b', 'depth': '#b8860b', 'pc': '#2a9d8f', 'text': '#30638e'}
# geometry, inches
W, H = 7.05, 3.15
T = 0.74                                         # tile side
ROW_Y = {'sorghum': 1.72, 'maize': 0.34}
X_IN = [0.30, 0.30 + T + 0.05, 0.30 + 2 * (T + 0.05)]
X_BOX, W_BOX = 2.80, 0.88
X_REC, W_REC = 3.92, 1.40
X_CLOUD = 5.50; X_TRAIT = 6.32; W_TRAIT = 0.72
HDR_Y = 2.72


# ------------------------------------------------------------------ model passes (cached)
def compute():
    D = {}
    for sp in ('sorghum', 'maize'):
        ds = V.dataset(sp, a.split); it = V.item_by_name(ds, f'{PL[sp]}_00', seed=a.seed)
        m, cfg, ep = V.load_model(sp, 'generalist', a.device)
        g = V.generate(m, it, ['rgb'], a.device, seed=a.seed)
        D[sp] = dict(plant=PL[sp], rgb=V.tensor_rgb(it[0]), depth=it[1][0].numpy(), cloud=it[2].numpy(), par=it[3].numpy(),
                     tv=it[4].numpy(), gen_pc=np.asarray(g['pc']), gen_par=g['par'].numpy(), epoch=ep)
    return D


if a.cache and pathlib.Path(a.cache).exists() and all(pickle.load(open(a.cache, 'rb'))[s]['plant'] == PL[s] for s in PL):
    DATA = pickle.load(open(a.cache, 'rb'))
else:
    DATA = compute()
    if a.cache: pickle.dump(DATA, open(a.cache, 'wb'))

# ------------------------------------------------------------------ the scene: one list of primitives
OPS = []   # z: 0 filled boxes, 1 images, 2 frames, 3 text and arrows


def rect(x, y, w, h, stroke, lw=1.0, fill=None, r=0.04):
    OPS.append(dict(kind='rect', z=0 if fill else 2, x=x, y=y, w=w, h=h, stroke=stroke, lw=lw, fill=fill, r=r))


def arrow(p, q, color=INK, lw=0.9):
    OPS.append(dict(kind='arrow', z=3, p=p, q=q, color=color, lw=lw))


def text(x, y, s, size=6.5, color=INK, ha='left', va='center', bold=False, mono=False, italic=False, rot=0, ls=1.25):
    OPS.append(dict(kind='text', z=3, x=x, y=y, s=s, size=size, color=color, ha=ha, va=va, bold=bold, mono=mono,
                    italic=italic, rot=rot, ls=ls))


def image(x, y, w, h, png, name):
    OPS.append(dict(kind='image', z=1, x=x, y=y, w=w, h=h, png=png, name=name))


def png_array(arr, max_side=448):
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    s = max_side / max(im.size)
    if s > 1: im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, 'PNG', optimize=True); return b.getvalue()


def png_cloud(P, frame, side, dpi=400):
    fig = plt.figure(figsize=(side, side), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    V.scatter_cloud(ax, P, c=-np.asarray(P)[:, 1], cmap='viridis', frame=frame, s=0.35, elev=12, azim=-35)
    b = io.BytesIO(); fig.savefig(b, format='png', dpi=dpi, transparent=True); plt.close(fig); return b.getvalue()


def square_crop(dep, pad=12):
    ys, xs = np.where(dep > 0.005)
    c0, c1 = max(0, ys.min() - pad), min(dep.shape[0], ys.max() + pad)
    d0, d1 = max(0, xs.min() - pad), min(dep.shape[1], xs.max() + pad)
    side = max(c1 - c0, d1 - d0); cy, cx = (c0 + c1) // 2, (d0 + d1) // 2
    return (slice(max(0, cy - side // 2), min(dep.shape[0], cy + side // 2)),
            slice(max(0, cx - side // 2), min(dep.shape[1], cx + side // 2)))


# column headers
text(0.30, HDR_Y + 0.22, 'Field sensing: any one of these', 7.5, bold=True)
text(X_BOX + W_BOX / 2, HDR_Y + 0.22, 'One model', 7.5, bold=True, ha='center')
text(X_REC, HDR_Y + 0.22, 'Procedural plant out, and what it gives', 7.5, bold=True)
for lab, x, c in (('photograph', X_IN[0], CS['rgb']), ('depth frame', X_IN[1], CS['depth']), ('point cloud', X_IN[2], CS['pc'])):
    text(x + T / 2, HDR_Y, lab, 6.2, color=c, ha='center', bold=True)
text(X_REC + W_REC / 2, HDR_Y, 'recipe (written by the model)', 6.2, color=CS['text'], ha='center', bold=True)
text(X_CLOUD + T / 2, HDR_Y, 're-rendered plant', 6.2, color=CS['pc'], ha='center', bold=True)
text(X_TRAIT + W_TRAIT / 2, HDR_Y, 'traits', 6.2, color=INK, ha='center', bold=True)

# the model box, spanning both rows
by0, by1 = ROW_Y['maize'], ROW_Y['sorghum'] + T
rect(X_BOX, by0, W_BOX, by1 - by0, NAVY, lw=1.3, fill=BOXFILL, r=0.06)
text(X_BOX + W_BOX / 2, (by0 + by1) / 2 + 0.22, 'CropCode', 8, color=NAVY, ha='center', bold=True)
text(X_BOX + W_BOX / 2, (by0 + by1) / 2 - 0.16, 'one encoder for\nphotographs, depth,\nclouds and recipes;\npretrained with 80 %\n'
     'of tokens hidden,\nthen distilled so\none sensor completes\nthe rest', 5.0, color=NAVY, ha='center')

for sp in ('sorghum', 'maize'):
    d = DATA[sp]; y = ROW_Y[sp]
    fr = V.frame_of(d['cloud'])
    rec_t = V.decode_recipe(sp, d['par'], d['tv']); rec_p = V.decode_recipe(sp, d['gen_par'], d['tv'], predicted=True)
    ch = V.chamfer(d['gen_pc'], d['cloud'])
    text(0.13, y + T / 2, sp, 7, ha='center', bold=True, rot=90)
    # inputs (photograph and depth cropped to the same square around the plant)
    cr = square_crop(d['depth'])
    image(X_IN[0], y, T, T, png_array(d['rgb'][cr]), f'{sp} photograph'); rect(X_IN[0], y, T, T, CS['rgb'], lw=1.6)
    image(X_IN[1], y, T, T, png_array(V.depth_rgb(d['depth'])[cr]), f'{sp} depth'); rect(X_IN[1], y, T, T, CS['depth'])
    image(X_IN[2] - 0.03, y - 0.03, T + 0.06, T + 0.06, png_cloud(d['cloud'], fr, T + 0.06), f'{sp} point cloud')
    rect(X_IN[2], y, T, T, CS['pc'])
    arrow((X_IN[2] + T + 0.02, y + T / 2), (X_BOX - 0.02, y + T / 2), color=INK)
    text(X_IN[0], y - 0.07, 'any one of these is enough; this example uses the photograph (thick frame)', 4.6, color=MUTED, va='top')
    # recipe written by the model
    arrow((X_BOX + W_BOX + 0.02, y + T / 2), (X_REC - 0.02, y + T / 2), color=CS['text'])
    rect(X_REC, y, W_REC, T, CS['text'])
    text(X_REC + 0.04, y + T - 0.04, '\n'.join(V.recipe_lines(sp, rec_p, max_leaves=5, fields=FIELDS[sp])), 4.4,
         color=CS['text'], va='top', mono=True, ls=1.3)
    # re-rendered plant
    arrow((X_REC + W_REC + 0.02, y + T / 2), (X_CLOUD - 0.02, y + T / 2), color=CS['pc'])
    image(X_CLOUD - 0.03, y - 0.03, T + 0.06, T + 0.06, png_cloud(d['gen_pc'], fr, T + 0.06), f'{sp} generated cloud')
    rect(X_CLOUD, y, T, T, CS['pc'])
    text(X_CLOUD + T / 2, y - 0.07, f'Chamfer {ch * 1e3:.2f}e-3 vs the true plant', 4.6, color=MUTED, ha='center', va='top')
    # traits, each beside its true value
    rect(X_TRAIT, y, W_TRAIT, T, INK, lw=0.6)
    rows = []
    for name, f, unit in TRAITS[sp]:
        vp, vt = f(rec_p), f(rec_t)
        fmt = (lambda v: f'{v:.0f}') if unit in ('', '°') else (lambda v: f'{v:.2f}' if v >= 0.1 else f'{v:.3f}')
        rows.append(f'{name}\n  {fmt(vp)}{unit}  (true {fmt(vt)}{unit})')
    text(X_TRAIT + 0.04, y + T - 0.04, '\n'.join(rows), 4.4, va='top', ls=1.2)
    print(sp, d['plant'], 'generalist ep', d['epoch'], f'chamfer {ch * 1e3:.2f}e-3', 'pred', rec_p['n'], f"{rec_p['height']:.2f}",
          'true', rec_t['n'], f"{rec_t['height']:.2f}", flush=True)
text(0.30, 0.12, f'Held-out {a.split} plants, view 0. The recipe and the cloud are generated from the photograph alone by the '
     'distilled generalist; the true cloud and the true traits are shown for comparison only.', 4.6, color=MUTED, italic=True)


# ------------------------------------------------------------------ renderer 1: matplotlib
def render_mpl(stem):
    fig = plt.figure(figsize=(W, H))

    def overlay():
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_axis_off(); ax.patch.set_alpha(0)
        return ax

    def draw_rect(ax, o):
        ax.add_patch(mpatches.FancyBboxPatch((o['x'], o['y']), o['w'], o['h'], boxstyle=f"round,pad=0,rounding_size={o['r']}",
                                             fill=bool(o['fill']), facecolor=o['fill'] or 'none', edgecolor=o['stroke'], lw=o['lw']))
    bg = overlay()
    for o in OPS:
        if o['kind'] == 'rect' and o['fill']: draw_rect(bg, o)
    for o in OPS:
        if o['kind'] == 'image':
            ax = fig.add_axes([o['x'] / W, o['y'] / H, o['w'] / W, o['h'] / H])
            ax.imshow(np.asarray(Image.open(io.BytesIO(o['png']))), aspect='equal'); ax.set_axis_off(); ax.patch.set_alpha(0)
    top = overlay()   # frames, arrows and text above the images
    for o in OPS:
        if o['kind'] == 'rect' and not o['fill']: draw_rect(top, o)
        elif o['kind'] == 'arrow':
            top.add_patch(FancyArrowPatch(o['p'], o['q'], arrowstyle='-|>', mutation_scale=7, color=o['color'], lw=o['lw'], shrinkA=0, shrinkB=0))
        elif o['kind'] == 'text':
            top.text(o['x'], o['y'], o['s'], fontsize=o['size'], color=o['color'], ha=o['ha'], va=o['va'], rotation=o['rot'],
                     fontweight='bold' if o['bold'] else 'normal', family='monospace' if o['mono'] else None,
                     style='italic' if o['italic'] else 'normal', linespacing=o['ls'])
    for ext in ('pdf', 'png'): fig.savefig(f'{stem}.{ext}', dpi=220)
    plt.close(fig)


# ------------------------------------------------------------------ renderer 2: draw.io
PX = 200                                   # canvas px per inch; font sizes and strokes scale by PX / 72


def render_drawio(path):
    cells, n = [], [1]
    pt = lambda v: v * PX / 72

    def vertex(x, y, w, h, style, value=''):          # inches, y from the bottom
        n[0] += 1
        cells.append(f'<mxCell id="c{n[0]}" value="{xml_escape(value, {chr(34): "&quot;"})}" style="{style}" vertex="1" parent="1">'
                     f'<mxGeometry x="{x * PX:.1f}" y="{(H - y - h) * PX:.1f}" width="{w * PX:.1f}" height="{h * PX:.1f}" as="geometry"/></mxCell>')

    def edge(p, q, style):
        n[0] += 1
        cells.append(f'<mxCell id="c{n[0]}" value="" style="{style}" edge="1" parent="1"><mxGeometry relative="1" as="geometry">'
                     f'<mxPoint x="{p[0] * PX:.1f}" y="{(H - p[1]) * PX:.1f}" as="sourcePoint"/>'
                     f'<mxPoint x="{q[0] * PX:.1f}" y="{(H - q[1]) * PX:.1f}" as="targetPoint"/></mxGeometry></mxCell>')
    for o in sorted(OPS, key=lambda o: o['z']):        # stable: keeps the build order within a layer
        k = o['kind']
        if k == 'rect':
            vertex(o['x'], o['y'], o['w'], o['h'],
                   f"rounded=1;absoluteArcSize=1;arcSize={2 * o['r'] * PX:.0f};whiteSpace=wrap;html=1;fillColor={o['fill'] or 'none'};"
                   f"strokeColor={o['stroke']};strokeWidth={pt(o['lw']):.1f};")
        elif k == 'image':
            b64 = base64.b64encode(o['png']).decode()
            vertex(o['x'], o['y'], o['w'], o['h'], f'shape=image;imageAspect=1;aspect=fixed;html=1;image=data:image/png,{b64};', '')
        elif k == 'arrow':
            edge(o['p'], o['q'], f"endArrow=block;endFill=1;endSize={pt(3.2):.0f};html=1;rounded=0;edgeStyle=none;"
                                 f"strokeColor={o['color']};strokeWidth={pt(o['lw']):.1f};")
        elif k == 'text':
            lines = o['s'].split('\n')
            w = max(len(l) for l in lines) * o['size'] * (0.60 if o['mono'] else 0.58) / 72 + 0.04
            h = len(lines) * o['size'] * o['ls'] / 72 + 0.02
            if o['rot']: w, h = h, w
            x = o['x'] - {'left': 0, 'center': w / 2, 'right': w}[o['ha']]
            y = o['y'] - {'bottom': 0, 'center': h / 2, 'top': h}[o['va']]
            label = '<br>'.join(html.escape(l, quote=False).replace(' ', '&nbsp;') for l in lines)
            label = f'<div style="line-height:{o["ls"]}">{label}</div>'
            style = (f"text;html=1;strokeColor=none;fillColor=none;whiteSpace=nowrap;spacing=0;align={o['ha']};"
                     f"verticalAlign={ {'bottom': 'bottom', 'center': 'middle', 'top': 'top'}[o['va']] };"
                     f"fontFamily={'Courier New' if o['mono'] else 'Helvetica'};fontSize={pt(o['size']):.1f};fontColor={o['color']};"
                     f"fontStyle={(1 if o['bold'] else 0) + (2 if o['italic'] else 0)};" + ('horizontal=0;' if o['rot'] else ''))
            vertex(x, y, w, h, style, label)
    xml = (f'<mxfile host="make_teaser_v2.py"><diagram name="Fig. 1 teaser" id="teaser">'
           f'<mxGraphModel dx="0" dy="0" grid="0" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" '
           f'pageScale="1" pageWidth="{W * PX:.0f}" pageHeight="{H * PX:.0f}" math="0" shadow="0" background="#ffffff">'
           f'<root><mxCell id="0"/><mxCell id="1" parent="0"/>\n' + '\n'.join(cells) + '\n</root></mxGraphModel></diagram></mxfile>')
    pathlib.Path(path).write_text(xml, encoding='utf-8')
    return len(cells)


stem = pathlib.Path(a.out)
render_mpl(stem)
n_cells = render_drawio(stem.with_suffix('.drawio'))
shutil.copy(stem.with_suffix('.pdf'), 'paper/overleaf/Figures/fig_teaser.pdf')
print('wrote', stem.with_suffix('.pdf'), stem.with_suffix('.png'), stem.with_suffix('.drawio'),
      f'({n_cells} draw.io cells, {stem.with_suffix(".drawio").stat().st_size / 1e6:.2f} MB)')
