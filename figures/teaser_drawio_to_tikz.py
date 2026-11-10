"""Convert the draw.io teaser (paper/figs/fig_teaser.drawio) to native TikZ.

Reads every cell of the drawio file: the 8 embedded PNG tiles are written to
paper/overleaf/Figures/teaser/, and all boxes, labels, recipe/trait text and
arrows become TikZ primitives in paper/overleaf/Figures/fig_teaser_tikz.tex
(coordinates in drawio px, 200 px/in, y down). main.tex includes it with
\\resizebox{\\textwidth}{!}{\\input{Figures/fig_teaser_tikz}} and needs
\\usetikzlibrary{arrows.meta}. Re-run after make_teaser_v2.py regenerates the
drawio file: python figures/teaser_drawio_to_tikz.py
"""
import base64, html, pathlib, re
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parent.parent
DRAWIO = ROOT / 'paper/figs/fig_teaser.drawio'
TILES = ROOT / 'paper/overleaf/Figures/teaser'
OUT = ROOT / 'paper/overleaf/Figures/fig_teaser_tikz.tex'
IMG_NAME = {'c3': 's_rgb', 'c4': 's_depth', 'c5': 's_pc', 'c6': 's_gen',
            'c7': 'm_rgb', 'c8': 'm_depth', 'c9': 'm_pc', 'c10': 'm_gen'}
# Column-header terminology set in the 2026-11-10 group meeting (applied to
# whole-cell matches only, so body text is untouched).
LABEL_MAP = {'photograph': 'Image', 'depth frame': 'Depth image',
             'point cloud': 'Point cloud',
             'recipe of the center plant': 'Procedural parameters',
             're-rendered plant': 'Generated point cloud',
             'traits': 'Procedural\nparameter / trait'}
PT = 72.0 / 200.0            # px -> pt
IN = 1.0 / 200.0             # px -> in


def style_dict(s):
    d = {}
    for part in (s or '').split(';'):
        if '=' in part:
            k, v = part.split('=', 1)
            d[k] = v
        elif part:
            d[part] = '1'
    return d


def tex_escape(s):
    s = s.replace('%', r'\%').replace('&', r'\&').replace('#', r'\#').replace('_', r'\_')
    s = s.replace('\u00b2', r'\textsuperscript{2}').replace('\u00b0', r'$^\circ$')
    s = s.replace('\u2026', r'\dots{}')
    return s


def node_text(value, mono):
    v = re.sub(r'<div[^>]*>', '', value or '').replace('</div>', '')
    lines = [html.unescape(x).replace('\u00a0', ' ') for x in v.split('<br>')]
    if len(lines) == 1 and lines[0] in LABEL_MAP:
        lines = LABEL_MAP[lines[0]].split('\n')
    out = []
    for ln in lines:
        ln = tex_escape(ln)
        if mono:
            ln = ln.replace(' ', '~')
        else:
            ln = re.sub(r'  +', lambda m: '~' * len(m.group(0)), ln)
            ln = re.sub(r'^ ', '~', ln)
        out.append(ln)
    return r'\\'.join(out)


tree = ET.parse(DRAWIO)
TILES.mkdir(exist_ok=True)
body = []
for c in tree.getroot().iter('mxCell'):
    st = style_dict(c.get('style'))
    g = c.find('mxGeometry')
    if c.get('edge') == '1':
        pts = {p.get('as'): (float(p.get('x')), float(p.get('y'))) for p in g.iter('mxPoint')}
        (sx, sy), (tx, ty) = pts['sourcePoint'], pts['targetPoint']
        lw = float(st.get('strokeWidth', 1)) * PT
        size = float(st.get('endSize', 9)) * PT
        body.append(f"\\draw[color={{rgbhex{st['strokeColor'][1:]}}}, line width={lw:.2f}pt, "
                    f"-{{Triangle[length={size:.1f}pt,width={size * 0.8:.1f}pt]}}] ({sx},{sy}) -- ({tx},{ty});")
        continue
    if g is None or c.get('id') in ('0', '1'):
        continue
    x, y = float(g.get('x', 0)), float(g.get('y', 0))
    w, h = float(g.get('width', 0)), float(g.get('height', 0))
    if 'image' in st and st['image'].startswith('data:image/png,'):
        name = IMG_NAME[c.get('id')]
        (TILES / f'{name}.png').write_bytes(base64.b64decode(st['image'][len('data:image/png,'):]))
        body.append(f"\\node[anchor=north west, inner sep=0] at ({x},{y}) "
                    f"{{\\includegraphics[width={w * IN:.4f}in, height={h * IN:.4f}in]{{Figures/teaser/{name}.png}}}};")
    elif st.get('rounded') == '1':
        lw = float(st.get('strokeWidth', 1)) * PT
        rad = float(st.get('arcSize', 16)) * PT
        fill = st.get('fillColor', 'none')
        opts = f"draw={{rgbhex{st['strokeColor'][1:]}}}, line width={lw:.2f}pt, rounded corners={rad:.1f}pt"
        if fill not in ('none', None):
            opts += f", fill={{rgbhex{fill[1:]}}}"
        body.append(f"\\draw[{opts}] ({x},{y}) rectangle ({x + w},{y + h});")
    elif (c.get('style') or '').startswith('text'):
        mono = 'Courier' in st.get('fontFamily', '')
        fs = float(st.get('fontSize', 12)) * PT
        ls_mult = 1.3 if mono else (1.2 if c.get('id') in ('c41', 'c49') else 1.25)
        colour = f"rgbhex{st.get('fontColor', '#000000')[1:]}"
        halign, valign = st.get('align', 'center'), st.get('verticalAlign', 'middle')
        rotated = st.get('horizontal') == '0'
        if rotated:
            anchor, ax, ay, align = 'center', x + w / 2, y + h / 2, 'center'
        else:
            av = {'top': 'north', 'middle': '', 'bottom': 'south'}[valign]
            ah = {'left': 'west', 'center': '', 'right': 'east'}[halign]
            anchor = (av + ' ' + ah).strip() or 'center'
            ax = {'left': x, 'center': x + w / 2, 'right': x + w}[halign]
            ay = {'top': y, 'middle': y + h / 2, 'bottom': y + h}[valign]
            align = halign
        font = f"\\fontsize{{{fs:.1f}}}{{{fs * ls_mult:.1f}}}\\selectfont"
        font += r'\ttfamily' if mono else r'\sffamily'
        if st.get('fontStyle') == '1':
            font += r'\bfseries'
        elif st.get('fontStyle') == '2':
            font += r'\itshape'
        opts = f"anchor={anchor}, inner sep=0, align={align}, text={{{colour}}}, font={{{font}}}"
        if rotated:
            opts += ', rotate=90'
        body.append(f"\\node[{opts}] at ({ax:.1f},{ay:.1f}) {{{node_text(c.get('value'), mono)}}};")

# colour definitions for every hex used
hexes = sorted(set(re.findall(r'rgbhex([0-9a-fA-F]{6})', '\n'.join(body))))
defs = [f"\\definecolor{{rgbhex{hx}}}{{HTML}}{{{hx.upper()}}}" for hx in hexes]

OUT.write_text(
    "% Generated by figures/teaser_drawio_to_tikz.py from paper/figs/fig_teaser.drawio -- do not edit by hand.\n"
    "% Include with \\resizebox{\\textwidth}{!}{\\input{Figures/fig_teaser_tikz}} (needs \\usetikzlibrary{arrows.meta}).\n"
    + '\n'.join(defs) + '\n'
    + "\\begin{tikzpicture}[x=0.005in, y=-0.005in]\n"
    + "\\useasboundingbox (0,0) rectangle (1410,630);\n"
    + '\n'.join(body) + '\n'
    + "\\end{tikzpicture}\n")
print('wrote', OUT, 'and', len(IMG_NAME), 'tiles in', TILES)
