"""A figure described once, drawn twice: matplotlib (PDF / PNG) and draw.io (editable).

A Scene is a list of primitives in inches (x from the left, y from the bottom): rectangles (optionally rounded,
filled, dashed), polygons, ellipses, text (any font size, bold / italic / mono, rotation, optional halo), arrows
(straight or curved), paths (polylines with rounded bends and an optional head), embedded PNG images, and plots
(a matplotlib drawing function: vector in the PDF, an embedded PNG in draw.io). render_mpl() draws it with
matplotlib; render_drawio() writes every box, label and arrow as a native draw.io cell and every image as an
embedded PNG (data URI), so the .drawio file is self-contained and can be opened and edited in app.diagrams.net
or the desktop app. Generalised from figures/make_teaser_v2.py.

Type is set in Liberation Sans / Liberation Mono (figures/fonts/, SIL OFL), metric-compatible with the Helvetica /
Courier New that draw.io uses, so both renderings lay out alike; glyphs they lack fall back to DejaVu.
lint() reports text below a minimum size and text or boxes that leave the canvas.
"""
import base64, html, io, pathlib
from xml.sax.saxutils import escape as xml_escape
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib import font_manager
from matplotlib.path import Path as MPath
from matplotlib.patches import FancyArrowPatch
from PIL import Image

PX = 200                                   # draw.io canvas px per inch; font sizes and strokes scale by PX / 72
PLANT_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list('plant', ['#1f4d22', '#2f7d32', '#5aa83e', '#a8d26a'])
for _f in sorted((pathlib.Path(__file__).resolve().parent / 'fonts').glob('*.ttf')):
    font_manager.fontManager.addfont(str(_f))
SANS, MONO = ['Liberation Sans', 'DejaVu Sans'], ['Liberation Mono', 'DejaVu Sans Mono']
matplotlib.rcParams.update({'font.family': SANS, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                            'mathtext.fontset': 'custom', 'mathtext.rm': 'Liberation Sans', 'mathtext.it': 'Liberation Sans:italic',
                            'mathtext.bf': 'Liberation Sans:bold', 'axes.unicode_minus': False})
DASH = (0, (3, 2))


def text_size(s, size, bold=False, mono=False, italic=False, ls=1.25):
    """(width, height) in inches of a (multi-line) string at `size` pt, from the font's own metrics."""
    from matplotlib.textpath import TextPath
    fp = font_manager.FontProperties(family=MONO if mono else SANS, weight='bold' if bold else 'normal', style='italic' if italic else 'normal')
    lines = s.split('\n'); w = 0.0
    for l in lines:
        if l.strip():
            w = max(w, TextPath((0, 0), l, size=size, prop=fp).get_extents().x1 / 72)
    return w, (len(lines) - 1) * size * ls / 72 + size * 0.95 / 72


def inset(fig, box, rel=(0, 0, 1, 1), **kw):
    """Axes inside `box` (figure fractions l, b, w, h) at box-relative fractions `rel`: for Scene.plot drawing functions."""
    return fig.add_axes([box[0] + rel[0] * box[2], box[1] + rel[1] * box[3], rel[2] * box[2], rel[3] * box[3]], **kw)


class Scene:
    def __init__(self, W, H, name='figure'):
        self.W, self.H, self.name, self.ops = W, H, name, []

    # ---- primitives (z: 0 filled shapes, 1 images and plots, 2 frames, 3 text and arrows; ties keep build order)
    def rect(self, x, y, w, h, stroke, lw=1.0, fill=None, r=0.04, dash=False, z=None):
        self.ops.append(dict(kind='rect', z=z if z is not None else (0 if fill else 2), x=x, y=y, w=w, h=h, stroke=stroke,
                             lw=lw, fill=fill, r=r, dash=dash))

    def poly(self, pts, stroke, lw=1.0, fill=None, dash=False, z=None):
        """Closed polygon through pts [(x, y), ...]: trapezoids for encoders / decoders, wedges, chevrons."""
        self.ops.append(dict(kind='poly', z=z if z is not None else (0 if fill else 2), pts=[tuple(map(float, p)) for p in pts],
                             stroke=stroke, lw=lw, fill=fill, dash=dash))

    def ellipse(self, cx, cy, rx, ry=None, stroke='#1d221c', lw=1.0, fill=None, z=None):
        self.ops.append(dict(kind='ellipse', z=z if z is not None else (0 if fill else 2), cx=cx, cy=cy, rx=rx, ry=rx if ry is None else ry,
                             stroke=stroke, lw=lw, fill=fill))

    def arrow(self, p, q, color='#1d221c', lw=0.9, dash=False, curve=0.0, head=True, ms=7):
        """curve: signed bend as a fraction of the arrow length (matplotlib arc3 rad); 0 = straight."""
        self.ops.append(dict(kind='arrow', z=3, p=p, q=q, color=color, lw=lw, dash=dash, curve=curve, head=head, ms=ms))

    def path(self, pts, color='#1d221c', lw=0.9, dash=False, head=True, r=0.05, ms=7):
        """Polyline through pts with bends rounded by radius r (inches) and an optional arrow head at its end."""
        self.ops.append(dict(kind='path', z=3, pts=[tuple(map(float, p)) for p in pts], color=color, lw=lw, dash=dash, head=head, r=r, ms=ms))

    def text(self, x, y, s, size=6.5, color='#1d221c', ha='left', va='center', bold=False, mono=False, italic=False,
             rot=0, ls=1.25, halo=None, small_ok=False):
        self.ops.append(dict(kind='text', z=3, x=x, y=y, s=s, size=size, color=color, ha=ha, va=va, bold=bold, mono=mono,
                             italic=italic, rot=rot, ls=ls, halo=halo, small_ok=small_ok))

    def image(self, x, y, w, h, png, name='image'):
        self.ops.append(dict(kind='image', z=1, x=x, y=y, w=w, h=h, png=png, name=name))

    def plot(self, x, y, w, h, draw, name='plot', dpi=600):
        """draw(fig, box): draws with matplotlib inside box = (l, b, w, h) in figure fractions (use inset()).
        Vector in the PDF; rendered to a PNG of the same size for draw.io."""
        self.ops.append(dict(kind='plot', z=1, x=x, y=y, w=w, h=h, draw=draw, name=name, dpi=dpi))

    # ---- checks
    def lint(self, min_size=6.0, verbose=True):
        """Text below min_size (unless small_ok), and text or shapes that leave the canvas. Returns the list of problems."""
        bad = []
        for o in self.ops:
            if o['kind'] == 'text':
                if o['size'] < min_size - 1e-9 and not o['small_ok']:
                    bad.append(f"small text {o['size']:g} pt: {o['s'][:40]!r}")
                w, h = text_size(o['s'], o['size'], o['bold'], o['mono'], o['italic'], o['ls'])
                if o['rot'] in (90, -90, 270): w, h = h, w
                x0 = o['x'] - {'left': 0, 'center': w / 2, 'right': w}[o['ha']]
                y0 = o['y'] - {'bottom': 0, 'center': h / 2, 'top': h, 'baseline': 0}[o['va']]
                if x0 < -0.005 or y0 < -0.005 or x0 + w > self.W + 0.005 or y0 + h > self.H + 0.005:
                    bad.append(f"text off canvas ({x0:.2f}..{x0 + w:.2f}, {y0:.2f}..{y0 + h:.2f}): {o['s'][:40]!r}")
            elif o['kind'] in ('rect', 'image', 'plot'):
                if o['x'] < -0.005 or o['y'] < -0.005 or o['x'] + o['w'] > self.W + 0.005 or o['y'] + o['h'] > self.H + 0.005:
                    bad.append(f"{o['kind']} off canvas at ({o['x']:.2f}, {o['y']:.2f}) size ({o['w']:.2f}, {o['h']:.2f}) {o.get('name', '')}")
        if verbose:
            n = sum(1 for o in self.ops if o['kind'] == 'text'); words = sum(len(o['s'].split()) for o in self.ops if o['kind'] == 'text')
            sizes = sorted({o['size'] for o in self.ops if o['kind'] == 'text'})
            print(f'lint {self.name}: {n} text items, {words} words, sizes {sizes}, {len(bad)} problems', flush=True)
            for b in bad: print('   ', b, flush=True)
        return bad

    # ---- renderer 1: matplotlib
    @staticmethod
    def _rounded(pts, r):
        """Path through pts with each interior corner replaced by a quadratic curve of reach r."""
        P = [np.asarray(p, float) for p in pts]
        if len(P) < 3 or r <= 0: return MPath(P, [MPath.MOVETO] + [MPath.LINETO] * (len(P) - 1))
        v, c = [P[0]], [MPath.MOVETO]
        for i in range(1, len(P) - 1):
            a, b, d = P[i - 1], P[i], P[i + 1]
            la, ld = np.linalg.norm(b - a), np.linalg.norm(d - b); rr = min(r, la / 2, ld / 2)
            v += [b - (b - a) / la * rr, b, b + (d - b) / ld * rr]; c += [MPath.LINETO, MPath.CURVE3, MPath.CURVE3]
        v.append(P[-1]); c.append(MPath.LINETO)
        return MPath(v, c)

    def render_mpl(self, stem, dpi=300):
        W, H = self.W, self.H
        fig = plt.figure(figsize=(W, H))

        def overlay():
            ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_axis_off(); ax.patch.set_alpha(0)
            return ax

        def draw_shape(ax, o):
            kw = dict(fill=bool(o['fill']), facecolor=o['fill'] or 'none', edgecolor=o['stroke'], lw=o['lw'],
                      linestyle=DASH if o.get('dash') else '-')
            if o['kind'] == 'rect':
                ax.add_patch(mpatches.FancyBboxPatch((o['x'], o['y']), o['w'], o['h'], boxstyle=f"round,pad=0,rounding_size={o['r']}", **kw))
            elif o['kind'] == 'poly':
                ax.add_patch(mpatches.Polygon(o['pts'], closed=True, joinstyle='round', **kw))
            else:
                ax.add_patch(mpatches.Ellipse((o['cx'], o['cy']), 2 * o['rx'], 2 * o['ry'], **kw))
        SH = ('rect', 'poly', 'ellipse')
        bg = overlay()
        for o in self.ops:
            if o['kind'] in SH and o['z'] == 0: draw_shape(bg, o)
        for o in self.ops:
            if o['kind'] == 'image':
                ax = fig.add_axes([o['x'] / W, o['y'] / H, o['w'] / W, o['h'] / H])
                ax.imshow(np.asarray(Image.open(io.BytesIO(o['png']))), aspect='auto', interpolation='lanczos'); ax.set_axis_off(); ax.patch.set_alpha(0)
            elif o['kind'] == 'plot':
                o['draw'](fig, (o['x'] / W, o['y'] / H, o['w'] / W, o['h'] / H))
        top = overlay()
        for o in self.ops:
            if o['kind'] in SH and o['z'] != 0: draw_shape(top, o)
            elif o['kind'] == 'arrow':
                top.add_patch(FancyArrowPatch(o['p'], o['q'], arrowstyle='-|>' if o['head'] else '-', mutation_scale=o['ms'],
                                              color=o['color'], lw=o['lw'], shrinkA=0, shrinkB=0, linestyle=DASH if o['dash'] else '-',
                                              connectionstyle=f"arc3,rad={o['curve']}", joinstyle='miter', capstyle='butt'))
            elif o['kind'] == 'path':
                top.add_patch(FancyArrowPatch(path=self._rounded(o['pts'], o['r']), arrowstyle='-|>' if o['head'] else '-',
                                              mutation_scale=o['ms'], color=o['color'], lw=o['lw'], linestyle=DASH if o['dash'] else '-',
                                              fill=False, joinstyle='round', capstyle='butt'))
                if o['head']:                                    # a filled head on the last segment (a path patch is unfilled)
                    a, b = np.asarray(o['pts'][-2]), np.asarray(o['pts'][-1]); u = (b - a) / max(np.linalg.norm(b - a), 1e-9)
                    top.add_patch(FancyArrowPatch(b - u * 0.03, b, arrowstyle='-|>', mutation_scale=o['ms'], color=o['color'], lw=o['lw'],
                                                  shrinkA=0, shrinkB=0))
            elif o['kind'] == 'text':
                t = top.text(o['x'], o['y'], o['s'], fontsize=o['size'], color=o['color'], ha=o['ha'], va=o['va'], rotation=o['rot'],
                             fontweight='bold' if o['bold'] else 'normal', family=MONO if o['mono'] else SANS,
                             style='italic' if o['italic'] else 'normal', linespacing=o['ls'])
                if o['halo']: t.set_path_effects([pe.withStroke(linewidth=1.6, foreground=o['halo'])])
        for ext in ('pdf', 'png'):
            fig.savefig(f'{stem}.{ext}', dpi=dpi)
        plt.close(fig)

    # ---- renderer 2: draw.io
    def render_drawio(self, path):
        W, H = self.W, self.H
        cells, n = [], [1]
        pt = lambda v: v * PX / 72

        def vertex(x, y, w, h, style, value=''):
            n[0] += 1
            cells.append(f'<mxCell id="c{n[0]}" value="{xml_escape(value, {chr(34): "&quot;"})}" style="{style}" vertex="1" parent="1">'
                         f'<mxGeometry x="{x * PX:.1f}" y="{(H - y - h) * PX:.1f}" width="{w * PX:.1f}" height="{h * PX:.1f}" as="geometry"/></mxCell>')

        def edge(p, q, color, lw, dash, head, ms, waypoints=(), curved=False, rounded=0.0):
            n[0] += 1
            pts = ''.join(f'<mxPoint x="{x * PX:.1f}" y="{(H - y) * PX:.1f}"/>' for x, y in waypoints)
            pts = f'<Array as="points">{pts}</Array>' if pts else ''
            style = (f"endArrow={'block' if head else 'none'};endFill=1;endSize={pt(ms * 0.46):.0f};html=1;"
                     f"{'curved=1;' if curved else ('rounded=1;arcSize=' + format(rounded * PX, '.0f') + ';absoluteArcSize=1;edgeStyle=none;' if rounded else 'rounded=0;edgeStyle=none;')}"
                     f"strokeColor={color};strokeWidth={pt(lw):.1f};{'dashed=1;' if dash else ''}")
            cells.append(f'<mxCell id="c{n[0]}" value="" style="{style}" edge="1" parent="1"><mxGeometry relative="1" as="geometry">'
                         f'<mxPoint x="{p[0] * PX:.1f}" y="{(H - p[1]) * PX:.1f}" as="sourcePoint"/>'
                         f'<mxPoint x="{q[0] * PX:.1f}" y="{(H - q[1]) * PX:.1f}" as="targetPoint"/>{pts}</mxGeometry></mxCell>')
        for o in sorted(self.ops, key=lambda o: o['z']):
            k = o['kind']
            if k == 'rect':
                vertex(o['x'], o['y'], o['w'], o['h'],
                       f"rounded=1;absoluteArcSize=1;arcSize={2 * o['r'] * PX:.0f};whiteSpace=wrap;html=1;fillColor={o['fill'] or 'none'};"
                       f"strokeColor={o['stroke']};strokeWidth={pt(o['lw']):.1f};{'dashed=1;' if o['dash'] else ''}")
            elif k == 'poly':
                xs, ys = [p[0] for p in o['pts']], [p[1] for p in o['pts']]
                x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys); w, h = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
                coords = ','.join(f'[{(x - x0) / w:.4f},{(y1 - y) / h:.4f}]' for x, y in o['pts'])
                vertex(x0, y0, w, h, f"shape=mxgraph.basic.polygon;polyCoords=[{coords}];polyline=0;html=1;fillColor={o['fill'] or 'none'};"
                                     f"strokeColor={o['stroke']};strokeWidth={pt(o['lw']):.1f};{'dashed=1;' if o['dash'] else ''}")
            elif k == 'ellipse':
                vertex(o['cx'] - o['rx'], o['cy'] - o['ry'], 2 * o['rx'], 2 * o['ry'],
                       f"ellipse;html=1;fillColor={o['fill'] or 'none'};strokeColor={o['stroke']};strokeWidth={pt(o['lw']):.1f};")
            elif k in ('image', 'plot'):
                if k == 'plot':
                    f = plt.figure(figsize=(o['w'], o['h'])); o['draw'](f, (0, 0, 1, 1))
                    b = io.BytesIO(); f.savefig(b, format='png', dpi=o['dpi'], transparent=True); plt.close(f); png = b.getvalue()
                else:
                    png = o['png']
                vertex(o['x'], o['y'], o['w'], o['h'], f'shape=image;imageAspect=0;html=1;image=data:image/png,{base64.b64encode(png).decode()};', '')
            elif k == 'arrow':
                way = ()
                if o['curve']:                                     # one waypoint at the arc's midpoint, curved=1
                    (x0, y0), (x1, y1) = o['p'], o['q']
                    mx, my = (x0 + x1) / 2, (y0 + y1) / 2; dx, dy = x1 - x0, y1 - y0
                    way = ((mx + o['curve'] * dy, my - o['curve'] * dx),)
                edge(o['p'], o['q'], o['color'], o['lw'], o['dash'], o['head'], o['ms'], way, curved=bool(o['curve']))
            elif k == 'path':
                edge(o['pts'][0], o['pts'][-1], o['color'], o['lw'], o['dash'], o['head'], o['ms'], o['pts'][1:-1], rounded=o['r'])
            elif k == 'text':
                w, h = text_size(o['s'], o['size'], o['bold'], o['mono'], o['italic'], o['ls']); w += 0.04; h += 0.03
                if o['rot']: w, h = h, w
                x = o['x'] - {'left': 0, 'center': w / 2, 'right': w}[o['ha']]
                y = o['y'] - {'bottom': 0, 'center': h / 2, 'top': h, 'baseline': 0}[o['va']]
                label = '<br>'.join(html.escape(l, quote=False).replace(' ', '&nbsp;') for l in o['s'].split('\n'))
                label = f'<div style="line-height:{o["ls"]}">{label}</div>'
                style = (f"text;html=1;strokeColor=none;fillColor=none;whiteSpace=nowrap;spacing=0;align={o['ha']};"
                         f"verticalAlign={ {'bottom': 'bottom', 'center': 'middle', 'top': 'top', 'baseline': 'bottom'}[o['va']] };"
                         f"fontFamily={'Courier New' if o['mono'] else 'Helvetica'};fontSize={pt(o['size']):.1f};fontColor={o['color']};"
                         f"fontStyle={(1 if o['bold'] else 0) + (2 if o['italic'] else 0)};" + ('horizontal=0;' if o['rot'] else '')
                         + (f"labelBackgroundColor={o['halo']};" if o['halo'] else ''))
                vertex(x, y, w, h, style, label)
        xml = (f'<mxfile host="figures/_scene.py"><diagram name="{xml_escape(self.name)}" id="{self.name.replace(" ", "_")}">'
               f'<mxGraphModel dx="0" dy="0" grid="0" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" '
               f'pageScale="1" pageWidth="{W * PX:.0f}" pageHeight="{H * PX:.0f}" math="0" shadow="0" background="#ffffff">'
               f'<root><mxCell id="0"/><mxCell id="1" parent="0"/>\n' + '\n'.join(cells) + '\n</root></mxGraphModel></diagram></mxfile>')
        pathlib.Path(path).write_text(xml, encoding='utf-8')
        return len(cells)

    def render(self, stem, dpi=300, drawio=True):
        stem = pathlib.Path(stem)
        self.render_mpl(stem, dpi=dpi)
        msg = f'wrote {stem.with_suffix(".pdf")} {stem.with_suffix(".png")}'
        if drawio:
            n = self.render_drawio(stem.with_suffix('.drawio'))
            msg += f' {stem.with_suffix(".drawio")} ({n} draw.io cells, {stem.with_suffix(".drawio").stat().st_size / 1e6:.2f} MB)'
        print(msg, flush=True)


# ---- image helpers
def png_array(arr, max_side=448):
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    s = max_side / max(im.size)
    if s > 1: im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, 'PNG', optimize=True); return b.getvalue()


def png_file(path):
    return pathlib.Path(path).read_bytes()


def png_fig(fig, dpi=600, transparent=True):
    b = io.BytesIO(); fig.savefig(b, format='png', dpi=dpi, transparent=transparent, facecolor='none' if transparent else 'white')
    plt.close(fig); return b.getvalue()


def png_cloud(P, side, dpi=400, elev=12, azim=-35, frame=None):
    """A cloud drawn to read as a plant: leaf greens dark at the base to light at the top, depth shading, far to near."""
    P = np.asarray(P, dtype=np.float64)
    X = np.stack([P[:, 0], P[:, 2], P[:, 1]], 1)
    e, z = np.radians(elev), np.radians(azim)
    toward = X @ np.array([np.cos(e) * np.cos(z), np.cos(e) * np.sin(z), np.sin(e)])
    h = (X[:, 2] - X[:, 2].min()) / max(np.ptp(X[:, 2]), 1e-9)
    near = (toward - toward.min()) / max(np.ptp(toward), 1e-9)
    rgb = PLANT_CMAP(0.15 + 0.85 * h)[:, :3] * (0.55 + 0.45 * near)[:, None]
    o = np.argsort(toward)
    if frame is None:
        c = (P.max(0) + P.min(0)) / 2; hw = np.ptp(P, 0).max() / 2 * 1.05; frame = (c, hw)
    ctr, hw = frame
    fig = plt.figure(figsize=(side, side), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    ax.scatter(X[o, 0], X[o, 1], X[o, 2], s=0.9, c=rgb[o], alpha=0.95, linewidths=0, depthshade=False, rasterized=True)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(ctr[0] - hw, ctr[0] + hw); ax.set_ylim(ctr[2] - hw, ctr[2] + hw); ax.set_zlim(ctr[1] - hw, ctr[1] + hw)
    ax.set_box_aspect((1, 1, 1), zoom=1.45); ax.set_axis_off()
    b = io.BytesIO(); fig.savefig(b, format='png', dpi=dpi, transparent=True); plt.close(fig); return b.getvalue()


def view_basis(elev=0.0, azim=0.0):
    """(right, up, toward) unit vectors of a viewer rotated azim degrees to the right and elev degrees up from the camera
    that made the cloud (camera frame: x right, y up, z toward the camera). elev = azim = 0 is the camera's own view."""
    e, z = np.radians(elev), np.radians(azim)
    d = np.array([np.sin(z) * np.cos(e), np.sin(e), np.cos(z) * np.cos(e)])
    r = np.array([np.cos(z), 0.0, -np.sin(z)])
    return r, np.cross(d, r), d


def cloud_frame(P, elev=0.0, azim=0.0, pad=1.06):
    """(center, half-width) of a cloud's projection in the given view: pass it to several png_points calls so that a
    prediction and its truth share one framing."""
    r, u, d = view_basis(elev, azim); P = np.asarray(P, float)
    U, Vv = P @ r, P @ u
    return ((U.min() + U.max()) / 2, (Vv.min() + Vv.max()) / 2), max(np.ptp(U), np.ptp(Vv)) / 2 * pad


def png_points(P, side=0.6, dpi=600, elev=8.0, azim=20.0, frame=None, s=0.55, color=None, alpha=0.95, shade=0.45, under=None,
               under_color='#c9ccc6', under_s=None, hfrac=None):
    """A cloud as an orthographic 2-D projection (camera frame: x right, y up, z toward the camera), drawn far to near.
    color None: leaf greens, dark at the base to light at the top (hfrac: per-point height in [0, 1] to colour by, default
    its own extent), darkened with distance. color '#rrggbb': that one colour with the same depth shading.
    under: a second cloud drawn first in a flat light tone (the scene behind a highlighted plant, or the truth behind a prediction)."""
    P = np.asarray(P, dtype=np.float64); r, u, d = view_basis(elev, azim)
    if frame is None: frame = cloud_frame(P, elev, azim)
    (cu, cv), hw = frame
    fig = plt.figure(figsize=(side, side), dpi=dpi); ax = fig.add_axes([0, 0, 1, 1])
    if under is not None:
        Q = np.asarray(under, float); ax.scatter(Q @ r, Q @ u, s=under_s or s, c=under_color, linewidths=0, alpha=0.9, rasterized=True)
    U, Vv, D = P @ r, P @ u, P @ d
    near = (D - D.min()) / max(np.ptp(D), 1e-9)
    if color is None:
        h = (Vv - Vv.min()) / max(np.ptp(Vv), 1e-9) if hfrac is None else np.asarray(hfrac)
        rgb = PLANT_CMAP(0.15 + 0.85 * h)[:, :3]
    else:
        rgb = np.tile(np.array(matplotlib.colors.to_rgb(color)), (len(P), 1))
    rgb = rgb * ((1 - shade) + shade * near)[:, None]
    o = np.argsort(D)
    ax.scatter(U[o], Vv[o], s=s, c=rgb[o], linewidths=0, alpha=alpha, rasterized=True)
    ax.set_xlim(cu - hw, cu + hw); ax.set_ylim(cv - hw, cv + hw); ax.set_aspect('equal'); ax.set_axis_off()
    return png_fig(fig, dpi)


def png_icon(kind, px=256):
    """Small raster marks: 'snow' (a frozen module) and 'fire' (a trained module)."""
    fig = plt.figure(figsize=(1, 1), dpi=px); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_axis_off()
    if kind == 'snow':
        c = '#2f86c9'
        for k in range(3):
            a = np.pi / 3 * k; dx, dy = np.cos(a), np.sin(a)
            ax.plot([-0.86 * dx, 0.86 * dx], [-0.86 * dy, 0.86 * dy], color=c, lw=7, solid_capstyle='round')
            for sgn in (-1, 1):
                bx, by = sgn * 0.52 * dx, sgn * 0.52 * dy
                for t in (-1, 1):
                    b = a + (0 if sgn > 0 else np.pi) + t * np.pi / 3.2
                    ax.plot([bx, bx + 0.30 * np.cos(b)], [by, by + 0.30 * np.sin(b)], color=c, lw=5.5, solid_capstyle='round')
    else:
        outer = [(0.0, -0.92), (-0.72, -0.92), (-0.80, -0.10), (-0.38, 0.20), (-0.30, 0.55), (-0.02, 0.98), (0.10, 0.50), (0.40, 0.30),
                 (0.50, 0.62), (0.84, 0.05), (0.72, -0.92), (0.0, -0.92)]
        codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
                 MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4]
        outer = [(0.0, -0.92), (-0.75, -0.92), (-0.85, -0.05), (-0.36, 0.22), (-0.30, 0.60), (-0.05, 0.75), (0.02, 0.98),
                 (0.30, 0.70), (0.26, 0.42), (0.46, 0.30), (0.62, 0.42), (0.66, 0.56), (0.62, 0.66), (1.00, 0.20), (0.80, -0.92), (0.0, -0.92)]
        codes = [MPath.MOVETO] + [MPath.CURVE4] * 15
        ax.add_patch(mpatches.PathPatch(MPath(outer, codes), facecolor='#e8590c', edgecolor='none'))
        inner = [(0.02, -0.86), (-0.42, -0.86), (-0.46, -0.30), (-0.14, -0.08), (-0.06, 0.10), (0.02, 0.22), (0.04, 0.36),
                 (0.30, 0.10), (0.48, -0.30), (0.46, -0.86), (0.02, -0.86)]
        ax.add_patch(mpatches.PathPatch(MPath(inner + [(0.02, -0.86), (0.02, -0.86)], [MPath.MOVETO] + [MPath.CURVE4] * 12), facecolor='#ffc233', edgecolor='none'))
    return png_fig(fig, px)
