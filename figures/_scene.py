"""A figure described once, drawn twice: matplotlib (PDF / PNG) and draw.io (editable).

A Scene is a list of primitives in inches (x from the left, y from the bottom): rectangles (optionally rounded,
filled, dashed), text (any font size, bold / italic / mono, rotation), arrows (straight or curved, solid or dashed)
and embedded PNG images. render_mpl() draws it with matplotlib; render_drawio() writes every box, label and arrow as
a native draw.io cell and every image as an embedded PNG (data URI), so the .drawio file is self-contained and can be
opened and edited in app.diagrams.net or the desktop app. Generalised from figures/make_teaser_v2.py.
"""
import base64, html, io, pathlib
from xml.sax.saxutils import escape as xml_escape
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from PIL import Image

PX = 200                                   # draw.io canvas px per inch; font sizes and strokes scale by PX / 72
PLANT_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list('plant', ['#1f4d22', '#2f7d32', '#5aa83e', '#a8d26a'])


class Scene:
    def __init__(self, W, H, name='figure'):
        self.W, self.H, self.name, self.ops = W, H, name, []

    # ---- primitives (z: 0 filled boxes, 1 images, 2 frames, 3 text and arrows)
    def rect(self, x, y, w, h, stroke, lw=1.0, fill=None, r=0.04, dash=False, z=None):
        self.ops.append(dict(kind='rect', z=z if z is not None else (0 if fill else 2), x=x, y=y, w=w, h=h, stroke=stroke,
                             lw=lw, fill=fill, r=r, dash=dash))

    def arrow(self, p, q, color='#1d221c', lw=0.9, dash=False, curve=0.0, head=True):
        """curve: signed bend as a fraction of the arrow length (matplotlib arc3 rad); 0 = straight."""
        self.ops.append(dict(kind='arrow', z=3, p=p, q=q, color=color, lw=lw, dash=dash, curve=curve, head=head))

    def text(self, x, y, s, size=6.5, color='#1d221c', ha='left', va='center', bold=False, mono=False, italic=False,
             rot=0, ls=1.25):
        self.ops.append(dict(kind='text', z=3, x=x, y=y, s=s, size=size, color=color, ha=ha, va=va, bold=bold, mono=mono,
                             italic=italic, rot=rot, ls=ls))

    def image(self, x, y, w, h, png, name='image'):
        self.ops.append(dict(kind='image', z=1, x=x, y=y, w=w, h=h, png=png, name=name))

    # ---- renderer 1: matplotlib
    def render_mpl(self, stem, dpi=220):
        W, H = self.W, self.H
        fig = plt.figure(figsize=(W, H))

        def overlay():
            ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_axis_off(); ax.patch.set_alpha(0)
            return ax

        def draw_rect(ax, o):
            ax.add_patch(mpatches.FancyBboxPatch((o['x'], o['y']), o['w'], o['h'], boxstyle=f"round,pad=0,rounding_size={o['r']}",
                                                 fill=bool(o['fill']), facecolor=o['fill'] or 'none', edgecolor=o['stroke'],
                                                 lw=o['lw'], linestyle=(0, (3, 2)) if o['dash'] else '-'))
        bg = overlay()
        for o in self.ops:
            if o['kind'] == 'rect' and o['z'] == 0: draw_rect(bg, o)
        for o in self.ops:
            if o['kind'] == 'image':
                ax = fig.add_axes([o['x'] / W, o['y'] / H, o['w'] / W, o['h'] / H])
                ax.imshow(np.asarray(Image.open(io.BytesIO(o['png']))), aspect='auto'); ax.set_axis_off(); ax.patch.set_alpha(0)
        top = overlay()
        for o in self.ops:
            if o['kind'] == 'rect' and o['z'] != 0: draw_rect(top, o)
            elif o['kind'] == 'arrow':
                top.add_patch(FancyArrowPatch(o['p'], o['q'], arrowstyle='-|>' if o['head'] else '-', mutation_scale=7,
                                              color=o['color'], lw=o['lw'], shrinkA=0, shrinkB=0,
                                              linestyle=(0, (3, 2)) if o['dash'] else '-',
                                              connectionstyle=f"arc3,rad={o['curve']}"))
            elif o['kind'] == 'text':
                top.text(o['x'], o['y'], o['s'], fontsize=o['size'], color=o['color'], ha=o['ha'], va=o['va'], rotation=o['rot'],
                         fontweight='bold' if o['bold'] else 'normal', family='monospace' if o['mono'] else None,
                         style='italic' if o['italic'] else 'normal', linespacing=o['ls'])
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

        def edge(o):
            n[0] += 1
            (x0, y0), (x1, y1) = o['p'], o['q']
            pts = ''
            if o['curve']:                                     # one waypoint at the arc's midpoint, curved=1
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2; dx, dy = x1 - x0, y1 - y0
                cx, cy = mx + o['curve'] * dy, my - o['curve'] * dx
                pts = f'<Array as="points"><mxPoint x="{cx * PX:.1f}" y="{(H - cy) * PX:.1f}"/></Array>'
            style = (f"endArrow={'block' if o['head'] else 'none'};endFill=1;endSize={pt(3.2):.0f};html=1;"
                     f"{'curved=1;' if o['curve'] else 'rounded=0;edgeStyle=none;'}strokeColor={o['color']};"
                     f"strokeWidth={pt(o['lw']):.1f};{'dashed=1;' if o['dash'] else ''}")
            cells.append(f'<mxCell id="c{n[0]}" value="" style="{style}" edge="1" parent="1"><mxGeometry relative="1" as="geometry">'
                         f'<mxPoint x="{x0 * PX:.1f}" y="{(H - y0) * PX:.1f}" as="sourcePoint"/>'
                         f'<mxPoint x="{x1 * PX:.1f}" y="{(H - y1) * PX:.1f}" as="targetPoint"/>{pts}</mxGeometry></mxCell>')
        for o in sorted(self.ops, key=lambda o: o['z']):
            k = o['kind']
            if k == 'rect':
                vertex(o['x'], o['y'], o['w'], o['h'],
                       f"rounded=1;absoluteArcSize=1;arcSize={2 * o['r'] * PX:.0f};whiteSpace=wrap;html=1;fillColor={o['fill'] or 'none'};"
                       f"strokeColor={o['stroke']};strokeWidth={pt(o['lw']):.1f};{'dashed=1;' if o['dash'] else ''}")
            elif k == 'image':
                b64 = base64.b64encode(o['png']).decode()
                vertex(o['x'], o['y'], o['w'], o['h'], f'shape=image;imageAspect=0;html=1;image=data:image/png,{b64};', '')
            elif k == 'arrow':
                edge(o)
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
        xml = (f'<mxfile host="figures/_scene.py"><diagram name="{xml_escape(self.name)}" id="{self.name.replace(" ", "_")}">'
               f'<mxGraphModel dx="0" dy="0" grid="0" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" '
               f'pageScale="1" pageWidth="{W * PX:.0f}" pageHeight="{H * PX:.0f}" math="0" shadow="0" background="#ffffff">'
               f'<root><mxCell id="0"/><mxCell id="1" parent="0"/>\n' + '\n'.join(cells) + '\n</root></mxGraphModel></diagram></mxfile>')
        pathlib.Path(path).write_text(xml, encoding='utf-8')
        return len(cells)

    def render(self, stem):
        stem = pathlib.Path(stem)
        self.render_mpl(stem)
        n = self.render_drawio(stem.with_suffix('.drawio'))
        print('wrote', stem.with_suffix('.pdf'), stem.with_suffix('.png'), stem.with_suffix('.drawio'),
              f'({n} draw.io cells, {stem.with_suffix(".drawio").stat().st_size / 1e6:.2f} MB)', flush=True)


# ---- image helpers
def png_array(arr, max_side=448):
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    s = max_side / max(im.size)
    if s > 1: im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, 'PNG', optimize=True); return b.getvalue()


def png_file(path):
    return pathlib.Path(path).read_bytes()


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
