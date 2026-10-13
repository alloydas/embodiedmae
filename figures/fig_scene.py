"""A tiny scene description with two renderers: matplotlib (PDF/PNG) and draw.io.

Coordinates are figure fractions, x from the left and y from the BOTTOM
(matplotlib's convention); the draw.io renderer flips y.  Sizes in pt.
Images are PNG bytes (see make_arch_drawio.img_png / pc_png) so both
renderers show the same pixels.
"""
import io
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import make_paper_figures as F
import make_arch_drawio as D


class Scene:
    def __init__(self, size_in):
        self.size = size_in
        self.ops = []

    # -- primitives --------------------------------------------------------
    def rect(self, x, y, w, h, fill='white', stroke=F.DARK, sw=0.8, arc=0.008, text='', color=F.DARK, size=7,
             bold=False, vertical=False):
        self.ops.append(('rect', dict(x=x, y=y, w=w, h=h, fill=fill, stroke=stroke, sw=sw, arc=arc, text=text,
                                      color=color, size=size, bold=bold, vertical=vertical)))

    def text(self, x, y, w, h, text, size=7, color=F.DARK, bold=False, italic=False, align='center', valign='middle', vertical=False):
        self.ops.append(('text', dict(x=x, y=y, w=w, h=h, text=text, size=size, color=color, bold=bold, italic=italic,
                                      align=align, valign=valign, vertical=vertical)))

    def image(self, x, y, w, h, png, border=None):
        self.ops.append(('image', dict(x=x, y=y, w=w, h=h, png=png, border=border)))

    def recipe(self, x, y, w, h, rows, color, mask=False, size=4.0):
        self.ops.append(('recipe', dict(x=x, y=y, w=w, h=h, rows=rows, color=color, mask=mask, size=size)))

    def arrow(self, p, q, color=F.DARK, width=0.8, dashed=False, both=False):
        self.ops.append(('arrow', dict(p=p, q=q, color=color, width=width, dashed=dashed, both=both)))

    def line(self, p, q, color='#adb5bd', width=0.6):
        self.ops.append(('line', dict(p=p, q=q, color=color, width=width)))

    # -- matplotlib --------------------------------------------------------
    def to_matplotlib(self, path, dpi=300):
        fig = plt.figure(figsize=self.size)
        ax = F.overlay(fig)
        for kind, o in self.ops:
            if kind == 'rect':
                F.box(ax, o['x'], o['y'], o['w'], o['h'], fill=o['fill'], edge=o['stroke'] or 'none', lw=o['sw'], rounding=o['arc'])
                if o['text']:
                    ax.text(o['x'] + o['w'] / 2, o['y'] + o['h'] / 2, o['text'], ha='center', va='center', fontsize=o['size'],
                            weight='bold' if o['bold'] else 'normal', color=o['color'], rotation=90 if o['vertical'] else 0, zorder=4, linespacing=1.15)
            elif kind == 'text':
                tx = o['x'] + {'left': 0, 'right': o['w']}.get(o['align'], o['w'] / 2)
                ty = o['y'] + {'top': o['h'], 'bottom': 0}.get(o['valign'], o['h'] / 2)
                ax.text(tx, ty, o['text'], ha=o['align'], va={'top': 'top', 'bottom': 'bottom'}.get(o['valign'], 'center'), fontsize=o['size'],
                        color=o['color'], weight='bold' if o['bold'] else 'normal', style='italic' if o['italic'] else 'normal',
                        rotation=90 if o['vertical'] else 0, zorder=4, linespacing=1.15)
            elif kind == 'image':
                a = F.img_axes(fig, o['x'], o['y'], o['w'], o['h'])
                a.imshow(np.asarray(Image.open(io.BytesIO(o['png']))), aspect='auto'); a.patch.set_alpha(0)
                if o['border']:
                    for sp_ in a.spines.values():
                        sp_.set_visible(True); sp_.set_edgecolor(o['border']); sp_.set_linewidth(0.5)
            elif kind == 'recipe':
                F.show_recipe(F.img_axes(fig, o['x'], o['y'], o['w'], o['h']), o['rows'], color=o['color'],
                              mask=[True] * len(o['rows']) if o['mask'] else None, fs=o['size'])
            elif kind == 'arrow':
                F.arrow(ax, o['p'], o['q'], color=o['color'], lw=o['width'], style='<|-|>' if o['both'] else '-|>',
                        ls=(0, (2, 1.5)) if o['dashed'] else '-')
            elif kind == 'line':
                ax.plot([o['p'][0], o['q'][0]], [o['p'][1], o['q'][1]], color=o['color'], lw=o['width'], ls=(0, (3, 3)))
        for ext in ('pdf', 'png'):
            fig.savefig(path.with_suffix('.' + ext), dpi=dpi, facecolor='white')
        plt.close(fig)
        print('wrote', path)

    # -- draw.io -----------------------------------------------------------
    def to_drawio(self, path, name='figure'):
        W, H = int(round(self.size[0] * D.PX)), int(round(self.size[1] * D.PX))
        D.W, D.H = W, H
        d = D.Diagram()
        px = lambda f: f * W
        B = lambda o: (px(o['x']), (1 - o['y'] - o['h']) * H, px(o['w']), o['h'] * H)
        P = lambda p: (px(p[0]), (1 - p[1]) * H)
        for kind, o in self.ops:
            if kind == 'rect':
                D.rect(d, *B(o), fill=o['fill'] if o['fill'] != 'white' else '#ffffff', stroke=o['stroke'], sw=o['sw'] * 2,
                       arc=max(4, int(o['arc'] * 1000)), value=o['text'].replace('\n', '<br>'), color=o['color'], size=o['size'],
                       bold=o['bold'], extra='horizontal=0;' if o['vertical'] else '')
            elif kind == 'text':
                D.text(d, *B(o), o['text'].replace('\n', '<br>'), size=o['size'], color=o['color'], bold=o['bold'], italic=o['italic'],
                       align=o['align'], valign=o['valign'], extra='horizontal=0;' if o['vertical'] else '')
            elif kind == 'image':
                D.image(d, *B(o), o['png'], border=o['border'])
            elif kind == 'recipe':
                D.text(d, *B(o), D.recipe_html(o['rows'], o['color'], o['mask']), align='left', valign='top')
            elif kind == 'arrow':
                D.arrow(d, P(o['p']), P(o['q']), color=o['color'], width=o['width'] * 1.6, dashed=o['dashed'], both=o['both'])
            elif kind == 'line':
                D.line(d, P(o['p']), P(o['q']), o['color'], o['width'] * 2)
        xml = d.xml().replace('name="Fig. 2 architecture" id="arch"', f'name="{name}" id="{name.replace(" ", "_")}"')
        path.write_text(xml)
        print('wrote', path, f'{path.stat().st_size / 1e6:.1f} MB,', len(d.cells), 'cells')
