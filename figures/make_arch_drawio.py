"""Editable draw.io version of Fig. 2 (paper/fig_architecture.drawio).

Same layout and the same real data as figures/make_paper_figures.py's
`architecture()`: every box, pill, arrow and label is a native draw.io shape,
and the panels that are data (input / masked / reconstruction images, the
visible-patch stacks, the two token columns, the encoder collage) are PNGs
embedded as data URIs, so the file is self-contained.

    conda activate det
    python figures/make_arch_drawio.py                        # runs the model (~2 min)
    python figures/make_arch_drawio.py --cache /tmp/species.pkl   # reuse make_paper_figures' cache

Open paper/fig_architecture.drawio in draw.io (app.diagrams.net or the desktop
app); File > Export as > PDF/SVG for the paper.
"""
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))

import argparse
import base64
import io
import pickle
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import make_paper_figures as F

W, H = 1380, 680          # canvas px (6.9 x 3.4 in at 200 px/in)
PX = 200                  # px per inch, for font sizes: pt * 200/72
MASK_FILL, MASK_TOKEN = F.MASK_FILL, F.MASK_TOKEN
C, DARK, GREY, NAVY = F.C, F.DARK, F.GREY, F.NAVY


# ── mxGraph writer ───────────────────────────────────────────────────────────
class Diagram:
    def __init__(self):
        self.cells = []
        self.n = 1

    def _id(self):
        self.n += 1
        return f'c{self.n}'

    def vertex(self, x, y, w, h, style, value=''):
        i = self._id()
        self.cells.append(f'<mxCell id="{i}" value="{escape(value, {chr(34): "&quot;"})}" style="{style}" vertex="1" parent="1">'
                          f'<mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" as="geometry"/></mxCell>')
        return i

    def edge(self, p, q, style, value=''):
        i = self._id()
        self.cells.append(f'<mxCell id="{i}" value="{escape(value, {chr(34): "&quot;"})}" style="{style}" edge="1" parent="1">'
                          f'<mxGeometry relative="1" as="geometry"><mxPoint x="{p[0]:.1f}" y="{p[1]:.1f}" as="sourcePoint"/>'
                          f'<mxPoint x="{q[0]:.1f}" y="{q[1]:.1f}" as="targetPoint"/></mxGeometry></mxCell>')
        return i

    def xml(self):
        body = '\n'.join(self.cells)
        return (f'<mxfile host="make_arch_drawio.py"><diagram name="Fig. 2 architecture" id="arch">'
                f'<mxGraphModel dx="0" dy="0" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" '
                f'page="1" pageScale="1" pageWidth="{W}" pageHeight="{H}" math="0" shadow="0" background="#ffffff">'
                f'<root><mxCell id="0"/><mxCell id="1" parent="0"/>\n{body}\n</root></mxGraphModel></diagram></mxfile>')


def fs(pt):
    return f'{pt * PX / 72:.0f}'


def rect(d, x, y, w, h, fill='#ffffff', stroke=DARK, sw=1.5, rounded=1, arc=8, value='', color=DARK, size=7, bold=False, align='center', valign='middle', extra=''):
    style = (f'rounded={rounded};arcSize={arc};whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth={sw};'
             f'fontColor={color};fontSize={fs(size)};fontStyle={1 if bold else 0};align={align};verticalAlign={valign};fontFamily=Helvetica;{extra}')
    if stroke is None:
        style = style.replace(f'strokeColor={stroke}', 'strokeColor=none')
    return d.vertex(x, y, w, h, style, value)


def text(d, x, y, w, h, value, size=7, color=DARK, bold=False, italic=False, align='center', valign='middle', rot=0, extra=''):
    fst = (1 if bold else 0) + (2 if italic else 0)
    style = (f'text;html=1;strokeColor=none;fillColor=none;align={align};verticalAlign={valign};whiteSpace=wrap;overflow=hidden;'
             f'fontSize={fs(size)};fontColor={color};fontStyle={fst};fontFamily=Helvetica;rotation={rot};{extra}')
    return d.vertex(x, y, w, h, style, value)


def image(d, x, y, w, h, png_bytes, border=None, extra=''):
    b64 = base64.b64encode(png_bytes).decode()
    style = f'shape=image;imageAspect=0;verticalLabelPosition=bottom;verticalAlign=top;image=data:image/png,{b64};{extra}'
    if border:
        style += f'imageBorder={border};'
    return d.vertex(x, y, w, h, style)


def arrow(d, p, q, color=DARK, width=1.2, dashed=False, both=False, dotted=False):
    style = (f'endArrow=classic;html=1;rounded=0;strokeColor={color};strokeWidth={width};endFill=1;'
             f'edgeStyle=none;{"dashed=1;" if dashed or dotted else ""}{"dashPattern=1 2;" if dotted else ""}'
             f'{"startArrow=classic;startFill=1;" if both else ""}')
    return d.edge(p, q, style)


def line(d, p, q, color, width=1.0, dashed=True):
    return d.edge(p, q, f'endArrow=none;html=1;strokeColor={color};strokeWidth={width};edgeStyle=none;{"dashed=1;" if dashed else ""}')


# ── panel renderers (PNG bytes) ──────────────────────────────────────────────
def png_of(draw, size_in=(1.0, 1.0), dpi=320, transparent=False):
    fig = plt.figure(figsize=size_in, dpi=dpi)
    draw(fig)
    buf = io.BytesIO()
    fig.savefig(buf, dpi=dpi, format='png', transparent=transparent, pad_inches=0)
    plt.close(fig)
    return buf.getvalue()


def img_png(im, grid_color=None):
    def draw(fig):
        a = fig.add_axes([0, 0, 1, 1]); a.imshow(im); a.axis('off')
        if grid_color:
            F._grid(a, grid_color, lw=0.6, alpha=0.8); a.axis('off')
    return png_of(draw)


def pc_png(pts, s=1.2):
    def draw(fig):
        a = fig.add_axes([0, 0, 1, 1], projection='3d')
        F.show_pc(a, pts, s=s, lim=1.0, zoom=1.35)
    return png_of(draw, transparent=True)


def stack_png(st):
    def draw(fig):
        a = fig.add_axes([0, 0, 1, 1]); a.imshow(st, aspect='auto'); a.axis('off')
    return png_of(draw, size_in=(0.2, 1.2))


def column_png(segments, gap=2, size_in=(0.1, 3.0)):
    def draw(fig):
        a = fig.add_axes([0, 0, 1, 1]); F._token_column(a, segments, gap=gap)
    return png_of(draw, size_in=size_in)


def recipe_html(rows, color, mask=False):
    tr = ''.join(f'<tr><td style="color:{color}">{escape(k)}</td><td align="right" style="color:{"#adb5bd" if mask else DARK}">'
                 f'{"&#9632;&#9632;&#9632;&#9632;&#9632;" if mask else escape(v)}</td></tr>' for k, v in rows)
    return f'<table style="width:100%;border-collapse:collapse;font-family:Courier New,monospace;font-size:{fs(4.3)}px;line-height:1.3">{tr}</table>'


# ── the figure ───────────────────────────────────────────────────────────────
def build(s, path):
    d = Diagram()
    m = s['masked']
    rows = ['rgb', 'depth', 'pc', 'text']
    n_vis = {k: int((m[f'm_{k}'] < 0.5).sum()) for k in rows}
    L = {'rgb': 196, 'depth': 196, 'pc': 196, 'text': 25}
    rgb_gt = np.clip(s['rgb'], 0, 1)
    rgb_rec = F._composite(rgb_gt, np.clip(m['rgb'], 0, 1), m['m_rgb'])
    dep_gt = F._depth_rgb(s['depth'], s['bg'])
    dep_pred = matplotlib.colormaps['viridis'](np.clip(m['depth'], 0, 1))[..., :3]; dep_pred[s['bg']] = 1.0
    dep_rec = F._composite(dep_gt, dep_pred, m['m_depth'])
    fg = (~s['bg']).astype(float)
    vis_pts = F.pc_visible_points(s)
    short = lambda rr: [(k.replace('leaf1 ', 'leaf '), v) for k, v in rr]
    rows_gt = short(F.recipe_rows(s['sp'], s['par'], s['tv'], (1,)))
    rows_pred = short(F.recipe_rows(s['sp'], m['par'], s['tv'], (1,)))

    # geometry: the same fractions as make_paper_figures.architecture, converted to px (y down)
    TOP, BOT, GAP = 0.865, 0.115, 0.02
    rh = (TOP - BOT - 3 * GAP) / 4
    ry = {r: TOP - (i + 1) * rh - i * GAP for i, r in enumerate(rows)}
    iw = rh * 3.4 / 6.9
    PW, TW, ENC_W, DEC_W = 0.019, 0.011, 0.128, 0.086
    x = 0.03
    X = {'inp': x}; x += iw + 0.006
    X['p1'] = x; x += PW + 0.007
    X['msk'] = x; x += iw + 0.008
    X['vis'] = x; x += TW + 0.007
    X['p2'] = x; x += PW + 0.009
    X['tin'] = x; x += TW + 0.009
    X['enc'] = x; x += ENC_W + 0.009
    X['tout'] = x; x += TW + 0.009
    X['dec'] = x; x += DEC_W + 0.009
    X['out'] = x; LEFT_END = x + iw + 0.006
    px = lambda f: f * W
    py = lambda f: (1 - f) * H                  # a y fraction measured from the bottom -> px from the top
    B = lambda xf, yf, wf, hf: (px(xf), py(yf + hf), px(wf), py(0) - py(hf))   # box: x, y(top), w, h in px

    # ---------- headers ----------
    text(d, *B(X['inp'], 0.94, LEFT_END - X['inp'], 0.05), 'Training', size=8, bold=True)
    for k, t in (('inp', 'Input'), ('msk', 'Masked input'), ('out', 'Reconstruction')):
        text(d, *B(X[k] - 0.01, 0.895, iw + 0.02, 0.035), t, size=6, bold=True, color=GREY, valign='bottom')
    text(d, *B(X['dec'] - 0.01, 0.895, DEC_W + 0.02, 0.035), 'Decoders', size=6, bold=True, color=GREY, valign='bottom')
    n_in = 1 + sum(n_vis[r] for r in ('rgb', 'depth', 'pc'))
    text(d, *B(X['tin'] - 0.02, 0.895, TW + 0.04, 0.045), f'{n_in}<br>tokens', size=4.2, color=GREY, valign='bottom')
    text(d, *B(X['tout'] - 0.02, 0.895, TW + 0.04, 0.045), '613<br>positions', size=4.2, color=GREY, valign='bottom')

    pill1 = {'rgb': 'Patchify', 'depth': 'Patchify', 'pc': 'FPS & kNN', 'text': 'Tokenise'}
    pill2 = {'rgb': 'Linear', 'depth': 'Linear', 'pc': 'PointNet'}
    head = {'rgb': 'Decoder<br>16²·3 / patch', 'depth': 'Decoder<br>16² / patch', 'pc': 'Decoder<br>41 pts / token', 'text': 'Decoder<br>9 floats / token'}
    for r in rows:
        y, c = ry[r], C[r]
        text(d, *B(0.0, y, 0.028, rh), F.ROW[r], size=6, bold=True, color=c, extra='horizontal=0;')
        for xx in (X['inp'], X['msk'], X['out']):
            rect(d, *B(xx, y, iw, rh), fill='#ffffff', stroke=c, sw=1.8, arc=6)
        pad = 0.004
        inner = lambda xx: B(xx + pad, y + pad, iw - 2 * pad, rh - 2 * pad)
        count = f'{n_vis[r]} / {L[r]} visible'
        if r in ('rgb', 'depth'):
            gt_, rec_, mk_ = (rgb_gt, rgb_rec, m['m_rgb']) if r == 'rgb' else (dep_gt, dep_rec, m['m_depth'])
            for xx, im_ in zip((X['inp'], X['msk'], X['out']), (gt_, F._mask_fill(gt_, mk_), rec_)):
                image(d, *inner(xx), img_png(im_, c))
            st = F._patch_stack(gt_, mk_, fg)
            sh = (rh - 0.02) * st.shape[0] / (16 * 6)
            image(d, *B(X['vis'], y + rh / 2 - sh / 2, TW, sh), stack_png(st), border=c)
        elif r == 'pc':
            for xx, pts in ((X['inp'], s['pc']), (X['msk'], vis_pts), (X['out'], m['pc'])):
                image(d, *B(xx - 0.004, y - 0.004, iw + 0.008, rh + 0.008), pc_png(pts))
            image(d, *B(X['vis'], y + 0.01, TW, rh - 0.02), column_png([(np.zeros(n_vis['pc']), c)], size_in=(0.1, 1.2)))
        else:
            for xx, rr, mk in ((X['inp'], rows_gt, False), (X['msk'], rows_gt, True), (X['out'], rows_pred, False)):
                text(d, *B(xx + 0.003, y + 0.012, iw - 0.006, rh - 0.016), recipe_html(rr, c, mk), align='left', valign='top')
            text(d, *B(X['vis'] - 0.004, y, TW + 0.008 + PW + 0.009 + TW, rh), 'target<br>only:<br>never<br>encoded', size=4.0, color=c, italic=True)
        # the visible count, bottom-right of the masked panel
        rect(d, *B(X['msk'] + iw - 0.045, y + 0.005, 0.04, 0.024), fill='#ffffff', stroke=None, arc=20,
             value=count, color=c, size=4.0, extra='opacity=90;')
        # pills
        rect(d, *B(X['p1'], y + 0.01, PW, rh - 0.02), fill=c, stroke=None, arc=15, value=pill1[r], color='#ffffff', size=5.2, bold=True, extra='horizontal=0;')
        if r != 'text':
            rect(d, *B(X['p2'], y + 0.01, PW, rh - 0.02), fill=c, stroke=None, arc=15, value=pill2[r], color='#ffffff', size=5.2, bold=True, extra='horizontal=0;')
            arrow(d, (px(X['p2'] + PW + 0.001), py(y + rh / 2)), (px(X['tin'] - 0.001), py(y + rh / 2)), color=c)
        # decoder
        rect(d, *B(X['dec'], y + 0.012, DEC_W, rh - 0.024), fill=c, stroke=None, arc=12, value=head[r], color='#ffffff', size=4.6, bold=True)
        arrow(d, (px(X['tout'] + TW + 0.001), py(y + rh / 2)), (px(X['dec'] - 0.001), py(y + rh / 2)), color=c)
        arrow(d, (px(X['dec'] + DEC_W + 0.001), py(y + rh / 2)), (px(X['out'] - 0.001), py(y + rh / 2)), color=c)

    # token columns
    image(d, *B(X['tin'], BOT, TW, TOP - BOT),
          column_png([(np.zeros(1), '#111111')] + [(np.zeros(n_vis[r]), C[r]) for r in ('rgb', 'depth', 'pc')]))
    image(d, *B(X['tout'], BOT, TW, TOP - BOT),
          column_png([(np.zeros(1), '#111111')] + [(m[f'm_{r}'], C[r]) for r in rows], gap=3))
    # encoder
    rect(d, *B(X['enc'], BOT, ENC_W, TOP - BOT), fill='#ffffff', stroke=NAVY, sw=2.5, arc=10)
    text(d, *B(X['enc'], TOP - 0.1, ENC_W, 0.09), 'Transformer<br>Encoder', size=7.2, bold=True, color=NAVY, valign='top')
    text(d, *B(X['enc'], TOP - 0.2, ENC_W, 0.055), 'ViT-B, 12 blocks, d = 768<br>CLS + visible tokens only', size=4.4, valign='top')
    cw_ = (ENC_W - 0.028) / 3
    for i, (im_, col, pts) in enumerate(((rgb_gt, C['rgb'], None), (dep_gt, C['depth'], None), (None, C['pc'], s['pc']))):
        xx = X['enc'] + 0.01 + i * (cw_ + 0.004); yy = BOT + 0.17
        if im_ is None:
            image(d, *B(xx - 0.004, yy - 0.012, cw_ + 0.008, cw_ * 6.9 / 3.4 + 0.024), pc_png(pts, s=0.5))
        else:
            image(d, *B(xx, yy, cw_, cw_ * 6.9 / 3.4), img_png(im_), border=col)
    text(d, *B(X['enc'], BOT + 0.04, ENC_W, 0.11), 'mask ratio 0.8;<br>Dirichlet(α = 1) splits the<br>visible budget across the<br>three sensor streams', size=4.2, valign='top')
    mid = (TOP + BOT) / 2
    arrow(d, (px(X['tin'] + TW + 0.001), py(mid)), (px(X['enc'] - 0.001), py(mid)), color=NAVY, width=1.6)
    arrow(d, (px(X['enc'] + ENC_W + 0.001), py(mid)), (px(X['tout'] - 0.001), py(mid)), color=NAVY, width=1.6)
    # legend
    ly = 0.072
    for i, (t, col) in enumerate((('CLS', '#111111'), ('RGB', C['rgb']), ('depth', C['depth']), ('point cloud', C['pc']), ('recipe', C['text']), ('mask token', MASK_TOKEN))):
        xx = 0.03 + i * 0.072
        rect(d, *B(xx, ly - 0.008, 0.012, 0.016), fill=col, stroke=None, rounded=0)
        text(d, *B(xx + 0.014, ly - 0.012, 0.055, 0.024), t, size=4.5, align='left')
    text(d, *B(0.03, 0.028, 0.45, 0.024), 'decoders: 8 shared blocks, d = 512, then one head per stream', size=4.5, color=GREY, align='left')

    # ---------- distillation ----------
    sx = LEFT_END + 0.012
    line(d, (px(sx), py(0.97)), (px(sx), py(0.03)), '#adb5bd', 1.2)
    RX0 = LEFT_END + 0.026; RW = 0.995 - RX0
    text(d, *B(RX0, 0.94, RW, 0.05), 'Distillation', size=8, bold=True)
    GAPB = 0.034
    bw = (RW - GAPB) / 2; bx = {'T': RX0, 'S': RX0 + bw + GAPB}
    by0, by1 = 0.22, 0.9
    for k, title in (('T', 'Teacher (frozen)'), ('S', 'Student')):
        rect(d, *B(bx[k], by0, bw, by1 - by0), fill='#ffffff', stroke=DARK, sw=2.0, arc=6)
        text(d, *B(bx[k], by1 - 0.045, bw, 0.04), title, size=6, bold=True, valign='top')
    blocks = [('Modality<br>Encoders', C['depth'], 0.255, 0.085), ('Transformer<br>Encoder', C['rgb'], 0.395, 0.2), ('Decoders', C['depth'], 0.665, 0.085)]
    inner_ = 0.016
    full = [(6, C['rgb']), (6, C['depth']), (6, C['pc']), (3, C['text'])]

    def strip(x0, y0, w0, h0, segs):
        tot = sum(n for n, _ in segs); cx = x0
        for n, col in segs:
            sw_ = w0 * n / tot - 0.002
            rect(d, *B(cx, y0, sw_, h0), fill=col, stroke=None, rounded=0)
            cx += w0 * n / tot

    for k in ('T', 'S'):
        x0 = bx[k] + inner_; w0 = bw - 2 * inner_
        for name, col, yb, hb in blocks:
            rect(d, *B(x0, yb, w0, hb), fill=col, stroke=None, arc=12, value=name, color='#ffffff', size=5.2, bold=True)
        cxm = px(x0 + w0 / 2)
        arrow(d, (cxm, py(0.34)), (cxm, py(0.395)), color=GREY, dashed=True)
        arrow(d, (cxm, py(0.595)), (cxm, py(0.665)), color=GREY, dashed=True)
        strip(x0, 0.61, w0, 0.016, full if k == 'T' else [(6, C['rgb']), (6, MASK_TOKEN), (6, MASK_TOKEN), (3, MASK_TOKEN)])
        strip(x0, by0 - 0.045, w0, 0.02, full if k == 'T' else [(6, C['rgb'])])
        arrow(d, (cxm, py(by0 - 0.024)), (cxm, py(0.255)), color=GREY, dashed=True)
    text(d, *B(bx['T'], by0 - 0.1, bw, 0.05), 'all four streams,<br>recipe included', size=4.2, valign='top')
    text(d, *B(bx['S'], by0 - 0.1, bw, 0.05), 'one sensor stream<br>(here RGB), rest absent', size=4.2, valign='top')
    strip(bx['S'] + inner_, 0.775, bw - 2 * inner_, 0.016, full)
    arrow(d, (px(bx['S'] + bw / 2), py(0.75)), (px(bx['S'] + bw / 2), py(0.775)), color=GREY)
    text(d, *B(bx['S'], 0.8, bw, 0.03), 'reconstructs all four', size=4.3, valign='bottom')
    text(d, *B(bx['T'], 0.8, bw, 0.03), 'initialises the student', size=4.3, color=GREY, italic=True, valign='bottom')
    gx0, gx1 = bx['T'] + bw - inner_, bx['S'] + inner_
    for yy, t in ((0.705, 'F'), (0.49, 'c')):
        arrow(d, (px(gx0), py(yy)), (px(gx1), py(yy)), color=DARK, dotted=True, both=True, width=1.4)
        text(d, *B(gx0, yy + 0.006, gx1 - gx0, 0.03), f'<i>{t}</i>', size=5.5, bold=True, valign='bottom')
    arrow(d, (px(RX0 + 0.005), py(0.092)), (px(RX0 + 0.035), py(0.092)), color=DARK, dotted=True, both=True, width=1.4)
    text(d, *B(RX0 + 0.04, 0.07, RW - 0.04, 0.045), 'MSE:  F = decoder features at the 417 generated<br>positions (×1);  c = CLS (×0.5)', size=4.2, align='left')
    text(d, *B(RX0, 0.028, RW, 0.024), 'init = teacher weights; 80 % of steps distil, 20 % are plain training steps', size=4.1, color=GREY)
    text(d, *B(0.02, 0.0, 0.96, 0.022),
         f'all panels: held-out sorghum plant {s["name"]} (data v2): real inputs, the sampled mask, and model outputs at epoch {s["epoch"]}; '
         'RGB shown at the current (brightened) gain, the model ran at the gain it was trained on; the maize model differs only in its recipe tokens (1 + 28 of 14 floats)',
         size=4.1, color=GREY, italic=True)

    path.write_text(d.xml())
    print('wrote', path, f'{path.stat().st_size / 1e6:.1f} MB,', len(d.cells), 'cells')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default=None, help='pickle written by make_paper_figures.py --cache')
    ap.add_argument('--out', default=str(F.OUT / 'fig_architecture.drawio'))
    args = ap.parse_args()
    if args.cache and Path(args.cache).exists():
        so, _ = pickle.load(open(args.cache, 'rb'))
        so['sp'] = dict(F.SPECIES[so['species']], folder=so['sp_folder'])
    else:
        so = F.load('sorghum')
    build(so, Path(args.out))


if __name__ == '__main__':
    main()
