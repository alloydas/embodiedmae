"""Fig. 1 (teaser) in the EmbodiedMAE Fig. 1 style, as PDF/PNG and draw.io.

Left, "in simulation": for each species the four streams of one held-out plant
(RGB and depth with their 14x14 patch grid, the point cloud, the recipe) feed
one shared encoder, which completes any stream from any other.  Right, "in the
field": one RGB photo of the same plant goes through the single-sensor student
and comes out as a point cloud and a recipe: real model outputs, no ground
truth shown.  Data and model passes are make_paper_figures.load's.

    conda activate det
    python figures/make_teaser.py                      # runs both models (~2 min)
    python figures/make_teaser.py --cache /tmp/species.pkl

Writes paper/fig_teaser.{pdf,png,drawio}.
"""
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))

import argparse
import pickle
from pathlib import Path

import numpy as np

import make_paper_figures as F
import make_arch_drawio as D
from fig_scene import Scene

C, DARK, GREY, NAVY = F.C, F.DARK, F.GREY, F.NAVY


def build(S):
    sc = Scene((6.9, 2.9))
    PW = 0.1; PH = PW * 6.9 / 2.9                    # square panels
    xs = [0.03 + i * (PW + 0.008) for i in range(4)]
    ys = [0.555, 0.135]                              # sorghum, maize
    mods = ['rgb', 'depth', 'pc', 'text']
    hdr = {'rgb': 'RGB', 'depth': 'Depth', 'pc': 'Point cloud', 'text': 'Recipe'}
    EX, EW = 0.466, 0.092
    SEP = 0.572
    RX, SX, SW, OX, QX = 0.588, 0.703, 0.065, 0.782, 0.892
    QW = 0.993 - QX
    HY = 0.805

    sc.text(0.03, 0.93, EX + EW - 0.03, 0.06, 'In simulation: two crops, each plant with its recipe', size=7.5, bold=True)
    sc.text(RX, 0.93, 0.993 - RX, 0.06, 'In the field: one camera, everything else inferred', size=7.5, bold=True)
    sc.line((SEP, 0.04), (SEP, 0.92))
    for m, x in zip(mods, xs):
        sc.text(x, HY, PW, 0.03, hdr[m], size=6, bold=True, color=C[m], valign='bottom')
    sc.text(RX, HY, PW, 0.03, 'RGB photo', size=6, bold=True, color=C['rgb'], valign='bottom')
    sc.text(OX, HY, PW, 0.03, 'Point cloud', size=6, bold=True, color=C['pc'], valign='bottom')
    sc.text(QX, HY, QW, 0.03, 'Recipe', size=6, bold=True, color=C['text'], valign='bottom')

    for s, y in zip(S, ys):
        title = s['sp']['title']
        sc.text(0.0, y, 0.028, PH, title, size=6.5, bold=True, color=DARK, vertical=True)
        rgb = np.clip(s['rgb'], 0, 1)
        dep = F._depth_rgb(s['depth'], s['bg'])
        # ── left: the four streams ──
        for m, x in zip(mods, xs):
            sc.rect(x, y, PW, PH, fill='white', stroke=C[m], sw=0.9, arc=0.004)
            pad = 0.004
            if m == 'rgb':
                sc.image(x + pad, y + pad, PW - 2 * pad, PH - 2 * pad, D.img_png(rgb, C['rgb']))
            elif m == 'depth':
                sc.image(x + pad, y + pad, PW - 2 * pad, PH - 2 * pad, D.img_png(dep, C['depth']))
            elif m == 'pc':
                sc.image(x - 0.004, y - 0.004, PW + 0.008, PH + 0.008, D.pc_png(s['pc']))
            else:
                sc.recipe(x + 0.003, y + 0.003, PW - 0.006, PH - 0.016, F.gt_rows(s, (1,)), C['text'], size=4.0)
        sc.arrow((xs[3] + PW + 0.002, y + PH / 2), (EX - 0.002, y + PH / 2), color=DARK, width=0.8, both=True)
        # ── right: one photo in, cloud + recipe out ──
        r = s['rgb_only']
        sc.rect(RX, y, PW, PH, fill='white', stroke=C['rgb'], sw=0.9, arc=0.004)
        sc.image(RX + 0.004, y + 0.004, PW - 0.008, PH - 0.008, D.img_png(rgb))
        sy, sh = y + PH / 2 - 0.07, 0.14
        sc.rect(SX, sy, SW, sh, fill=NAVY, stroke=None, arc=0.01, text='Single-\nsensor\nstudent', color='white', size=5.2, bold=True)
        sc.arrow((RX + PW + 0.002, y + PH / 2), (SX - 0.002, y + PH / 2), color=C['rgb'])
        sc.rect(OX, y, PW, PH, fill='white', stroke=C['pc'], sw=0.9, arc=0.004)
        sc.image(OX - 0.004, y - 0.004, PW + 0.008, PH + 0.008, D.pc_png(r['pc']))
        sc.text(OX, y - 0.038, PW, 0.03, f"Chamfer {r['chamfer']:.4f} vs. true", size=4.4, color=GREY, valign='top')
        sc.arrow((SX + SW + 0.002, y + PH / 2), (OX - 0.002, y + PH / 2), color=C['pc'])
        sc.rect(QX, y, QW, PH, fill='white', stroke=C['text'], sw=0.9, arc=0.004)
        sc.recipe(QX + 0.003, y + 0.003, QW - 0.006, PH - 0.016, F.recipe_rows(s['sp'], r['par'], s['tv'], (1, 2)), C['text'], size=3.9)
        sc.arrow((OX + PW + 0.002, y + PH / 2), (QX - 0.002, y + PH / 2), color=C['text'])

    # the shared encoder, spanning both rows
    EY0, EY1 = ys[1], ys[0] + PH
    sc.rect(EX, EY0, EW, EY1 - EY0, fill='white', stroke=NAVY, sw=1.4, arc=0.012)
    sc.text(EX, EY0, EW, EY1 - EY0, 'Transformer\nEncoder\n+\nDecoders', size=5.6, bold=True, color=NAVY)
    sc.text(EX - 0.03, EY0 - 0.095, EW + 0.06, 0.085, '80 % of tokens hidden;\nany stream completes\nany other', size=4.4, color=GREY, valign='top')

    zs = ', '.join(f"{s['sp']['title'].lower()} {'zero-shot (pretrained arm)' if s['sp']['zero_shot'] else 'distilled'}" for s in S)
    sc.text(0.02, 0.004, 0.96, 0.03,
            f"held-out {S[0]['name']} and {S[1]['name']} (raw files in paper/data); right: real outputs, {zs}; no ground truth shown",
            size=4.6, color=GREY, italic=True, valign='bottom')
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default=None, help='pickle written by make_paper_figures.py --cache')
    ap.add_argument('--out', default=str(F.OUT))
    args = ap.parse_args()
    if args.cache and Path(args.cache).exists():
        S = pickle.load(open(args.cache, 'rb'))
        for s in S:
            s['sp'] = dict(F.SPECIES[s['species']], folder=s['sp_folder'])
    else:
        S = [F.load('sorghum'), F.load('maize')]
    sc = build(S)
    out = Path(args.out)
    sc.to_matplotlib(out / 'fig_teaser.pdf')
    sc.to_drawio(out / 'fig_teaser.drawio', name='Fig. 1 teaser')


if __name__ == '__main__':
    main()
