"""Fig. S10 (fig:supp-e10-maize) in the layout of Fig. S11 (user, 2026-11-20: "Format figure 10 same as figure 11").

S11's format: a grid of four-tile blocks (Image | Ground truth | Row fine-tune | Real-row fine-tune), column headers over
the top row only, each prediction's Chamfer distance under its tile, and one line under each block naming the row
layout and the true height. Maize keeps two things S11 does not have, because the S10 caption promises them: the
predicted height under each prediction, and the error percentile of the stop. The twelve stops fill three rows of four
blocks, parallel rows in the left two columns and perpendicular rows in the right two, so the caption's
"parallel (left) and perpendicular (right)" still holds; percentiles run left to right, then down.

layout(stops) -> Scene. stops: {'parallel': [...], 'perpendicular': [...]}, each in percentile order, each stop a dict
with img_png, truth_png, pred_png (two PNGs of one frame: synthetic row fine-tune, real-row fine-tune; trimmed here to their joint outline),
height (true, m), ch (two Chamfer distances, 1e-4 m^2), hp (two predicted heights, m), pct ('10th', ...).
Used by figures/make_supp_tikz.py (s10, from the --proc data on Nova) and figures/make_s10_from_assets.py (local).
"""
import io
import numpy as np
from PIL import Image
from _scene import Scene
from ugg_style import STREAM, STREAM_INK, STREAM_FILL, OURS

INK, NEUTRAL_E, NEUTRAL_F = '#111111', '#9a9a9a', '#f4f4f5'
BOX = {k: (STREAM[k], STREAM_FILL[k]) for k in STREAM}; BOX[None] = (NEUTRAL_E, NEUTRAL_F)


def _box(S, x, y, w, h, kind, dash=False, r=0.04):
    e, f = BOX[kind]; S.rect(x, y, w, h, e, lw=1.0, fill=f, r=r, dash=dash)


def _trim_pair(pngs, pad=0.04):
    """crop a stop's two prediction PNGs to the union of their opaque pixels (one box, so they keep one scale);
    -> (png bytes list, width / height)"""
    ims = [np.asarray(Image.open(io.BytesIO(b)).convert('RGBA')) for b in pngs]
    u = np.zeros(ims[0].shape[:2], bool)
    for a in ims: u |= a[..., 3] > 8
    ys, xs = np.nonzero(u); H, W = u.shape
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    py, px = int(pad * (y1 - y0)), int(pad * (x1 - x0))
    y0, y1, x0, x1 = max(0, y0 - py), min(H, y1 + py), max(0, x0 - px), min(W, x1 + px)
    out = []
    for a in ims:
        b = io.BytesIO(); Image.fromarray(a[y0:y1, x0:x1]).save(b, format='PNG'); out.append(b.getvalue())
    return out, (x1 - x0) / (y1 - y0)


def layout(stops):
    W, m, g, gb = 6.875, 0.025, 0.035, 0.16                     # S11's page width, tile margin, tile and block gaps
    t = (W - 0.10 - 3 * gb - 4 * 3 * g) / 16; th = t            # S11's tile width; square tiles (maize is not tall)
    F, hdr, gy, x00 = 5.8, 0.40, 0.56, 0.05
    order = []
    for r in range(3):                                          # left half parallel, right half perpendicular
        for L in ('parallel', 'perpendicular'):
            order += [(L, 2 * r, stops[L][2 * r]), (L, 2 * r + 1, stops[L][2 * r + 1])]
    H = hdr + 3 * th + 2 * gy + 0.50
    S = Scene(W, H, 'S10 real maize gallery')
    head = [('Image', INK), ('Ground\ntruth', INK), ('Row\nfine-tune', STREAM_INK['text']), ('Real-row\nfine-tune', OURS)]
    kinds = [('rgb', True), (None, False), ('text', False), ('text', False)]
    for i, (L, n, s) in enumerate(order):
        rr, cc = divmod(i, 4); xb = x00 + cc * (4 * t + 3 * g + gb); yb = H - hdr - (rr + 1) * th - rr * gy
        iw = t - 2 * m; preds, asp = _trim_pair(s['pred_png'])
        if rr == 0:
            for k, (s_, c) in enumerate(head):
                S.text(xb + k * (t + g) + t / 2, H - 0.03, s_, 6.2, color=c, ha='center', va='top', ls=1.0)
        for k in range(4):
            xk = xb + k * (t + g); _box(S, xk, yb, t, th, kinds[k][0], kinds[k][1])
            if k == 0: S.image(xk + m, yb + m, iw, iw, s['img_png'], f'm {L} {n} image')
            elif k == 1: S.image(xk + m, yb + m, iw, iw, s['truth_png'], f'm {L} {n} truth')
            else:
                j = k - 2                                       # fit inside the tile, bottom-centred: the plant stands on its floor
                pw, ph = (iw, iw / asp) if asp >= 1 else (iw * asp, iw)
                S.image(xk + m + (iw - pw) / 2, yb + m, pw, ph, preds[j], f'm {L} {n} pred {j}')
                S.text(xk + t / 2, yb - 0.04, f"CD =\n{s['ch'][j]:.0f}\n{s['hp'][j]:.2f} m", F,
                       color=STREAM_INK['text'] if j == 0 else OURS, ha='center', va='top', ls=1.05, small_ok=True)
        S.text(xb + (4 * t + 3 * g) / 2, yb - 0.36, f"{L} row, true height {s['height']:.2f} m\n{s['pct']} percentile", 6.0,
               color='#444444', ha='center', va='top', ls=1.1)
    return S
