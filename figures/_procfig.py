"""Textured 3D procedural-model tiles for the supplement figures (user, 2026-11-20: "generate rest of the supplementary
figures" like Fig. 4 / Figs. S8-S9, then "S1 to S11").

  decode_pred(sp, par, cfg)   a model's written recipe -> recipe dict (as _plantviz.decode_recipe), with the leaf
                              ranges the run decodes with (real-row fine-tunes may widen them: real_finetune.leaf_scale)
  tile(layers, sp, aspect, lim)   RGBA array of the textured 3D render (_proc3d), cropped to width/height = aspect
  shared_lim(recs, sp, heights)   one frame for several plants of a block, tall enough for every given true height (m),
                              so the written heights compare by eye
"""
import io
import numpy as np
from PIL import Image
import _proc3d as P3
import _plantviz as V


def decode_pred(sp, par, cfg=None):
    par = np.clip(np.asarray(par, np.float64), 0, 1)
    if sp == 'sorghum':
        return V.decode_recipe('sorghum', par, None, predicted=True)
    from embodied_mae_4m_maize import _PLANT_SCALE_V2, _PLANT_SHIFT_V2, PLANT_FIELDS_V2, LEAF_FIELDS_V2, _LEAF_SCALE_V2, _LEAF_SHIFT_V2
    rf = (cfg or {}).get('real_finetune', {})
    sc = np.array(rf.get('leaf_scale', _LEAF_SCALE_V2), np.float64); sh = np.array(rf.get('leaf_shift', _LEAF_SHIFT_V2), np.float64)
    p = par[0] * _PLANT_SCALE_V2 - _PLANT_SHIFT_V2
    n = int(np.clip(np.rint(p[PLANT_FIELDS_V2.index('leafCount')]), 0, 28)); nf = len(LEAF_FIELDS_V2)
    raw = par[1:1 + n, :nf] * sc[:nf] - sh[:nf]
    return dict(height=float(p[PLANT_FIELDS_V2.index('height')]), n=n, leaves=[dict(zip(LEAF_FIELDS_V2, r)) for r in raw])


def shared_lim(recs, sp, heights=(), by_height=True):
    """by_height: the frame is the tallest plant's height (leaves may run past the sides, which a portrait tile crops
    anyway); else the cube around every plant's full extent (_proc3d.lim_of)"""
    recs = [x for x in recs if x and x.get('leaves') is not None]
    if not by_height:
        cx, cy, r, z0 = P3.lim_of(recs, sp)
    else:
        P = np.concatenate([np.concatenate([q for q, _ in P3.mesh(x, sp)])[:, [0, 2, 1]] for x in recs])
        z0 = float(P[:, 2].min()); r = (float(P[:, 2].max()) - z0) / 2 * 1.06
        cx, cy = float(np.median(P[:, 0])), float(np.median(P[:, 1])); z0 -= 0.02 * r
    for h in heights:                                   # the frame holds a plant of height h standing at z0
        r = max(r, 0.53 * float(h))
    return (cx, cy, r, z0)


def tile(layers, sp, aspect=1.0, lim=None, px=900, elev=None, azim=None):
    """layers as _proc3d.draw: [(rec, colour or P3.TEXTURE, alpha)]; returns an RGBA float array, height px"""
    raw = P3.png(layers, sp, side=px / 300, dpi=300, elev=elev, azim=azim, lim=lim)
    a = np.asarray(Image.open(io.BytesIO(raw)).convert('RGBA'), dtype=np.float32) / 255.0
    if aspect < 1.0:                                    # portrait tile: keep the middle columns
        w = a.shape[1]; k = max(1, int(round(w * aspect))); x0 = (w - k) // 2; a = a[:, x0:x0 + k]
    return a


def ghost_layers(pred, truth=None):
    """the prediction textured, the true model as the grey ghost behind it (Figs. S8-S9)"""
    return ([(truth, P3.TRUTH_C, 0.55)] if truth else []) + [(pred, P3.TEXTURE, 1.0)]


def trim_group(tiles, pad=0.03, keep_aspect=None):
    """crop every tile of a group to the union of their opaque pixels (one box for all, so a shared frame keeps its
    scale and the plants' bases stay aligned); keep_aspect: then widen or heighten that box to width/height = it"""
    al = [t[..., 3] > 0.02 for t in tiles]; u = np.zeros_like(al[0])
    for a in al: u |= a
    ys, xs = np.nonzero(u)
    if not len(ys): return tiles
    H, W = u.shape; y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    py, px = int(pad * (y1 - y0)), int(pad * (x1 - x0)); y0, y1, x0, x1 = max(0, y0 - py), min(H, y1 + py), max(0, x0 - px), min(W, x1 + px)
    if keep_aspect:
        h, w = y1 - y0, x1 - x0
        if w / h < keep_aspect:
            nw = int(round(h * keep_aspect)); c = (x0 + x1) // 2; x0, x1 = max(0, c - nw // 2), min(W, c - nw // 2 + nw)
        else:
            nh = int(round(w / keep_aspect)); y0 = max(0, y1 - nh)               # grow upward: the bases stay at the bottom
    return [t[y0:y1, x0:x1] for t in tiles]
