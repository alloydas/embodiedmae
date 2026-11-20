"""Re-render the S8/S9 gallery tiles (paper/overleaf/Figures/fig_rows_more/*_{true,pred}.png) at the Figure 4 tile spec
(504 px, plant filling the tile, the true plant on a soft ground shadow, the prediction textured over the grey truth),
from the recipe files in procedural_models_for_viz.zip: <root>/{sorghum,maize}/<plant>_row_yaw{000,090}/
{true,predicted_row_finetune}.yml. CPU only, no torch (only figures/_proc3d), ~1 min for all 80 tiles:
  python figures/make_rows_more_tiles.py <unzipped>/procedural_models_for_viz/supp_rows_galleries <out_dir> 504 [stems]
The true and predicted plant of a pair share one frame and one crop, so their heights compare by eye."""
import sys, pathlib, yaml, numpy as np
from PIL import Image
REPO = pathlib.Path('/Users/alloydas/embodiedmae')
sys.path.insert(0, str(REPO / 'figures'))
import _proc3d as P3

def wrap180(a): return (a + 180.0) % 360.0 - 180.0

def load(sp, path):
    d = yaml.safe_load(open(path))
    if sp == 'sorghum':
        s = d['Sorghums'][0]; H = float(s['Parameters']['stem_length'])
        leaves = [{k: float(l[k]) for k in ('starting_point', 'length', 'roll_angle', 'branching_angle', 'width')}
                  for l in s['Leaves']]
    else:
        p = d['plant']; H = float(p['height']); leaves = []
        for i, l in enumerate(p['leaves']):
            leaves.append(dict(distance=float(l['distance']), leafLength=float(l['leafLength']),
                               leafWidth=float(l['leafWidth']), leafAngle=float(l['leafAngle']),
                               droopiness=float(l['droopiness']), avgCurvature=float(l.get('avgCurvature', 0.0)),
                               azJitterDeg=float(wrap180(float(l['leafAzimuthDeg']) - 180.0 * i))))
    return dict(height=H, leaves=leaves, n=len(leaves))

def rgba(raw):
    import io
    return np.asarray(Image.open(io.BytesIO(raw)).convert('RGBA'))

def render_pair(sp, true_rec, pred_rec, px):
    lim = P3.lim_of([true_rec, pred_rec], sp)                     # one frame for the pair: heights compare by eye
    t = rgba(P3.png([(true_rec, P3.TEXTURE, 1.0)], sp, side=px / 300, dpi=300, lim=lim))
    p = rgba(P3.png([(true_rec, P3.TRUTH_C, 0.55), (pred_rec, P3.TEXTURE, 1.0)], sp, side=px / 300, dpi=300, lim=lim))
    return t, p

def trim_pair(t, p, pad=0.03):
    """crop both to the union of their opaque pixels, padded to a square (bases stay aligned at the bottom)"""
    u = (t[..., 3] > 5) | (p[..., 3] > 5); ys, xs = np.nonzero(u)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    s = int(max(y1 - y0, x1 - x0) * 1.12); cx = (x0 + x1) // 2          # 7 % under the base (shadow), >= 3 % above the top
    out = []
    for a in (t, p):
        c = np.zeros((s, s, 4), np.uint8); c[..., :3] = 255
        sy0 = y1 - s + int(0.07 * s); ty0 = max(0, sy0); sx0 = cx - s // 2; tx0 = max(0, sx0)
        src = a[ty0:min(a.shape[0], sy0 + s), tx0:min(a.shape[1], sx0 + s)]
        c[ty0 - sy0:ty0 - sy0 + src.shape[0], tx0 - sx0:tx0 - sx0 + src.shape[1]] = src
        out.append(c)
    return out

def ground_shadow(a, rx=0.13, ry=0.032, peak=0.42, grey=150):
    """soft elliptical shadow under the stem base, composited beneath the plant (the Fig. 4 ground-truth tiles)"""
    s = a.shape[0]; op = a[..., 3] > 40; ys = np.nonzero(op.any(1))[0]; yb = ys.max()
    cols = np.nonzero(op[max(0, yb - int(0.02 * s)):yb + 1].any(0))[0]; xb = cols.mean()
    yy, xx = np.mgrid[0:s, 0:s]
    d = ((xx - xb) / (rx * s)) ** 2 + ((yy - yb) / (ry * s)) ** 2
    sa = peak * np.clip(1 - d, 0, 1) ** 1.5                                        # shadow alpha
    fa = a[..., 3:4] / 255.0; sa = sa[..., None]
    out_a = fa + sa * (1 - fa)
    rgb = (a[..., :3] * fa + grey * sa * (1 - fa)) / np.maximum(out_a, 1e-6)
    return np.concatenate([rgb, out_a * 255], -1).clip(0, 255).astype(np.uint8)

if __name__ == '__main__':
    root, outdir, px = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3])
    only = sys.argv[4:]                                            # optional subset of '<sp>_<id>_<yaw>' stems
    outdir.mkdir(parents=True, exist_ok=True); n = 0
    for sp, pre in (('sorghum', 'Sorghum_'), ('maize', 'plant_')):
        for d in sorted((root / sp).iterdir()):
            pid, yaw = d.name[len(pre):].split('_row_yaw')
            stem = f'{sp}_{pid}_{yaw}'
            if only and stem not in only: continue
            tr, pr = load(sp, d / 'true.yml'), load(sp, d / 'predicted_row_finetune.yml')
            t, p = trim_pair(*render_pair(sp, tr, pr, px * 2))
            t = ground_shadow(t)                                         # Fig. 4: only the true plant stands on a shadow
            for a, tag in ((t, 'true'), (p, 'pred')):
                Image.fromarray(a).resize((px, px), Image.LANCZOS).save(outdir / f'{stem}_{tag}.png')
            n += 1
            print(f'{stem}: H true {tr["height"]:.2f} m / pred {pr["height"]:.2f} m, leaves {tr["n"]} / {pr["n"]}', flush=True)
    print('pairs rendered:', n)
