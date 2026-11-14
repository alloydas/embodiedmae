"""The procedural model in 3D, for figures (user, 2026-11-14: "instead of 2d procedural model I need 3d procedural model
in the figures").

Python port of site/assets/pcviewer.js PCV.procedural: the geometry of the stem-and-leaves schematic (_fig12_kit.glyph)
in 3D. Each leaf is a ribbon along a quadratic Bezier in its own vertical plane, turned about the stem by its azimuth
(maize: 180 deg x leaf index + azJitterDeg; sorghum: roll_angle). Width follows the leaf's width field, tapering to the
tip. Recipe dicts are those of figures/_plantviz.decode_recipe (metres, degrees). Drawn as shaded polygons with
matplotlib's 3D axes (y up in the plant frame -> z up on screen).
"""
import io
import numpy as np, matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import to_rgb

LEAF_C, STEM_C, TRUTH_C = '#7d67c4', '#6b5a3e', '#c4c4c4'


def mesh(rec, species, step=0.012, across=4):
    """-> list of (quad (4, 3) in plant frame x, y_up, z) and kind ('leaf' | 'stem')"""
    rad = np.radians; H = float(rec['height']); quads = []
    sr = 0.011 if species == 'maize' else 0.009
    ang = np.linspace(0, 2 * np.pi, 9); ys = np.linspace(0, H, max(2, int(H / 0.05) + 1))
    for a0, a1 in zip(ang[:-1], ang[1:]):
        for y0, y1 in zip(ys[:-1], ys[1:]):
            quads.append((np.array([[sr * np.cos(a0), y0, sr * np.sin(a0)], [sr * np.cos(a1), y0, sr * np.sin(a1)],
                                    [sr * np.cos(a1), y1, sr * np.sin(a1)], [sr * np.cos(a0), y1, sr * np.sin(a0)]]), 'stem'))
    y = 0.0
    for i, l in enumerate(rec['leaves']):
        if species == 'maize':
            y += float(l['distance']); y0 = y; ln = float(l['leafLength']); w = float(l['leafWidth'])
            az = rad(180.0 * i + float(l['azJitterDeg'])); phi0 = rad(35.0 + float(l['leafAngle'])); phi1 = phi0 - rad(float(l['droopiness']))
        else:
            y0 = float(l['starting_point']) * H; ln = float(l['length']); w = float(l['width'])
            az = rad(float(l['roll_angle'])); phi0 = rad(float(l['branching_angle'])); phi1 = phi0 + rad(75.0)
        if not ln > 0: continue
        d = np.array([np.cos(az), np.sin(az)]); e = np.array([-np.sin(az), np.cos(az)])
        p1 = np.array([0.5 * ln * np.sin(phi0), y0 + 0.5 * ln * np.cos(phi0)]); p2 = p1 + 0.5 * ln * np.array([np.sin(phi1), np.cos(phi1)])
        t = np.linspace(0, 1, max(12, int(ln / step) + 1))[:, None]
        a, b, c = (1 - t) ** 2, 2 * (1 - t) * t, t ** 2
        hz = (b * p1[0] + c * p2[0])[:, 0]; vy = (a * y0 + b * p1[1] + c * p2[1])[:, 0]
        hw = 0.5 * w * np.sin(np.pi * np.minimum(1, 0.15 + 0.85 * t[:, 0])) ** 0.6
        o = np.linspace(-1, 1, across + 1)
        P = np.stack([hz[:, None] * d[0] + (hw[:, None] * o[None]) * e[0], np.repeat(vy[:, None], len(o), 1),
                      hz[:, None] * d[1] + (hw[:, None] * o[None]) * e[1]], -1)             # (steps, across+1, 3)
        for s in range(len(t) - 1):
            for k in range(across):
                quads.append((np.array([P[s, k], P[s, k + 1], P[s + 1, k + 1], P[s + 1, k]]), 'leaf'))
    return quads


def _shade(quads, base, light):
    out = []
    for q in quads:
        n = np.cross(q[1] - q[0], q[3] - q[0]); nn = np.linalg.norm(n)
        f = 0.55 + 0.45 * (abs(n @ light) / nn if nn > 1e-12 else 0.5)
        out.append(np.clip(np.array(base) * f, 0, 1))
    return out


VIEW = {'maize': (24, -38), 'sorghum': (16, -55)}             # three-quarter views, slightly from above


def draw(ax, layers, species, elev=None, azim=None, lim=None):
    """layers: [(rec, leaf colour, alpha)], drawn in one collection so they depth-sort together"""
    elev = VIEW[species][0] if elev is None else elev; azim = VIEW[species][1] if azim is None else azim
    polys, cols = [], []
    L = np.array([0.4, 0.75, -0.55]); L /= np.linalg.norm(L)
    for rec, colr, alpha in layers:
        for q, kind in mesh(rec, species):
            base = to_rgb(STEM_C if (kind == 'stem' and colr != TRUTH_C) else colr)
            c = _shade([q], base, L)[0]
            polys.append(q[:, [0, 2, 1]]); cols.append((*c, alpha))
    ec = [c if c[3] >= 1 else (0, 0, 0, 0) for c in cols]                             # opaque: edges = faces (no seams)
    pc = Poly3DCollection(polys, facecolors=cols, edgecolors=ec, linewidths=0.25)
    ax.add_collection3d(pc)
    if lim is None:
        allp = np.concatenate(polys); lim = extent(allp)
    (cx, cy, r, z0) = lim
    ax.set_xlim(cx - r, cx + r); ax.set_ylim(cy - r, cy + r); ax.set_zlim(z0, z0 + 2 * r)
    ax.set_box_aspect((1, 1, 1), zoom=1.55); ax.view_init(elev=elev, azim=azim); ax.set_axis_off()
    try: ax.set_proj_type('persp', focal_length=0.6)
    except Exception: pass


def extent(P):
    """cube around plant-frame points already swapped to (x, z, y_up)"""
    r = max(np.ptp(P[:, 0]), np.ptp(P[:, 1]), np.ptp(P[:, 2])) / 2 * 1.04
    return (P[:, 0].min() + np.ptp(P[:, 0]) / 2, P[:, 1].min() + np.ptp(P[:, 1]) / 2, r, P[:, 2].min() - 0.02 * r)


def lim_of(recs, species):
    return extent(np.concatenate([np.concatenate([q for q, _ in mesh(r, species)])[:, [0, 2, 1]] for r in recs]))


def png(layers, species, side=2.0, dpi=300, elev=None, azim=None, lim=None):
    fig = plt.figure(figsize=(side, side), dpi=dpi); ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    draw(ax, layers, species, elev, azim, lim)
    b = io.BytesIO(); fig.savefig(b, format='png', dpi=dpi, transparent=True); plt.close(fig); return b.getvalue()
