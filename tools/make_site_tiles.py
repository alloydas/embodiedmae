"""Render the project page's task tiles: point clouds on black, 4:5, from the site's own exported data.

Reads site/data/**.json (written by tools/export_site_data.py) and writes site/data/tiles/<name>.jpg.
Colours match the viewer: green ramp by height for the plant (centre plant in rows), warm grey for neighbours.
usage: python tools/make_site_tiles.py
"""
import base64, json, pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SITE = pathlib.Path(__file__).resolve().parents[1] / 'site'
OUT = SITE / 'data' / 'tiles'
LEAF = LinearSegmentedColormap.from_list('leaf', ['#1f4d22', '#2f7d32', '#5aa83e', '#a8d26a'])
NEIGHBOUR = np.array([0x8a, 0x84, 0x76]) / 255   # the viewer's #c9c2b0, dimmed for black

# name -> (json, cloud, yaw for the view in degrees)
TILES = {
    'data':     ('single/sorghum/Sorghum_2750_06.json', 'truth', 30),
    'pretrain': ('single/sorghum/Sorghum_7712_05.json', 'recon', 20),
    'any':      ('single/maize/plant_1875_09.json', 'rgb:after', 35),
    'partial':  ('partial/maize/plant_10608_00.json', '1view:recon', 25),
    'rows':     ('rows/maize/plant_0320_y0.json', 'scene', 25),
    'real':     ('real/maize/parallel_0019_17.json', 'scene', 20),
}


def load(path, name):
    m = json.loads((SITE / 'data' / path).read_text())
    buf = base64.b64decode(m['b64'])
    c = m['clouds'][name]
    pos = np.frombuffer(buf, np.int16, c['n'] * 3, c['off']).reshape(-1, 3) / m['q']
    o = c['off'] + c['n'] * 6 + (c['n'] if c.get('err') else 0)
    mark = np.frombuffer(buf, np.uint8, c['n'], o).copy() if c.get('mark') else None
    return pos, mark


def render(pos, mark, yaw, out):
    a = np.radians(yaw)
    x = pos[:, 0] * np.cos(a) + pos[:, 2] * np.sin(a)
    z = -pos[:, 0] * np.sin(a) + pos[:, 2] * np.cos(a)
    y = pos[:, 1]
    keep = np.ones(len(y), bool)
    plant = keep if mark is None else mark.astype(bool)
    lo, hi = y[plant].min(), y[plant].max()
    col = LEAF(0.12 + 0.88 * (y - lo) / max(hi - lo, 1e-9))[:, :3]
    if mark is not None: col[~plant] = NEIGHBOUR
    order = np.argsort(z)                                    # far points first
    fig = plt.figure(figsize=(4, 5), dpi=200, facecolor='black')
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor('black'); ax.axis('off')
    ax.scatter(x[order], y[order], s=1.6 if mark is None else 1.0, c=col[order], linewidths=0)
    # frame the plant (centre plant in rows) at 4:5 with headroom for the caption
    cx = (x[plant].min() + x[plant].max()) / 2
    h = (hi - lo) * 1.12; w = h * 4 / 5
    ax.set_xlim(cx - w / 2, cx + w / 2); ax.set_ylim(lo - 0.30 * h, lo + 0.95 * h)
    ax.set_aspect('equal', adjustable='datalim')
    fig.savefig(out, facecolor='black', pil_kwargs={'quality': 88})
    plt.close(fig)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (path, cloud, yaw) in TILES.items():
        pos, mark = load(path, cloud)
        render(pos, mark, yaw, OUT / f'{name}.jpg')
        print('wrote', OUT / f'{name}.jpg')
