"""Viewer page for the maize row renders (maize_rows_seed1_{parallel,perpendicular}_* in /work/mech-ai/alloydas).

Reads three sample rows per layout and the dataset tables, and writes one self-contained HTML page body
(an Artifact): the 18 photographs of a row, one stop's photograph beside its point cloud, the whole row's
cloud merged from the 18 stops in the world frame, a top-down map of plants and camera stops, and
dataset-wide histograms. Clouds are subsampled; colours by plant (the dataset's own per-plant colour)
or by part (stalk / leaf).
usage: python -I figures/make_maize_rows_page.py --out <file.html> [--rows 0305 0481 0389] [--stop 9]
"""
import argparse, base64, csv, io, json, pathlib
import numpy as np
from PIL import Image

ROOT = pathlib.Path('/work/mech-ai/alloydas')
LAYOUTS = {'parallel': 'maize_rows_seed1_parallel_dist1m_0001-0520',
           'perpendicular': 'maize_rows_seed1_perpendicular_dist1m_clear0.5m_0001-0520'}
ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--rows', nargs='+', default=['0305', '0481', '0389'])
ap.add_argument('--stop', type=int, default=9, help='the stop shown with its own cloud')
ap.add_argument('--frame-points', type=int, default=20000)
ap.add_argument('--row-points', type=int, default=60000)
a = ap.parse_args()
rng = np.random.default_rng(0)
b64 = lambda b: base64.b64encode(b).decode()


def read_pcd(path):
    raw = path.read_bytes(); i = raw.find(b'DATA'); j = raw.find(b'\n', i) + 1
    hdr = {l.split()[0]: l.split()[1:] for l in raw[:j].decode('ascii').splitlines() if l and not l.startswith('#')}
    assert hdr['DATA'] == ['binary'], hdr['DATA']
    kind = {('F', '4'): '<f4', ('U', '4'): '<u4', ('U', '1'): 'u1', ('I', '4'): '<i4'}
    dt = np.dtype([(f, kind[(t, s)]) for f, t, s in zip(hdr['FIELDS'], hdr['TYPE'], hdr['SIZE'])])
    n = int(hdr['POINTS'][0])
    return np.frombuffer(raw, dtype=dt, count=n, offset=j)


def quat_wxyz_to_R(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def jpeg(path, width, q=82):
    im = Image.open(path).convert('RGBA'); bg = Image.new('RGBA', im.size, (255, 255, 255, 255))
    im = Image.alpha_composite(bg, im).convert('RGB')
    im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, 'JPEG', quality=q, optimize=True)
    return 'data:image/jpeg;base64,' + b64(buf.getvalue()), im.size


def pack(xyz_m, label, part):
    q = np.clip(np.round(np.asarray(xyz_m) * 1000), -32767, 32767).astype('<i2')    # millimetres
    return {'xyz': b64(q.tobytes()), 'lab': b64(np.asarray(label, 'u1').tobytes()), 'part': b64(np.asarray(part, 'u1').tobytes()), 'n': int(len(q))}


def row_data(layout, rid):
    d = ROOT / LAYOUTS[layout] / f'row_{rid}_seed1'
    plants = list(csv.DictReader(open(d / f'row_{rid}_seed1_plants.csv')))
    frames = list(csv.DictReader(open(d / f'row_{rid}_seed1_frames.csv')))
    thumbs = [jpeg(d / f['photo'], 96)[0] for f in frames]
    fs = frames[a.stop - 1]
    photo, psize = jpeg(d / fs['photo'], 300, 86)
    P = read_pcd(d / fs['pcd'])
    k = rng.choice(len(P), min(a.frame_points, len(P)), replace=False)
    cam_xyz = np.stack([P['x'], P['y'], P['z']], 1)[k]
    stop_cloud = pack(cam_xyz, P['label'][k], P['part'][k])
    # whole row: every stop's cloud moved to the world frame, de-duplicated on an 8 mm grid, subsampled
    W, L, Pt = [], [], []
    for f in frames:
        Q = read_pcd(d / f['pcd'])
        R = quat_wxyz_to_R(*(float(f[c]) for c in ('cam_to_world_qw', 'cam_to_world_qx', 'cam_to_world_qy', 'cam_to_world_qz')))
        t = np.array([float(f['camera_x_m']), float(f['camera_y_m']), float(f['camera_z_m'])])
        W.append(np.stack([Q['x'], Q['y'], Q['z']], 1).astype(np.float64) @ R.T + t); L.append(Q['label']); Pt.append(Q['part'])
    W = np.concatenate(W); L = np.concatenate(L); Pt = np.concatenate(Pt)
    _, u = np.unique(np.floor(W / 0.008).astype(np.int64), axis=0, return_index=True)
    u = rng.choice(u, min(a.row_points, len(u)), replace=False)
    row_cloud = pack(W[u], L[u], Pt[u])
    # checks: plant bases on the ground, each plant's points centred on its recorded x
    zmin = float(W[:, 2].min()); off = []
    for p in plants:
        m = L == int(p['plant_number'])
        if m.sum() > 500: off.append(abs(float(np.median(W[m, 0])) - float(p['x_m'])))
    print(f'{layout} row {rid}: {len(W)} world points from {len(frames)} stops, ground z min {zmin:+.3f} m, '
          f'plant x offset median {1000 * np.median(off):.0f} mm (n={len(off)})', flush=True)
    s0 = plants[0]
    return dict(layout=layout, row=rid, height=float(s0['height_m']), leaves=int(s0['leaves']), n_plants=len(plants),
                spacing_in=[float(p['gap_to_next_in']) for p in plants[:-1]],
                plants=[dict(n=int(p['plant_number']), x=float(p['x_m']), y=float(p['y_m']), color=p['pcd_color'],
                             leaves_adj=int(p['leaves_adjusted'])) for p in plants],
                cams=[[float(f['camera_x_m']), float(f['camera_y_m']), float(f['camera_z_m'])] for f in frames],
                cam_height=float(fs['camera_height_m']), dist=float(fs['distance_to_row_m']), hfov=float(fs['hfov_deg']),
                vfov=float(fs['vfov_deg']), img=[int(fs['width_px']), int(fs['height_px'])],
                points=[int(f['pcd_points']) for f in frames], stalks=[int(f['stalks_in_view']) for f in frames],
                thumbs=thumbs, stop=a.stop, photo=photo, stop_cloud=stop_cloud, row_cloud=row_cloud,
                stop_cam=[float(fs['camera_x_m']), float(fs['camera_y_m']), float(fs['camera_z_m'])])


def hist(vals, edges):
    c, _ = np.histogram(vals, edges); return {'edges': [float(e) for e in edges], 'counts': [int(x) for x in c]}


DATA = {'rows': {}, 'stats': {}}
for layout in LAYOUTS:
    DATA['rows'][layout] = [row_data(layout, r) for r in a.rows]
    base = ROOT / LAYOUTS[layout]
    first = {}
    for r in csv.DictReader(open(base / 'maize_rows_seed1_all_plants.csv')): first.setdefault(r['plant_id'], r)
    fr = list(csv.DictReader(open(base / 'maize_rows_seed1_all_frames.csv')))
    settings = {r['setting']: r['value'] for r in csv.DictReader(open(base / 'maize_rows_seed1_settings.csv'))}
    H = [float(r['height_m']) for r in first.values()]; NL = [int(r['leaves']) for r in first.values()]
    PTS = [int(f['pcd_points']) for f in fr]
    DATA['stats'][layout] = dict(n_rows=len(first), n_frames=len(fr), settings=settings,
                                 height=hist(H, np.arange(0.6, 2.61, 0.1)), leaves=hist(NL, np.arange(5.5, 17.6, 1)),
                                 points=hist(np.array(PTS) / 1000, np.arange(0, 375, 25)),
                                 h_range=[min(H), float(np.median(H)), max(H)], l_range=[min(NL), int(np.median(NL)), max(NL)],
                                 p_range=[min(PTS), int(np.median(PTS)), max(PTS)])

TEMPLATE = pathlib.Path(__file__).with_name('maize_rows_page_template.html').read_text()
html = TEMPLATE.replace('/*__DATA__*/null', json.dumps(DATA, separators=(',', ':')))
pathlib.Path(a.out).write_text(html, encoding='utf-8')
print('wrote', a.out, f'{len(html) / 1e6:.1f} MB')
