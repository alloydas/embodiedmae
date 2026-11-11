"""E10 data: the maize rows built from the 520 real plants (maize_rows_seed1_{parallel,perpendicular}_* in
/work/mech-ai/alloydas), one item per camera stop, converted into the maize row model's input conventions so
the Regime III models run on it unchanged (eval/eval_real_rows.py).

What the files hold (figures/make_maize_rows_page.py shows them): per stop a 640 x H RGBA photograph with an
opaque white background and a binary PCD in the OpenCV camera frame (x right, y down, z forward) holding every
plant surface point inside the field of view at ~3 mm spacing, INCLUDING points hidden behind other leaves,
labelled by plant number; per row a table of plant positions, heights and leaf counts. The camera stands 1 m
from the row, level, at half the plant height, and stop k is centred on plant k + 1.

The training rows (maize_row, row_dataset.RowDataset): a level camera 1 m from the row at half the plant
height, 40 x 40 deg field of view, OpenGL camera frame (x right, y up, looking down -z), the scene cloud = the
24,576 points the camera sees (none hidden), the target = the centre plant alone as the camera sees it (8,192
points), depth encoded (z - 0.05) / 2.95 with 0 for background, photographs composited over black.

Conversion of one stop:
  window   the training camera's 40 x 40 deg window around the optical axis. The real image is only 32.9 deg
           wide, so the two sides are padded with background (the neighbours there are simply not imaged).
  photo    the window, white background (every channel >= BG_THRESH) set to black, resized to 224 x 224.
  cloud    the points inside the window that the camera sees (z-buffer of ZB_RES^2 cells over the window,
           VIS_TOL tolerance), moved to the OpenGL frame (x, -y, -z), subsampled to 24,576, centred and scaled
           to the unit sphere exactly as RowDataset.load_clouds does.
  depth    the visible points splatted into a 224 x 224 z-buffer (2 x 2 footprint), one-pixel holes filled
           from their neighbours, encoded (z - 0.05) / 2.95.
  target   the centre plant's points alone, visible against themselves only (8,192), in the input's frame.
  truth    the centre plant's height and leaf count from the row table. The files carry no per-leaf recipe.
  occ      the share of the target's own visible points that its neighbours hide (as RowDataset's occ_frac,
           measured on points instead of pixels).
"""
import csv, pathlib
import numpy as np, torch
from PIL import Image
from torchvision import transforms

ROOT = pathlib.Path('/work/mech-ai/alloydas')
LAYOUTS = {'parallel': ROOT / 'maize_rows_seed1_parallel_dist1m_0001-0520',
           'perpendicular': ROOT / 'maize_rows_seed1_perpendicular_dist1m_clear0.5m_0001-0520'}
FOV_DEG, IMG, N_SCENE, N_GT = 40.0, 224, 24576, 8192
BG_THRESH, VIS_TOL, ZB_RES = 235, 0.010, 160
NEAR, FAR = 0.05, 3.0                       # maize_row depth encoding (training camera_pose.json near / far)
N_TOKENS, N_PARAMS = 29, 14                 # maize recipe tensor (1 plant + 28 leaf tokens); zeros, never an input


def read_pcd(path):
    raw = pathlib.Path(path).read_bytes(); i = raw.find(b'DATA'); j = raw.find(b'\n', i) + 1
    hdr = {l.split()[0]: l.split()[1:] for l in raw[:j].decode('ascii').splitlines() if l and not l.startswith('#')}
    if hdr['DATA'] != ['binary']:
        raise ValueError(f'{path}: DATA {hdr["DATA"]}, expected binary')
    kind = {('F', '4'): '<f4', ('U', '4'): '<u4', ('U', '1'): 'u1', ('I', '4'): '<i4'}
    dt = np.dtype([(f, kind[(t, s)]) for f, t, s in zip(hdr['FIELDS'], hdr['TYPE'], hdr['SIZE'])])
    return np.frombuffer(raw, dtype=dt, count=int(hdr['POINTS'][0]), offset=j)


def subsample(pts, k, rng):
    n = len(pts)
    if n >= k:
        return pts[rng.choice(n, k, replace=False)]
    return np.vstack([pts, pts[rng.choice(n, k - n, replace=True)]])


class RealRowDataset(torch.utils.data.Dataset):
    def __init__(self, layouts=('parallel', 'perpendicular'), rows=None, stops=None, seed=0, with_targets=False):
        # with_targets: also return the centre plant's own depth image (all its points, splatted alone), the
        # depth target of the real-row fine-tune (train_real_rows.py). Off = the E10 evaluation's items, unchanged.
        self.seed, self.items, self._tables, self.with_targets = seed, [], {}, with_targets
        rows = None if rows is None else {f'{int(r):04d}' for r in rows}
        for layout in layouts:
            for d in sorted(LAYOUTS[layout].glob('row_*_seed1')):
                rid = d.name.split('_')[1]
                if rows is not None and rid not in rows:
                    continue
                f_csv, p_csv = d / f'row_{rid}_seed1_frames.csv', d / f'row_{rid}_seed1_plants.csv'
                if not (f_csv.exists() and p_csv.exists()):
                    continue
                frames = list(csv.DictReader(open(f_csv)))
                if not all((d / f['photo']).exists() and (d / f['pcd']).exists() for f in frames):
                    continue                                    # row still being extracted
                for f in frames:
                    if stops is None or int(f['frame']) in stops:
                        self.items.append((layout, str(d), rid, int(f['frame'])))
        self.tf = transforms.Compose([transforms.Resize((IMG, IMG)), transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        print(f'real maize rows: {len(self.items)} stops from {len({(i[0], i[2]) for i in self.items})} rows, layouts {list(layouts)}')

    def __len__(self):
        return len(self.items)

    def _table(self, d, rid):
        if d not in self._tables:
            p = pathlib.Path(d)
            self._tables[d] = (list(csv.DictReader(open(p / f'row_{rid}_seed1_frames.csv'))),
                               list(csv.DictReader(open(p / f'row_{rid}_seed1_plants.csv'))))
        return self._tables[d]

    def __getitem__(self, idx):
        layout, d, rid, stop = self.items[idx]
        rng = np.random.default_rng(self.seed * 1000003 + idx)
        frames, plants = self._table(d, rid)
        f = frames[stop - 1]
        fx, fy, cx, cy = (float(f[k]) for k in ('fx', 'fy', 'cx', 'cy'))
        half = np.tan(np.radians(FOV_DEG / 2)) * fx                      # window half-width in image pixels
        xs = np.array([float(p['x_m']) for p in plants])
        tp = plants[int(np.argmin(np.abs(xs - float(f['camera_x_m']))))]  # the plant on the optical axis
        target = int(tp['plant_number'])

        P = read_pcd(pathlib.Path(d) / f['pcd'])
        X = np.stack([P['x'], P['y'], P['z']], 1).astype(np.float64)
        lab = P['label'].astype(np.int64)
        z = X[:, 2]; u = cx + fx * X[:, 0] / z; v = cy + fy * X[:, 1] / z
        win = (z > 0) & (np.abs(u - cx) <= half) & (np.abs(v - cy) <= half)

        def visible(mask, res):
            idx_ = np.flatnonzero(mask)
            ui = np.clip(((u[idx_] - (cx - half)) / (2 * half) * res).astype(np.int64), 0, res - 1)
            vi = np.clip(((v[idx_] - (cy - half)) / (2 * half) * res).astype(np.int64), 0, res - 1)
            zb = np.full((res, res), np.inf); np.minimum.at(zb, (vi, ui), z[idx_])
            vis = np.zeros(len(z), bool); vis[idx_] = z[idx_] <= zb[vi, ui] + VIS_TOL
            return vis

        vis_scene = visible(win, ZB_RES)
        vis_target = visible(win & (lab == target), ZB_RES)
        occ = 1.0 - float((vis_scene & vis_target).sum()) / max(1, int(vis_target.sum()))

        gl = X * np.array([1.0, -1.0, -1.0])                              # OpenCV -> OpenGL camera frame
        pin = subsample(gl[vis_scene], N_SCENE, rng)
        centre = pin.mean(0); pin = pin - centre
        scale = float(np.linalg.norm(pin, axis=1).max()) or 1.0
        pin = pin / scale
        pgt = (subsample(gl[vis_target], N_GT, rng) - centre) / scale

        # depth: the visible points splatted at 224^2 with a 2 x 2 footprint (near leaves are sampled more coarsely
        # than one pixel), then one-pixel holes filled from the 3 x 3 neighbourhood minimum
        def splat(sel):
            uf = (u[sel] - (cx - half)) / (2 * half) * IMG; vf = (v[sel] - (cy - half)) / (2 * half) * IMG
            zb = np.full((IMG, IMG), np.inf)
            for du in (0.0, 1.0):
                for dv in (0.0, 1.0):
                    ui = np.clip(np.floor(uf - 0.5 + du).astype(np.int64), 0, IMG - 1)
                    vi = np.clip(np.floor(vf - 0.5 + dv).astype(np.int64), 0, IMG - 1)
                    np.minimum.at(zb, (vi, ui), z[sel])
            pad = np.pad(zb, 1, constant_values=np.inf)
            nb = np.stack([pad[1 + dy:IMG + 1 + dy, 1 + dx:IMG + 1 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
            filled = np.isfinite(nb).sum(0)
            hole = ~np.isfinite(zb) & (filled >= 5)
            zb[hole] = np.min(nb, axis=0)[hole]
            return np.where(np.isfinite(zb), np.clip((zb - NEAR) / (FAR - NEAR), 1e-6, 1.0), 0.0).astype(np.float32)

        dep = splat(np.flatnonzero(vis_scene))

        # photograph: the 40 deg window, padded with background, white background -> black
        im = np.asarray(Image.open(pathlib.Path(d) / f['photo']).convert('RGB'))
        H, W = im.shape[:2]; r0, c0 = int(round(cy - half)), int(round(cx - half)); n = int(round(2 * half))
        canvas = np.full((n, n, 3), 255, np.uint8)
        ys, xs_ = slice(max(0, r0), min(H, r0 + n)), slice(max(0, c0), min(W, c0 + n))
        canvas[ys.start - r0:ys.stop - r0, xs_.start - c0:xs_.stop - c0] = im[ys, xs_]
        canvas[(canvas >= BG_THRESH).all(-1)] = 0
        rgb = self.tf(Image.fromarray(canvas))

        info = dict(layout=layout, row=rid, stop=stop, target=target, height=float(tp['height_m']), leaves=int(tp['leaves']),
                    occ_frac=occ, scale=scale, n_scene_visible=int(vis_scene.sum()), n_target_visible=int(vis_target.sum()),
                    plants_in_window=int(len(np.unique(lab[win]))), cam_height=float(f['camera_height_m']))
        out = (rgb, torch.from_numpy(dep)[None], torch.from_numpy(pin.astype(np.float32)),
               torch.zeros(N_TOKENS, N_PARAMS), torch.zeros(N_TOKENS), torch.from_numpy(pgt.astype(np.float32)))
        if self.with_targets:
            out = out + (torch.from_numpy(splat(np.flatnonzero(win & (lab == target))))[None],)
        return out + (info,)


def collate(batch):
    n = len(batch[0]) - 1
    out = [torch.stack([b[k] for b in batch]) for k in range(n)]
    return (*out, [b[n] for b in batch])
