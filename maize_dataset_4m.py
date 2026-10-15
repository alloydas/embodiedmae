#!/usr/bin/env python3
"""4M dataset for MAIZE — RGB + Depth + PointCloud + procedural params (XML).

A standalone class, deliberately NOT a subclass of `SorghumDataset`. The two
species are kept as separate pipelines, and inheriting would mean a change to
sorghum's loader silently changes maize behaviour — the failure mode that is
hardest to notice and hardest to attribute later. The two decoders below are
therefore copies, verified byte-equivalent against sorghum's (see below), not
imports.

Why sorghum's class cannot read this data at all: `sorghum_dataset.py:181`
globs `'*_nc_cam.ply'`, maize's file is the literal `pointcloud_cam.ply`, every
folder fails the three-way existence check, and `__init__` raises
"No valid samples found".

Verified format facts (measured over the complete 2,250-plant test split):

* **rgb.png** — RGB, 1024², uint8, same as sorghum. The Resize(224) +
  ToTensor + ImageNet-normalise transform is valid unchanged. Note the input
  statistics differ sharply though: maize uses the full 8-bit range (foreground
  reaches +2.2σ after normalisation) while sorghum is clipped near +0.08. That
  is a real domain shift for any sorghum→maize warm start — a modelling
  concern, not a loader one.
* **depth.png** — RGBA, 1024², uint8, **the same big-endian packed uint32** as
  sorghum. `camera_pose.json` states it: "RGBA uint32 of (z - near)/(far - near);
  0 = background". Confirmed two ways: a byte-order discriminator (mean |horiz.
  gradient| 4.8e-04 for big-endian RGBA vs 1.6e-01 for every other ordering —
  300-3000x), and reprojecting `pointcloud_cam.ply` through the JSON intrinsics,
  where 90.4 % of points land within 1e-3 of the decoded depth. Decoded maize
  foreground spans ~0.26-0.74 against sorghum's ~0.03-0.07, because maize
  normalises per plant by its own near/far.
* **pointcloud_cam.ply** — binary LE double, no normals, **exactly 8,192
  points** (sorghum has ~38k). `num_points` therefore defaults to 8192: asking
  for more would silently pad with duplicates. **It is what its own camera sees,
  not the whole plant** (tools/check_maize_cloud_views.py, 2026-10-15, 20 val
  plants: 2 % of a cloud's points are hidden from the camera that made it, 25 %
  from another view's camera, and the depth image back-projects to within 1 mm
  of the cloud). Sorghum's `_nc_cam.ply` is the complete plant in the camera's
  frame; a maize model trains on single-camera clouds, and the ten views of a
  plant are ten different clouds (eval/pc_view_eval_maize.py fuses them).
* **`<split>/_params.json`** exists per split but is not read here; the
  per-folder XML is the source of truth and is byte-identical across a plant's
  ten view folders.

`persistent_workers` must stay OFF in any DataLoader over this class, exactly as
for sorghum: a persistent worker keeps the copy of the dataset it made on first
iteration, so `set_epoch()` never reaches it and view sampling silently freezes
at epoch 0.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from embodied_mae_4m_maize import MAX_LEAVES, load_spline_params

PC_FILENAME = 'pointcloud_cam.ply'
INDEX_CACHE_VERSION = 1


def _index_cache_dir():
    """Anchored on this file, so cwd never changes which cache is used."""
    env = os.environ.get('MAIZE_INDEX_CACHE')
    return Path(env) if env else Path(__file__).resolve().parent / '.dataset_index_cache'


def _index_cache_path(load_dir, tag):
    # 'maize_' prefix keeps this in its own namespace: a sorghum and a maize
    # split could otherwise share a directory name and collide in the cache.
    key = f"maize|{Path(load_dir).resolve()}|{tag}|v{INDEX_CACHE_VERSION}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return _index_cache_dir() / f"maize_{Path(load_dir).name}_{tag}_{digest}.json"


def _read_index_cache(load_dir, tag):
    p = _index_cache_path(load_dir, tag)
    if os.environ.get('MAIZE_INDEX_REBUILD'):
        return None
    try:
        blob = json.loads(p.read_text())
    except Exception:
        return None
    # Invalidate when the split directory itself changed. This catches folders
    # ADDED mid-transfer, but NOT files arriving inside a folder that already
    # exists (that never touches the split dir's mtime) -- so an index built
    # during a transfer can stay short for good. Build it only after the transfer
    # completes; slurm/delta/check_split_index.py enforces that on Delta.
    if blob.get('mtime_ns') != Path(load_dir).stat().st_mtime_ns:
        return None
    return blob.get('entries')


def _write_index_cache(load_dir, tag, entries):
    p = _index_cache_path(load_dir, tag)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f'.tmp{os.getpid()}')
    tmp.write_text(json.dumps({
        'mtime_ns': Path(load_dir).stat().st_mtime_ns,
        'entries': entries,
    }))
    os.replace(tmp, p)          # atomic: a reader never sees a half-written index


VIEW_TABLE_DEFAULT = Path(__file__).resolve().parent / 'data_split' / 'maize_view_elevation.csv.gz'
_VIEW_RANKS = ('top', 'horizon', 'bottom')


def parse_train_views(spec):
    """Canonical train_views: None, a sorted list of view indices, or '<rank>:<k>'.

    Accepts what a YAML or a CLI hands over: None / 'null', a list of ints, a comma
    string of ints ('4,5'), or a rank spec 'top:2' / 'horizon:2' / 'bottom:1'. The
    canonical form is what config.json records and what a resume is checked against.
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        t = spec.strip()
        if t in ('', 'null', 'None'):
            return None
        if ':' in t:
            rank, _, k = t.partition(':')
            rank = rank.strip().lower()
            if rank not in _VIEW_RANKS or not k.strip().isdigit() or int(k) < 1:
                raise ValueError(f"train_views {spec!r}: expected '<top|horizon|bottom>:<k>' with k >= 1")
            return f'{rank}:{int(k)}'
        spec = [int(x) for x in t.replace(',', ' ').split()]
    views = sorted({int(v) for v in spec})
    if not views:
        raise ValueError('train_views must name at least one view')
    if not all(0 <= v <= 9 for v in views):
        raise ValueError(f'train_views must be view indices 0-9, got {views}')
    return views


def load_view_table(path, split):
    """{plant: [(view, sin_elev), ...]} for one split, from tools/maize_view_elevation.py's CSV."""
    import csv
    import gzip
    p = Path(path or os.environ.get('MAIZE_VIEW_TABLE') or VIEW_TABLE_DEFAULT)
    if not p.exists():
        raise FileNotFoundError(f'{p}: the maize view-elevation table is missing; build it with '
                                f'python tools/maize_view_elevation.py --data-root <Maize root>')
    opener = gzip.open if p.suffix == '.gz' else open
    table = {}
    with opener(p, 'rt', newline='') as fh:
        for r in csv.DictReader(fh):
            if r['split'] == split:
                table.setdefault(r['plant'], []).append((int(r['view']), float(r['sin_elev'])))
    if not table:
        raise ValueError(f'{p}: no rows for split {split!r}')
    return table


class MaizeDataset4M(Dataset):
    """Maize 4M samples.

    __getitem__ returns the same 6-tuple as `SorghumDataset4M`, so the training
    loop is unchanged:
        (rgb, depth, pc, param_floats, text_valid, name)
         (3,H,W) (1,H,W) (N,3)  (1+max_leaves, 14)  (1+max_leaves,)  str
    """

    def __init__(self, data_root, img_size=224, num_points=8192, split=None,
                 max_leaves=MAX_LEAVES, view_sampling=False, view_seed=0,
                 deterministic_view=False, max_plants=None, plant_subset_seed=42,
                 rgb_file='rgb.png', param_schema=None, train_views=None,
                 view_table=None):
        # Data version 2 (2026-10-06) adds rgba.png -- straight alpha, background
        # transparent -- and a trimmed spline XML (embodied_mae_4m_maize,
        # PLANT_FIELDS_V2). A d2 run passes rgb_file='rgba.png',
        # param_schema='v2'; the defaults are what every earlier run read.
        # The folder index still keys on rgb.png, which v2 folders also carry.
        self.data_root = Path(data_root)
        self.rgb_file = rgb_file
        self.param_schema = param_schema
        self.img_size = img_size
        self.num_points = num_points
        self.max_leaves = max_leaves
        self.view_sampling = view_sampling
        self.deterministic_view = deterministic_view
        self._view_seed = int(view_seed)
        self._epoch = 0
        # ── View regime (CVPR plan E5), the maize form ──────────────────────
        # Sorghum restricts the per-epoch draw to a band of VIEW INDICES because
        # its ten renders are an elevation ladder. Maize's are not: each index
        # spans elevations of -85..+81 deg with a random azimuth (measured over
        # the train split, tools/maize_view_elevation.py), so the band is chosen
        # per PLANT from the real camera. train_views is one of
        #   None            every view (every run before 2026-10-15)
        #   [i, j, ...]     view indices, as sorghum (e.g. [0]: view 0 only, the
        #                   view val/test are scored on)
        #   'top:k'         the k views whose camera sits highest above the plant
        #   'horizon:k'     the k views nearest the horizon
        #   'bottom:k'      the k views furthest below it
        # ranked by sin(elevation) from data_split/maize_view_elevation.csv.gz
        # (view_table / $MAIZE_VIEW_TABLE override the path). Every plant keeps
        # exactly k views, so the plant set, the steps per epoch and the schedule
        # are those of the unrestricted arm; only the augmentation pool shrinks.
        # Train split only: val/test stay on view 0 (deterministic_view).
        self.train_views = parse_train_views(train_views)
        self._view_table = view_table
        if self.train_views is not None and not self.view_sampling:
            raise ValueError('train_views needs view_sampling=True (one drawn view per plant)')

        self.load_dir = self.data_root / split if split else self.data_root
        print(f"Loading data from: {self.load_dir}")

        self.rgb_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        entries = _read_index_cache(self.load_dir, 'base')
        if entries is None:
            entries = self._scan()
            _write_index_cache(self.load_dir, 'base', entries)
            print(f"🔍 indexed {len(entries)} samples (cache written)")
        else:
            print(f"⚡ index cache hit ({len(entries)} samples) — skipped the folder scan")

        self.samples = [self.load_dir / e['name'] for e in entries]
        self._xml_names = {e['name']: e['xml'] for e in entries}
        if not self.samples:
            raise ValueError(f"No valid samples found in {self.load_dir}!")
        print(f"✅ Loaded {len(self.samples)} samples from {self.load_dir.name}")

        self._apply_plant_cap(max_plants, plant_subset_seed)
        self._group_views()

    # ── indexing ─────────────────────────────────────────────────────────────

    def _scan(self):
        """One pass over the split. Folders are plant_<id>_<view>."""
        entries = []
        skipped = 0
        for folder in sorted(self.load_dir.iterdir()):
            if not folder.is_dir():
                continue            # skips the per-split _params.json
            xml = next((f.name for f in folder.glob('maize_*_spline.xml')), None)
            ok = ((folder / 'rgb.png').exists()
                  and (folder / 'depth.png').exists()
                  and (folder / PC_FILENAME).exists()
                  and xml is not None)
            if not ok:
                skipped += 1
                continue
            entries.append({'name': folder.name, 'xml': xml})
        if skipped:
            # Expected while a transfer is in flight; loud so a half-copied
            # split is never mistaken for a complete one.
            print(f"⚠️  skipped {skipped} incomplete folders in {self.load_dir.name}")
        return entries

    @staticmethod
    def plant_of(name: str) -> str:
        """'plant_0004_07' -> 'plant_0004'. Not sorghum's parse."""
        return name.rsplit('_', 1)[0]

    @staticmethod
    def plant_id(name: str) -> int:
        """'plant_0004_07' -> 4. Joins to plant_scores.csv via plant_0004."""
        return int(name.rsplit('_', 1)[0].split('_')[-1])

    def _apply_plant_cap(self, max_plants, seed):
        """Nested random subset of plants, keeping all views of each."""
        if not max_plants:
            return
        groups = sorted({self.plant_of(p.name) for p in self.samples})
        if max_plants >= len(groups):
            print(f"max_plants={max_plants} >= {len(groups)} plants — no-op")
            return
        shuffled = list(groups)
        random.Random(int(seed)).shuffle(shuffled)
        keep = set(shuffled[:max_plants])
        self.samples = [p for p in self.samples if self.plant_of(p.name) in keep]
        self._xml_names = {k: v for k, v in self._xml_names.items()
                           if self.plant_of(k) in keep}
        print(f"🌱 capped to {max_plants} plants -> {len(self.samples)} view folders")

    def _group_views(self):
        """Group view folders by plant so an epoch is one view per plant."""
        if not self.view_sampling:
            self.plant_ids, self.plant_views = [], []
            return
        groups = {}
        for i, p in enumerate(self.samples):
            groups.setdefault(self.plant_of(p.name), []).append(i)
        # NOTE sorted() on strings: 'plant_0004' zero-pads to 4 digits but ids
        # run to 14999, so 'plant_10000' < 'plant_9987' lexicographically. Row i
        # is NOT plant i — always carry the parsed id, never the row index.
        self.plant_ids = sorted(groups)
        self.plant_views = [sorted(groups[k]) for k in self.plant_ids]
        tag = '  [deterministic: view 0]' if self.deterministic_view else ''
        if self.train_views is not None:
            self.plant_views, elev = self._restrict_views(groups)
            tag += (f"  [E5 view regime: {self.train_views!r}; realised elevation "
                    f"median {np.median(elev):+.1f} deg, p10 {np.percentile(elev, 10):+.1f}, "
                    f"p90 {np.percentile(elev, 90):+.1f}]")
        counts = sorted({len(v) for v in self.plant_views})
        print(f"🎥 view sampling: {len(self.plant_ids)} plants, {counts} views each "
              f"-> epoch is {len(self.plant_ids)} items, not {len(self.samples)}{tag}")

    def _restrict_views(self, groups):
        """Per plant, the sample indices train_views keeps, and their elevations (deg)."""
        def view_of(i):
            return int(self.samples[i].name.rsplit('_', 1)[1])
        if isinstance(self.train_views, list):
            keep = set(self.train_views)
            table = None
        else:
            rank, k = self.train_views.split(':')
            k = int(k)
            table = load_view_table(self._view_table, self.load_dir.name)
        out, elev = [], []
        for pid in self.plant_ids:
            idx = sorted(groups[pid])
            if table is None:
                sel = [i for i in idx if view_of(i) in keep]
            else:
                rows = table.get(pid)
                if rows is None or len(rows) != len(idx):
                    raise ValueError(f'{pid}: {0 if rows is None else len(rows)} rows in the view '
                                     f'table for {len(idx)} view folders; rebuild it with '
                                     f'tools/maize_view_elevation.py')
                by_view = {view_of(i): i for i in idx}
                key = {'top': lambda r: -r[1], 'bottom': lambda r: r[1],
                       'horizon': lambda r: abs(r[1])}[rank]
                chosen = sorted(rows, key=key)[:k]
                sel = sorted(by_view[v] for v, _ in chosen)
                elev.extend(math.degrees(math.asin(max(-1.0, min(1.0, s)))) for _, s in chosen)
            if not sel:
                raise ValueError(f'train_views={self.train_views!r}: {pid} has none of these views')
            out.append(sel)
        if table is None:
            table_all = None
            try:
                table_all = load_view_table(self._view_table, self.load_dir.name)
            except (FileNotFoundError, ValueError):
                pass                # the table is a report here, not a dependency
            if table_all is not None:
                for pid, sel in zip(self.plant_ids, out):
                    s_of = dict(table_all[pid])
                    elev.extend(math.degrees(math.asin(max(-1.0, min(1.0, s_of[view_of(i)]))))
                                for i in sel if view_of(i) in s_of)
        return out, (np.asarray(elev) if elev else np.asarray([float('nan')]))

    def set_epoch(self, epoch: int):
        """Re-draw which view each plant contributes. Inert when deterministic."""
        self._epoch = int(epoch)

    def __len__(self):
        return len(self.plant_views) if self.view_sampling else len(self.samples)

    def _resolve_index(self, idx):
        if not self.view_sampling:
            return idx
        views = self.plant_views[idx]
        if self.deterministic_view:
            return views[0]
        rng = random.Random(
            (self._view_seed * 1_000_003 + self._epoch) * 1_000_033 + idx)
        return views[rng.randrange(len(views))]

    # ── modality loaders (copies, not imports — see the module docstring) ─────

    def load_rgb(self, folder):
        """(3, img_size, img_size) normalised RGB from folder/self.rgb_file.

        An RGBA file (rgba.png) is composited over opaque black with its alpha,
        the same convention as sorghum's rgb_nobg.png (sorghum_dataset.load_rgb).
        """
        with Image.open(folder / self.rgb_file) as im:
            if im.mode == 'RGBA':
                im = Image.alpha_composite(Image.new('RGBA', im.size, (0, 0, 0, 255)), im)
            rgb = im.convert('RGB')
        return self.rgb_transform(rgb)

    def load_depth(self, depth_path):
        """Big-endian packed RGBA uint32 -> float32 (1, img_size, img_size)."""
        with Image.open(depth_path) as image:
            arr = np.asarray(image)
        if arr.ndim == 3:
            if arr.shape[2] != 4 or arr.dtype != np.uint8:
                raise ValueError(f"Unsupported depth image: shape={arr.shape}, "
                                 f"dtype={arr.dtype}")
            rgba = arr.astype(np.float32)
            arr = (rgba[..., 0] * (256.0 ** 3) + rgba[..., 1] * (256.0 ** 2)
                   + rgba[..., 2] * 256.0 + rgba[..., 3]) / float((256 ** 4) - 1)
        elif arr.ndim == 2:
            if np.issubdtype(arr.dtype, np.integer):
                arr = arr.astype(np.float32) / np.iinfo(arr.dtype).max
            else:
                arr = arr.astype(np.float32)
        else:
            raise ValueError(f"Unsupported depth image shape: {arr.shape}")
        depth = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0)
        return TF.resize(depth, [self.img_size, self.img_size],
                         interpolation=InterpolationMode.BILINEAR, antialias=True)

    def load_pointcloud(self, ply_path):
        """Sample/pad to num_points, centre, scale onto the unit sphere."""
        pcd = o3d.io.read_point_cloud(str(ply_path))
        points = np.asarray(pcd.points)
        if len(points) == 0:
            raise ValueError(f"Empty point cloud in {ply_path}")
        n = points.shape[0]
        if n >= self.num_points:
            # n == num_points is the normal maize case: a permutation, no loss.
            points = points[np.random.choice(n, self.num_points, replace=False)]
        else:
            pad = points[np.random.choice(n, self.num_points - n, replace=True)]
            points = np.vstack([points, pad])
        points = points - np.mean(points, axis=0)
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist > 0:
            points = points / max_dist
        return points.astype(np.float32)

    def __getitem__(self, idx):
        idx = self._resolve_index(idx)
        folder = self.samples[idx]

        rgb = self.load_rgb(folder)
        depth = self.load_depth(folder / 'depth.png')
        pc = torch.from_numpy(self.load_pointcloud(folder / PC_FILENAME))

        cached = self._xml_names.get(folder.name)
        xml = (folder / cached) if cached else next(folder.glob('maize_*_spline.xml'))
        text_valid, param_floats = load_spline_params(xml, self.max_leaves,
                                                      schema=self.param_schema)

        return rgb, depth, pc, param_floats, text_valid, folder.name


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('data_root', help='path containing train/ val/ test/')
    ap.add_argument('--split', default='test')
    args = ap.parse_args()

    ds = MaizeDataset4M(args.data_root, split=args.split,
                        view_sampling=True, deterministic_view=True)
    print(f"{args.split}: {len(ds)} plants")
    rgb, depth, pc, params, valid, name = ds[0]
    print(f"  {name}  rgb={tuple(rgb.shape)}  depth={tuple(depth.shape)}  "
          f"pc={tuple(pc.shape)}  params={tuple(params.shape)}  "
          f"valid={int(valid.sum())} tokens")
    print(f"  plant id -> {MaizeDataset4M.plant_id(name)}")
    print(f"  rgb range [{rgb.min():.3f}, {rgb.max():.3f}]  "
          f"depth [{depth.min():.4f}, {depth.max():.4f}]  "
          f"pc |max| {pc.norm(dim=1).max():.4f}")
