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
  for more would silently pad with duplicates.
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
    # Invalidate when the split directory itself changed — this is what makes
    # the cache safe to keep while a Globus transfer is still adding folders.
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


class MaizeDataset4M(Dataset):
    """Maize 4M samples.

    __getitem__ returns the same 6-tuple as `SorghumDataset4M`, so the training
    loop is unchanged:
        (rgb, depth, pc, param_floats, text_valid, name)
         (3,H,W) (1,H,W) (N,3)  (1+max_leaves, 14)  (1+max_leaves,)  str
    """

    def __init__(self, data_root, img_size=224, num_points=8192, split=None,
                 max_leaves=MAX_LEAVES, view_sampling=False, view_seed=0,
                 deterministic_view=False, max_plants=None, plant_subset_seed=42):
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.num_points = num_points
        self.max_leaves = max_leaves
        self.view_sampling = view_sampling
        self.deterministic_view = deterministic_view
        self._view_seed = int(view_seed)
        self._epoch = 0

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
        counts = sorted({len(v) for v in self.plant_views})
        tag = '  [deterministic: view 0]' if self.deterministic_view else ''
        print(f"🎥 view sampling: {len(self.plant_ids)} plants, {counts} views each "
              f"-> epoch is {len(self.plant_ids)} items, not {len(self.samples)}{tag}")

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

        rgb = self.rgb_transform(Image.open(folder / 'rgb.png').convert('RGB'))
        depth = self.load_depth(folder / 'depth.png')
        pc = torch.from_numpy(self.load_pointcloud(folder / PC_FILENAME))

        cached = self._xml_names.get(folder.name)
        xml = (folder / cached) if cached else next(folder.glob('maize_*_spline.xml'))
        text_valid, param_floats = load_spline_params(xml, self.max_leaves)

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
