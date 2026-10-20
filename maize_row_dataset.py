#!/usr/bin/env python3
"""MaizeRowDataset -- Alloy's RENDERED occluded maize rows (2026-10-16), the real version of
what occlusion_scene.py builds on the fly.

Layout, checked 2026-10-17 (`/work/mech-ai/alloydas/maize_row`):

    <root>/{train,val,test}/plant_XXXX/
        row.json                      neighbours (left / right plant ids, same split), spacing
                                      d = 0.15 m, the rover camera, the 10 orientations
        theta_TTT/shot_000/           TTT = the MAIN plant's yaw, 000 ... 090
            rgba.png    depth.png     pointcloud_cam.ply (24,576 pts)   <- the SCENE (input)
            gt_rgba.png gt_depth.png  gt_pointcloud_cam.ply (8,192 pts) <- the target ALONE
            maize_XXXX_spline.xml     the TARGET's parameters only
            camera_pose.json          world frame = the main plant's: stem base at the
                                      origin, y up, row along x; fixed depth planes 0.05 / 3.0

Facts this loader relies on (measured on 20 train shots): both clouds are camera-view clouds
(99.8 % of gt points lie on gt_depth); neighbours hide 4-24 % of the target's points; the scene
cloud is sampled independently of the gt one (only 2-3 % of its points coincide), so there are
NO per-point plant ids -- a scene point is labelled TARGET when it lies within `label_tol` (1 cm)
of the target's own cloud, neighbour otherwise; an image pixel is NEIGHBOUR when the scene shows
a plant there that is not the target's surface (scene depth not within 1 cm of the target-alone
depth). Alpha in both RGBA images is a 0 / 255 foreground mask; RGB is composited on grey 127,
the background of every earlier maize render. Known hint, NOT removed: the rover's height is
0.55 x the main plant's height (r = 0.94) and the camera is level, so the image's centre row
gives away the plant's height.

A sample is one (plant, theta) shot. It returns the CLEAN TARGET in the usual slots, so every
metric and the clean validation pass work unchanged, and the scene as an 11th item:

    rgb, depth, pc        the target alone (rgb ImageNet-normalised, depth as decoded,
                          pc normalised by its own centroid / farthest point, as MaizeDataset4M)
    param_floats, text_valid, name
    pc_norm (4,)          that centroid (camera metres) and scale
    cam2world (4, 4)      near_far (2,)    vis (N,) all True (camera-view clouds already)
    scene                 {'rgb', 'depth': the scene images; 'pc': the scene cloud cropped to a
                           `crop_radius` cylinder around the stem, resampled to num_points, in the
                           TARGET's normalised frame; 'nb_point' (N,) bool; 'nb_px' (1, H, W) float
                           share of each pixel showing a neighbour; 'hidden' share of the target's
                           own pixels the neighbours cover}

occlusion_scene.compose(..., given=scene) turns that into the dict every trainer and eval
already consumes (same masking policies, frames, stats) -- see compose_given().
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.spatial import cKDTree
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from embodied_mae_4m_maize import MAX_LEAVES, load_spline_params

BG = 127                       # grey of every earlier maize rgb.png
INDEX_VERSION = 1


def _cache_path(load_dir):
    d = Path(os.environ.get('MAIZE_INDEX_CACHE') or Path(__file__).resolve().parent / '.dataset_index_cache')
    key = f"maize_row|{Path(load_dir).resolve()}|v{INDEX_VERSION}"
    return d / f"maize_row_{Path(load_dir).name}_{hashlib.sha256(key.encode()).hexdigest()[:16]}.json"


def _read_ply(path):
    with open(path, 'rb') as f:
        head = b''
        while not head.endswith(b'end_header\n'):
            line = f.readline()
            if not line:
                raise ValueError(f"no PLY header end in {path}")
            head += line
        n = int([l for l in head.split(b'\n') if l.startswith(b'element vertex')][0].split()[-1])
        if b'property double x' not in head:
            raise ValueError(f"expected double xyz in {path}")
        return np.frombuffer(f.read(n * 24), dtype='<f8').reshape(n, 3).astype(np.float64)


def _decode_depth(path):
    """Big-endian packed RGBA uint32 -> float32 [0, 1] (0 = background), full resolution."""
    with Image.open(path) as im:
        a = np.asarray(im).astype(np.float64)
    return ((a[..., 0] * 256.0 ** 3 + a[..., 1] * 256.0 ** 2 + a[..., 2] * 256.0 + a[..., 3])
            / float(256 ** 4 - 1)).astype(np.float32)


class MaizeRowDataset(Dataset):

    def __init__(self, data_root, img_size=224, num_points=8192, split=None, max_leaves=MAX_LEAVES,
                 view_sampling=True, view_seed=0, deterministic_view=False, crop_radius=0.45,
                 label_tol=0.01, scene_points=None, include=None, exclude=None, **unused):
        """split may join several with '+' ('train+val'); include / exclude are plant-name
        collections (a monitor set held out of training); scene_points: the scene cloud's size
        (None = num_points; 24576 = every point Alloy's renders carry -- after the crop, topped
        up by repeats). The TARGET cloud is always num_points (8,192 = all of it)."""
        if unused.get('max_plants') not in (None, 0):
            raise ValueError("MaizeRowDataset does not support max_plants")
        self.scene_points = int(scene_points) if scene_points else None
        splits = [x for x in str(split).split('+')] if split else [None]
        self.load_dirs = [Path(data_root) / x if x else Path(data_root) for x in splits]
        self.load_dir = self.load_dirs[0]
        self.img_size, self.num_points, self.max_leaves = int(img_size), int(num_points), int(max_leaves)
        self.view_sampling, self.deterministic_view = bool(view_sampling), bool(deterministic_view)
        self._view_seed, self._epoch = int(view_seed), 0
        self.crop_radius = None if crop_radius is None else float(crop_radius)
        self.label_tol = float(label_tol)
        self.rgb_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.plants = []
        for d in self.load_dirs:
            print(f"Loading maize rows from: {d}")
            self.plants += [dict(p, dir=str(d)) for p in self._index(d)]
        if include is not None:
            inc = set(include)
            self.plants = [p for p in self.plants if p['plant'] in inc]
        if exclude is not None:
            exc = set(exclude)
            self.plants = [p for p in self.plants if p['plant'] not in exc]
        self.samples = [(p['dir'], p['plant'], s, p['xml']) for p in self.plants for s in p['shots']]
        self.plant_views = []
        k = 0
        for p in self.plants:
            self.plant_views.append(list(range(k, k + len(p['shots'])))); k += len(p['shots'])
        print(f"✅ {len(self.plants)} plants, {len(self.samples)} shots in {'+'.join(d.name for d in self.load_dirs)}"
              + (f"  [one shot per plant per epoch{', deterministic' if self.deterministic_view else ''}]"
                 if self.view_sampling else ''))

    def _index(self, load_dir):
        cp = _cache_path(load_dir)
        mt = load_dir.stat().st_mtime_ns
        try:
            blob = json.loads(cp.read_text())
            if blob.get('mtime_ns') == mt:
                print(f"⚡ index cache hit ({len(blob['entries'])} plants)")
                return blob['entries']
        except Exception:
            pass
        out = []
        for name in sorted(os.listdir(load_dir)):
            pdir = load_dir / name
            try:
                row = json.loads((pdir / 'row.json').read_text())
            except Exception:
                continue
            shots = []
            for o in row.get('orientations', []):
                if o.get('failed'):
                    continue
                t = f"theta_{int(round(o['theta']['main'])):03d}"
                shots += [f"{t}/{s}" for s in sorted(o.get('shots', {}))]
            shots = [s for s in shots if (pdir / s / 'gt_pointcloud_cam.ply').exists()]
            if shots:
                out.append({'plant': name, 'shots': shots, 'xml': f"maize_{name.split('_')[1]}_spline.xml"})
        cp.parent.mkdir(parents=True, exist_ok=True)
        tmp = cp.with_suffix(f'.tmp{os.getpid()}')
        tmp.write_text(json.dumps({'mtime_ns': mt, 'entries': out}))
        os.replace(tmp, cp)
        print(f"🔍 indexed {len(out)} plants (cache written)")
        return out

    def set_epoch(self, epoch):
        self._epoch = int(epoch)

    def __len__(self):
        return len(self.plant_views) if self.view_sampling else len(self.samples)

    def _resolve_index(self, idx):
        if not self.view_sampling:
            return idx
        views = self.plant_views[idx]
        if self.deterministic_view:
            # a FIXED shot per plant that still varies across plants (theta 0 ... 90)
            return views[int(self.plants[idx]['plant'].split('_')[1]) % len(views)]
        rng = random.Random((self._view_seed * 1_000_003 + self._epoch) * 1_000_033 + idx)
        return views[rng.randrange(len(views))]

    def _rgb(self, path):
        with Image.open(path) as im:
            a = np.asarray(im.convert('RGBA')).astype(np.float32)
        al = a[..., 3:4] / 255.0
        rgb = (a[..., :3] * al + BG * (1 - al)).round().clip(0, 255).astype(np.uint8)
        return self.rgb_transform(Image.fromarray(rgb))

    def _depth(self, d):
        t = torch.from_numpy(np.ascontiguousarray(d))[None]
        return TF.resize(t, [self.img_size, self.img_size], interpolation=InterpolationMode.BILINEAR, antialias=True)

    def __getitem__(self, idx):
        pdir, plant, shot, xml = self.samples[self._resolve_index(idx)]
        sdir = Path(pdir) / plant / shot
        pose = json.loads((sdir / 'camera_pose.json').read_text())
        near, far = float(pose['near']), float(pose['far'])
        c2w = np.array(pose['cameraToWorld'], np.float64).reshape(4, 4)

        # target alone (the reconstruction targets)
        gt = _read_ply(sdir / 'gt_pointcloud_cam.ply')
        if len(gt) != self.num_points:
            gt = gt[np.random.choice(len(gt), self.num_points, replace=len(gt) < self.num_points)]
        cen = gt.mean(0)
        rad = float(np.linalg.norm(gt - cen, axis=1).max()) or 1.0
        pc = torch.from_numpy(((gt - cen) / rad).astype(np.float32))
        pc_norm = torch.tensor(np.append(cen, rad), dtype=torch.float32)
        d_gt, d_sc = _decode_depth(sdir / 'gt_depth.png'), _decode_depth(sdir / 'depth.png')
        rgb, depth = self._rgb(sdir / 'gt_rgba.png'), self._depth(d_gt)
        text_valid, param_floats = load_spline_params(sdir / xml, self.max_leaves)

        # the scene: cloud cropped around the stem, resampled, in the target's normalised frame
        sc = _read_ply(sdir / 'pointcloud_cam.ply')
        keep = np.ones(len(sc), bool)
        if self.crop_radius is not None:
            w = sc @ c2w[:3, :3].T + c2w[:3, 3]
            keep = np.hypot(w[:, 0], w[:, 2]) <= self.crop_radius
        sc = sc[keep] if keep.any() else sc
        n_sc = self.scene_points or self.num_points
        sel = np.random.choice(len(sc), n_sc, replace=len(sc) < n_sc)
        sc = sc[sel]
        nb_point = cKDTree(gt).query(sc, k=1)[0] > self.label_tol
        z_gt = np.where(d_gt > 0, near + d_gt * (far - near), np.inf)
        z_sc = np.where(d_sc > 0, near + d_sc * (far - near), np.inf)
        own = (d_gt > 0) & (np.abs(z_sc - z_gt) < self.label_tol)          # target surface showing
        nb_full = (d_sc > 0) & ~own
        hidden = float(((d_gt > 0) & ~own).sum() / max((d_gt > 0).sum(), 1))
        nb_px = TF.resize(torch.from_numpy(nb_full.astype(np.float32))[None], [self.img_size, self.img_size],
                          interpolation=InterpolationMode.BILINEAR, antialias=True)
        scene = {'rgb': self._rgb(sdir / 'rgba.png'), 'depth': self._depth(d_sc),
                 'pc': torch.from_numpy(((sc - cen) / rad).astype(np.float32)),
                 'nb_point': torch.from_numpy(nb_point), 'nb_px': nb_px.clamp(0, 1),
                 'hidden': torch.tensor(hidden)}
        return (rgb, depth, pc, param_floats, text_valid, f"{plant}_{shot.split('/')[0]}", pc_norm,
                torch.tensor(c2w, dtype=torch.float32), torch.tensor([near, far], dtype=torch.float32),
                torch.ones(self.num_points, dtype=torch.bool), scene)


def monitor_plants(data_root, n=200, split='val'):
    """A fixed, spread-out set of n plants of `split` (every k-th by sorted name): held OUT of a
    train+val run and used as its only validation, so a long run is not blind."""
    names = sorted(x for x in os.listdir(Path(data_root) / split) if x.startswith('plant_'))
    k = max(len(names) // int(n), 1)
    return names[::k][:int(n)]


def build_maize_dataset(cfg, split, **kw):
    """MaizeDataset4M, or MaizeRowDataset when the run's config says data.dataset: maize_row.
    `cfg` is a run's config.json dict (or vars(args)); kw override / add constructor arguments."""
    from maize_dataset_4m import MaizeDataset4M
    if cfg.get('dataset') == 'maize_row':
        crop = (cfg.get('occlusion_scene') or {}).get('crop_radius', 0.45)
        return MaizeRowDataset(cfg['data_root'], img_size=cfg['img_size'], num_points=cfg['num_points'], split=split,
                               max_leaves=cfg['max_leaves'], view_sampling=cfg['view_sampling'],
                               view_seed=cfg.get('view_seed', 0), crop_radius=crop,
                               scene_points=cfg.get('scene_points'),
                               **{k: v for k, v in kw.items() if k in ('deterministic_view', 'view_seed')})
    return MaizeDataset4M(cfg['data_root'], img_size=cfg['img_size'], num_points=cfg['num_points'], split=split,
                          max_leaves=cfg['max_leaves'], view_sampling=cfg['view_sampling'], **kw)
