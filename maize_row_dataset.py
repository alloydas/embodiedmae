#!/usr/bin/env python3
"""Maize ROW data (Regime III, neighbour occlusion): a rover camera looking at a planted
row, the target plant between two bent neighbours.

Layout written by the row renderer (/work/mech-ai/alloydas/maize_row, 2026-10-16):

    <root>/<split>/plant_<id>/row.json
    <root>/<split>/plant_<id>/theta_<deg>/shot_<k>/
        rgba.png              row render: target + both neighbours (RGBA, alpha = any plant)
        depth.png             row depth, big-endian packed RGBA uint32 of (z-near)/(far-near)
        pointcloud_cam.ply    the SCENE cloud the rover sees, 24,576 points (camera frame)
        gt_rgba.png           the target plant ALONE under the same camera
        gt_depth.png          its depth alone
        gt_pointcloud_cam.ply the target plant's own cloud, 8,192 points (camera frame)
        maize_<id>_spline.xml the target's recipe (schema v2 + the post-bend bendAngleDeg /
                              bladeLength, which the v2 parser ignores)
        camera_pose.json      intrinsics, pose, near/far, the row record

theta_<deg> is the row's yaw against the rover (0 = the rover looks along the row's
normal, 90 = along the row), ten values 0..90; shot_<k> is a rover position (one so far).
`row.json` records the neighbours, the in-row spacing d, the rover and, per orientation,
the leaf-bending solve and the pixel counts (`plantPixels`: all plants in the render,
`mainAlonePixels`: the target alone).

An item is one (plant, theta, shot). Two input modes share the same targets:
  mode='row'      : the encoder sees the row render (rgba / depth / scene cloud)
  mode='isolated' : the encoder sees the gt_* files (the target alone, same camera),
                    i.e. the paired occlusion-free control.
Targets are always the target plant's own streams (gt rgb / gt depth / gt cloud) and its
recipe: a model trained in 'row' mode learns to reconstruct the plant BEHIND the
neighbours, which is the Regime III task.

Cloud frame. The input cloud is centred on ITS OWN centroid and scaled onto the unit
sphere (the maize loader's convention), and the gt cloud is moved by the SAME centroid and
scale, so prediction and target live in one frame and the Chamfer between them means
what it means for the isolated runs. In 'row' mode that frame is the scene's (the target
fills roughly half of it), which is part of the shift the model has to absorb.

Occlusion measures per item (computed from the two depth images, cheap at 256 px):
  occ_frac   : fraction of the target's own pixels hidden behind a neighbour
               (gt alpha > 0 and the scene depth is nearer than the gt depth)
  nbr_frac   : fraction of plant pixels in the row render that are not the target
`persistent_workers` must stay OFF for the same reason as every other loader here.
"""
from __future__ import annotations
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from embodied_mae_4m_maize import load_spline_params, MAX_LEAVES

MODES = ('row', 'isolated')
IN_FILES = {'row': ('rgba.png', 'depth.png', 'pointcloud_cam.ply'),
            'isolated': ('gt_rgba.png', 'gt_depth.png', 'gt_pointcloud_cam.ply')}
GT_FILES = ('gt_rgba.png', 'gt_depth.png', 'gt_pointcloud_cam.ply')


def read_ply_xyz(path):
    """binary_little_endian PLY with double x y z (Open3D), no normals."""
    with open(path, 'rb') as f:
        header = b''
        while not header.endswith(b'end_header\n'):
            line = f.readline()
            if not line:
                raise ValueError(f'{path}: no end_header')
            header += line
        n = int([l for l in header.decode().split('\n') if l.startswith('element vertex')][0].split()[-1])
        props = [l.split()[1:] for l in header.decode().split('\n') if l.startswith('property')]
        if 'binary_little_endian' not in header.decode():
            raise ValueError(f'{path}: not binary_little_endian')
        dt = np.dtype([(name, {'double': '<f8', 'float': '<f4'}[t]) for t, name in props])
        arr = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)
    return np.stack([arr['x'], arr['y'], arr['z']], axis=1).astype(np.float64)


def decode_depth(arr):
    """Big-endian packed RGBA uint32 -> float32 in [0, 1] (0 = background)."""
    if arr.ndim == 3 and arr.shape[2] == 4 and arr.dtype == np.uint8:
        a = arr.astype(np.float32)
        return (a[..., 0] * 256.0 ** 3 + a[..., 1] * 256.0 ** 2 + a[..., 2] * 256.0 + a[..., 3]) / float(256 ** 4 - 1)
    if arr.ndim == 2:
        return arr.astype(np.float32) / (np.iinfo(arr.dtype).max if np.issubdtype(arr.dtype, np.integer) else 1.0)
    raise ValueError(f'unsupported depth image shape {arr.shape} dtype {arr.dtype}')


class MaizeRowDataset(Dataset):
    def __init__(self, root, split='val', mode='row', img_size=224, num_points=8192,
                 max_leaves=MAX_LEAVES, thetas=None, shots=None,
                 sample_per_plant=False, seed=0, deterministic=False, plants=None,
                 occlusion_stats=True):
        """
        root/split      : the row data root and split folder
        mode            : 'row' (occluded inputs) or 'isolated' (gt_* inputs)
        thetas / shots  : restrict the orientations / rover positions (ints; None = all)
        sample_per_plant: an epoch is ONE item per plant (view-sampling style, seeded by
                          (seed, epoch, plant index) as in the sorghum loader); False
                          enumerates every item, which is what evaluation wants
        deterministic   : with sample_per_plant, always the first item of each plant
        plants          : optional iterable of plant folder names to keep
        """
        if mode not in MODES:
            raise ValueError(f'mode must be one of {MODES}, got {mode!r}')
        self.root, self.split, self.mode = Path(root), split, mode
        self.img_size, self.num_points, self.max_leaves = int(img_size), int(num_points), int(max_leaves)
        self.occlusion_stats = bool(occlusion_stats)
        split_dir = self.root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(split_dir)
        keep_plants = None if plants is None else set(plants)
        keep_t = None if thetas is None else {int(t) for t in thetas}
        keep_s = None if shots is None else {int(s) for s in shots}

        self.items = []            # (plant, theta, shot, folder)
        self.rows = {}             # plant -> row.json
        for pdir in sorted(split_dir.glob('plant_*')):
            if keep_plants is not None and pdir.name not in keep_plants:
                continue
            rj = pdir / 'row.json'
            try:
                row = json.loads(rj.read_text())
            except (OSError, json.JSONDecodeError):
                continue                      # still being written / uploaded
            n_before = len(self.items)
            for tdir in sorted(pdir.glob('theta_*')):
                theta = int(tdir.name.split('_')[1])
                if keep_t is not None and theta not in keep_t:
                    continue
                for sdir in sorted(tdir.glob('shot_*')):
                    shot = int(sdir.name.split('_')[1])
                    if keep_s is not None and shot not in keep_s:
                        continue
                    need = set(IN_FILES[mode]) | set(GT_FILES) | {'camera_pose.json'}
                    if not all((sdir / f).exists() for f in need) or not list(sdir.glob('maize_*_spline.xml')):
                        continue              # partial upload
                    self.items.append((pdir.name, theta, shot, sdir))
            if len(self.items) > n_before:
                self.rows[pdir.name] = row
        if not self.items:
            raise ValueError(f'no complete items under {split_dir}')
        self.plant_ids = sorted({it[0] for it in self.items})
        self._by_plant = {p: [i for i, it in enumerate(self.items) if it[0] == p] for p in self.plant_ids}
        self.sample_per_plant, self.deterministic = bool(sample_per_plant), bool(deterministic)
        self._seed, self._epoch = int(seed), 0

        self.rgb_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        n_t = sorted({it[1] for it in self.items})
        print(f'🌽 maize row [{split}/{mode}]: {len(self.plant_ids)} plants, {len(self.items)} items, '
              f'thetas {n_t}' + ('  [one item per plant per epoch]' if self.sample_per_plant else ''))

    # ── sampling ──────────────────────────────────────────────────────────────
    def set_epoch(self, epoch):
        self._epoch = int(epoch)

    def __len__(self):
        return len(self.plant_ids) if self.sample_per_plant else len(self.items)

    def _resolve(self, idx):
        if not self.sample_per_plant:
            return idx
        cands = self._by_plant[self.plant_ids[idx]]
        if self.deterministic:
            return cands[0]
        rng = random.Random((self._seed * 1_000_003 + self._epoch) * 1_000_033 + idx)
        return cands[rng.randrange(len(cands))]

    # ── readers ───────────────────────────────────────────────────────────────
    def load_rgb(self, path):
        with Image.open(path) as im:
            if im.mode == 'RGBA':
                im = Image.alpha_composite(Image.new('RGBA', im.size, (0, 0, 0, 255)), im)
            return self.rgb_transform(im.convert('RGB'))

    def load_depth(self, path):
        with Image.open(path) as im:
            d = decode_depth(np.asarray(im))
        t = torch.from_numpy(np.ascontiguousarray(d)).unsqueeze(0)
        return TF.resize(t, [self.img_size, self.img_size], interpolation=InterpolationMode.BILINEAR, antialias=True)

    def _subsample(self, pts):
        n = pts.shape[0]
        if n >= self.num_points:
            return pts[np.random.choice(n, self.num_points, replace=False)]
        return np.vstack([pts, pts[np.random.choice(n, self.num_points - n, replace=True)]])

    def load_clouds(self, in_ply, gt_ply):
        """(input cloud, gt cloud) in the INPUT cloud's normalised frame."""
        pin = self._subsample(read_ply_xyz(in_ply))
        centre = pin.mean(axis=0)
        pin = pin - centre
        scale = float(np.max(np.linalg.norm(pin, axis=1))) or 1.0
        pin = pin / scale
        pgt = (self._subsample(read_ply_xyz(gt_ply)) - centre) / scale
        return pin.astype(np.float32), pgt.astype(np.float32), centre.astype(np.float32), np.float32(scale)

    @staticmethod
    def occlusion(folder, size=256):
        """occ_frac (target pixels hidden by a neighbour), nbr_frac (non-target plant pixels)."""
        def load(name):
            with Image.open(folder / name) as im:
                im = im.resize((size, size), Image.NEAREST)
                a = np.asarray(im)
            return a
        scene_rgba, gt_rgba = load('rgba.png'), load('gt_rgba.png')
        scene_d, gt_d = decode_depth(load('depth.png')), decode_depth(load('gt_depth.png'))
        tgt = gt_rgba[..., 3] > 0
        plant = scene_rgba[..., 3] > 0
        nearer = (scene_d > 0) & (scene_d < gt_d - 2e-3)        # something in front of the target
        occ = float((tgt & nearer).sum() / max(1, tgt.sum()))
        nbr = float((plant & ~tgt).sum() / max(1, plant.sum()))
        return occ, nbr

    def meta(self, i):
        plant, theta, shot, folder = self.items[i]
        row = self.rows.get(plant, {})
        ori = next((o for o in row.get('orientations', []) if abs(float(o['theta']['main']) - theta) < 1e-6), {})
        px = ori.get('shots', {}).get(f'shot_{shot:03d}', {})
        return {'plant': plant, 'theta': theta, 'shot': shot, 'folder': str(folder),
                'd': row.get('d'), 'bend_stage': ori.get('bendStage'),
                'plant_pixels': px.get('plantPixels'), 'main_alone_pixels': px.get('mainAlonePixels')}

    def __getitem__(self, idx):
        i = self._resolve(idx)
        plant, theta, shot, folder = self.items[i]
        f_rgb, f_depth, f_ply = IN_FILES[self.mode]
        rgb = self.load_rgb(folder / f_rgb)
        depth = self.load_depth(folder / f_depth)
        gt_rgb = self.load_rgb(folder / 'gt_rgba.png')
        gt_depth = self.load_depth(folder / 'gt_depth.png')
        pc, gt_pc, centre, scale = self.load_clouds(folder / f_ply, folder / 'gt_pointcloud_cam.ply')
        xml = next(folder.glob('maize_*_spline.xml'))
        valid, params = load_spline_params(xml, self.max_leaves, schema='v2')
        occ, nbr = self.occlusion(folder) if self.occlusion_stats else (float('nan'), float('nan'))
        info = {'plant': plant, 'theta': theta, 'shot': shot, 'idx': i,
                'occ_frac': occ, 'nbr_frac': nbr, 'centre': centre, 'scale': scale}
        return (rgb, depth, torch.from_numpy(pc), params, valid,
                gt_rgb, gt_depth, torch.from_numpy(gt_pc), info)


def collate(batch):
    """Default collate for everything but the info dict, which becomes a list."""
    cols = list(zip(*batch))
    out = [torch.stack(c) for c in cols[:-1]]
    out.append(list(cols[-1]))
    return out


if __name__ == '__main__':
    import argparse, time
    ap = argparse.ArgumentParser()
    ap.add_argument('root', nargs='?', default='/work/mech-ai/alloydas/maize_row')
    ap.add_argument('--split', default='val')
    ap.add_argument('--mode', default='row', choices=MODES)
    ap.add_argument('--n', type=int, default=4)
    a = ap.parse_args()
    ds = MaizeRowDataset(a.root, a.split, a.mode)
    t0 = time.time()
    for k in range(min(a.n, len(ds))):
        rgb, depth, pc, params, valid, g_rgb, g_depth, g_pc, info = ds[k]
        print(f"{info['plant']} theta {info['theta']:3d}: rgb {tuple(rgb.shape)} depth {tuple(depth.shape)} "
              f"pc {tuple(pc.shape)} gt_pc {tuple(g_pc.shape)} leaves {int(valid.sum()) - 1} "
              f"occ {info['occ_frac']:.2f} nbr {info['nbr_frac']:.2f} "
              f"|gt| max {float(g_pc.norm(dim=1).max()):.2f} (input cloud = unit sphere)")
    print(f'{(time.time() - t0) / max(1, min(a.n, len(ds))) * 1e3:.0f} ms / item')
