#!/usr/bin/env python3
"""Maize 4M pretraining on LEVELLED point clouds: the pretrained arm of the pose question.

Identical to train_maize_4m.py in every respect but one. Each point cloud, after
the loader has centred it and scaled it onto the unit sphere, is rotated by its
view's camera_pose.json into a level frame at the camera's own azimuth:
cameraToWorld, then the yaw about world +Y that puts the camera back at azimuth
0. Up is world +Y; the plant's azimuth is whatever that camera saw. RGB and depth
stay as rendered.

WHY. On maize, Point-MAE given the full cameraToWorld rotation beats our frozen
arms on 8 of 11 targets, but that rotation also hands over the renderer's
canonical plant azimuth (every plant's leaf plane is the same world plane), which
no rig knows. The levelled frame keeps only what a calibrated rig knows: the up
axis. Our frozen arms, trained on camera-frame clouds, cannot use a levelled cloud
after the fact (the eval/baselines *_gravity rows lose with it), so whether our
model can use gravity needs an arm PRETRAINED on it. This is that arm.

PROBE IT through the E8 harness row of the same name (eval/baselines/
maize_e2_pcrgbd_levelled.py), which applies this rotation to the probe's
camera-frame batch. eval/linear_probe_maize.py would feed it camera-frame clouds.

HOW, without editing the files the running maize distillation re-imports at
every requeue (maize_dataset_4m.py, and train_maize_4m.py beyond its pc_frame
guard): it installs a MaizeDataset4M subclass, and a config_to_namespace that
records pc_frame, into train_maize_4m, then calls its main(). Both are installed
at import, so torchrun ranks and mp.spawn children get them. The rotation draws
no random numbers, so the view drawn and the point permutation for every item
are the camera-frame arm's: the two arms differ in the cloud's frame only.

    torchrun --standalone --nproc_per_node=2 train_maize_4m_gravity.py \\
        --config configs/config_maize_e2_pcrgbd_levelled.yaml
"""
import json
import math
from pathlib import Path

import torch

import train_maize_4m as T
from maize_dataset_4m import MaizeDataset4M

PC_FRAME = 'gravity'


def camera_to_gravity_rotation(sample_dir):
    """3x3, camera frame -> level frame at the camera's azimuth (world +Y up).

    The rotation of eval/baselines/_gravity.camera_to_gravity_rotation, duplicated
    so that training code never imports eval/ (the two are checked equal on real
    poses before launch).
    """
    with open(Path(sample_dir) / 'camera_pose.json') as f:
        m = json.load(f)
    R = torch.tensor(m['cameraToWorld'], dtype=torch.float64).reshape(4, 4)[:3, :3]
    err = (R @ R.T - torch.eye(3, dtype=torch.float64)).abs().max().item()
    if err > 1e-4 or torch.det(R).item() < 0:
        raise ValueError(f'{Path(sample_dir).name}: cameraToWorld is not a rotation '
                         f'(|RR^T - I| {err:.2e}, det {torch.det(R).item():.4f})')
    rel = [p - c for p, c in zip(m['position'], m['plantCenter'])]
    th = math.atan2(rel[0], rel[2])          # the camera's azimuth about world +Y
    c, s = math.cos(th), math.sin(th)
    yaw = torch.tensor([[c, 0.0, -s],
                        [0.0, 1.0, 0.0],
                        [s, 0.0, c]], dtype=torch.float64)
    return (yaw @ R).float()


class MaizeDataset4MGravity(MaizeDataset4M):
    """MaizeDataset4M with every cloud levelled; everything else unchanged."""

    def __getitem__(self, idx):
        rgb, depth, pc, params, text_valid, name = super().__getitem__(idx)
        R = camera_to_gravity_rotation(self.load_dir / name)
        return rgb, depth, pc @ R.T, params, text_valid, name


_camera_namespace = T.config_to_namespace


def _gravity_namespace(config):
    ns = _camera_namespace(config)
    ns.pc_frame = PC_FRAME           # recorded in config.json via vars(args)
    prev = Path(ns.output_dir) / 'config.json'
    if prev.exists():
        was = json.loads(prev.read_text()).get('pc_frame', 'camera')
        if was != PC_FRAME:
            raise SystemExit(f'{prev} records pc_frame {was!r}: refusing to resume a '
                             f'{was}-frame run on {PC_FRAME}-frame clouds')
    return ns


T.PC_FRAME = PC_FRAME
T.MaizeDataset4M = MaizeDataset4MGravity
T.config_to_namespace = _gravity_namespace


if __name__ == '__main__':
    T.main()
