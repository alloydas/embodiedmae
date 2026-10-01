"""The rotation of the *_gravity rows: level each camera, keep its azimuth.

WHY NOT cameraToWorld. In the maize renderer's world frame every plant has the
SAME azimuth: maize is distichous (horizontal anisotropy 0.97) and its leaf plane
is the world x-y plane for every plant (axial resultant 0.993 over 194 val plants,
view 00; 1 = identical, 0 = uniform). The generator places leaf i at azimuth
180 i + U(-15, 15) with no per-plant yaw. So cameraToWorld, which is what
pointmae_upright and the *_upright arm rows apply, hands a model gravity AND a
canonical plant orientation. A calibrated rig knows the first and never the
second.

This rotation keeps only what a rig knows: cameraToWorld, then the yaw about
world +Y that puts the camera back at azimuth 0, i.e. the frame of a LEVEL camera
at the camera's own azimuth. Up is world +Y; the plant's azimuth is whatever the
camera saw. Over the same 194 plants the leaf-plane resultant drops to 0.114.
The view-00 camera azimuths span the full circle and its elevations -86 to +86
deg, so the camera-frame input is tilted arbitrarily and this one is not.
"""

import json
import math

import torch

from .pointmae_upright import camera_to_world_rotation


def camera_to_gravity_rotation(sample_dir):
    """3x3: camera frame -> a level frame at the camera's azimuth (world +Y up)."""
    R = camera_to_world_rotation(sample_dir).double()   # rejects a non-rotation
    with open(sample_dir / 'camera_pose.json') as f:
        m = json.load(f)
    rel = [p - c for p, c in zip(m['position'], m['plantCenter'])]
    th = math.atan2(rel[0], rel[2])          # the camera's azimuth about world +Y
    c, s = math.cos(th), math.sin(th)
    yaw = torch.tensor([[c, 0.0, -s],        # rotation by -th about +Y: the camera
                        [0.0, 1.0, 0.0],     # lands on +z, up stays +Y
                        [s, 0.0, c]], dtype=torch.float64)
    return (yaw @ R).float()
