"""pointmae_gravity -- Point-MAE fed levelled clouds, the plant's azimuth left as seen.

A SENSITIVITY row, maize only. Identical to pointmae_upright except for the
rotation: _gravity.camera_to_gravity_rotation, not the full cameraToWorld.
pointmae_upright's input carries the maize renderer's canonical plant azimuth
(every plant's leaf plane is the same world plane, see _gravity.py); this one
carries only the up axis, which a calibrated rig would know. The gap between
the two rows is what the canonical azimuth is worth, and this row, not
pointmae_upright, is the realistic "Point-MAE + camera pose" baseline.

Caption it "Point-MAE + gravity (up axis known)". Compare it to an arm only if
that arm is given the same levelled clouds (the *_gravity arm rows).
"""

import torch

from . import ctx
from . import pointmae as _pm
from ._gravity import camera_to_gravity_rotation

NAME = 'pointmae_gravity'
INPUTS = ('pc',)
SOURCE = _pm.SOURCE.replace(
    'PC in camera frame, sorghum rotated y-up by the one fixed view-00 camera rotation',
    'PC levelled by each VIEW\'s camera_pose.json: cameraToWorld, then the yaw that '
    'restores the camera\'s azimuth (up axis known, plant azimuth not)')
assert SOURCE != _pm.SOURCE, 'pointmae.SOURCE wording changed: update the replace above'
MODEL_SIZE = _pm.MODEL_SIZE
EPOCH = _pm.EPOCH
FEATURES = _pm.FEATURES
FEATURE_LABELS = _pm.FEATURE_LABELS
USES_CAMERA_POSE = True        # unlocks ctx.sample_dir(); a captioned sensitivity row

build = _pm.build          # same weights, same strict load, same report


def features(model, batch):
    _rgb, _depth, pc, _params, _text_valid, names = batch
    c = ctx(model)
    if c.species != 'maize':
        raise RuntimeError('pointmae_gravity is maize-only: _gravity.py reads the maize '
                           'camera_pose.json plantCenter')
    R = torch.stack([camera_to_gravity_rotation(c.sample_dir(n)) for n in names])
    level = torch.einsum('bij,bnj->bni', R.to(pc.device), pc.float())
    # Straight to tokens, as pointmae_upright: pointmae.features would add its
    # own fixed sorghum rotation on top of this one.
    return _pm.pool(model.tokens(level), c.feature)
