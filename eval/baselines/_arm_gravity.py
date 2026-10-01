"""Shared body of the *_gravity arm rows: our frozen maize arms fed levelled clouds.

The like-for-like partners of pointmae_gravity, as the *_upright rows are of
pointmae_upright. Same checkpoint, strict load, feature step and plants as
_arm_upright.py (its build is reused unchanged). The only difference is the
rotation: _gravity.camera_to_gravity_rotation (up axis known, plant azimuth as
the camera saw it) instead of the full cameraToWorld (which also hands over the
renderer's canonical plant azimuth).

What it does not test is what _arm_upright.py says: each arm was pretrained on
camera-frame clouds, so a levelled cloud is off its training distribution and no
longer lines up with the camera-frame RGB and depth. No gain says nothing about
an arm PRETRAINED on levelled clouds.
"""

import torch

from . import _arm_upright as _A

CODE_DEPS = _A.CODE_DEPS
FEATURES = _A.FEATURES
USES_CAMERA_POSE = True
MODEL_SIZE = _A.MODEL_SIZE
EPOCH = _A.EPOCH
CKPT = _A.CKPT


def source(run):
    return (f'this repo: outputs/{run}/{CKPT}, strict load via linear_probe_maize.build_model; '
            f'PC levelled by each VIEW\'s camera_pose.json: cameraToWorld, then the yaw that '
            f'restores the camera\'s azimuth (camera pose the arm never saw in training)')


def build(run, device, inputs):
    """_arm_upright.build, copied only so its log line is true for these rows.
    (Editing _arm_upright.py would still re-key these caches, since this module
    imports it, but this file is outside the upright rows' fingerprint.)"""
    import linear_probe_maize as P
    model, cfg, epoch = P.build_model(_A.REPO / 'outputs' / run, CKPT, device)
    if epoch != EPOCH:
        raise RuntimeError(f'{run}: {CKPT} is epoch {epoch}, expected {EPOCH}')
    want = {m for m in model.active_modalities if m != 'text'}
    if want != set(inputs):
        raise RuntimeError(f'{run}: active {model.active_modalities} does not match INPUTS {inputs}')
    print(f'  {run}: epoch {epoch}, active {model.active_modalities}, PC levelled per view '
          f'(up axis known, plant azimuth as the camera saw it)')
    return model


def features(model, batch):
    from . import ctx
    from . import random_init as _ri
    from ._gravity import camera_to_gravity_rotation
    c = ctx(model)
    if c.species != 'maize':
        raise RuntimeError('the *_gravity arm rows are maize-only: _gravity.py reads the '
                           'maize camera_pose.json plantCenter')
    rgb, depth, pc, params, text_valid, names = batch
    R = torch.stack([camera_to_gravity_rotation(c.sample_dir(n)) for n in names])
    level = torch.einsum('bij,bnj->bni', R.to(pc.device), pc.float())
    return _ri.features(model, (rgb, depth, level, params, text_valid, names))
