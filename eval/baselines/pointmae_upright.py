"""pointmae_upright -- Point-MAE fed gravity-aligned clouds. A SENSITIVITY row, not a headline one.

Identical to pointmae (same checkpoint, strict load, FPS, pooling) except for
one step: before the FPS, each cloud is rotated from the camera frame into the
world frame with that view's camera_pose.json `cameraToWorld` rotation. Both
renderers' worlds are y-up: the sorghum camera sits at y = 2.94 above a plant
at the origin, and the maize `plantCenter` is at y ~ 0.6. That matches
ShapeNet's y-up canon, which Point-MAE was pretrained on with no rotation
augmentation.

WHY IT EXISTS. On maize, whose view-00 camera tilts 0-86 deg per plant, the
camera-frame row loses up to 0.49 val R2 to this one (the table is in
pointmae.py, step 3). Without this row, a reader cannot tell whether
Point-MAE's representation is weak, or whether it was just never trained to
ignore camera pose, which our arms were (random views every epoch).

WHY IT IS NOT THE HEADLINE. The rotation is the camera pose: per plant on
maize, information no E2/E3/E4 arm receives. On sorghum, view 00 is one fixed
camera for every plant, so there the rotation is a constant that pointmae
itself now applies, and the two rows coincide. Caption the row "Point-MAE +
camera pose (gravity-aligned input)". Compare it to an arm only if that arm is
also given upright clouds.

The rotation only turns the cloud: centring and max-norm are unchanged, so
pointmae's unit-sphere guard still holds. It needs the per-sample folder, so
it declares USES_CAMERA_POSE = True, the only adapter allowed to call
model.probe_ctx.sample_dir(name); the framework prints a caption warning at
every run of it. Its cache name differs from pointmae's, so the two rows can
never be served from each other's features.
"""

import json

import torch

from . import ctx
from . import pointmae as _pm

NAME = 'pointmae_upright'
INPUTS = ('pc',)
SOURCE = _pm.SOURCE.replace(
    'PC in camera frame, sorghum rotated y-up by the one fixed view-00 camera rotation',
    'PC gravity-aligned by each VIEW\'s camera_pose.json cameraToWorld rotation (camera '
    'pose the arms never see)')
assert SOURCE != _pm.SOURCE, 'pointmae.SOURCE wording changed: update the replace above'
MODEL_SIZE = _pm.MODEL_SIZE
EPOCH = _pm.EPOCH
FEATURES = _pm.FEATURES
FEATURE_LABELS = _pm.FEATURE_LABELS
USES_CAMERA_POSE = True        # unlocks ctx.sample_dir(); a captioned sensitivity row

build = _pm.build          # same weights, same strict load, same report


def camera_to_world_rotation(sample_dir):
    """The 3x3 rotation of camera_pose.json's row-major 4x4 cameraToWorld."""
    with open(sample_dir / 'camera_pose.json') as f:
        m = json.load(f)['cameraToWorld']
    R = torch.tensor(m, dtype=torch.float64).reshape(4, 4)[:3, :3]
    # A scale or shear here would silently break the unit-sphere convention.
    err = (R @ R.T - torch.eye(3, dtype=torch.float64)).abs().max().item()
    if err > 1e-4 or torch.det(R).item() < 0:
        raise ValueError(f'{sample_dir.name}: cameraToWorld is not a rotation '
                         f'(|RR^T - I| {err:.2e}, det {torch.det(R).item():.4f})')
    return R.float()


def features(model, batch):
    _rgb, _depth, pc, _params, _text_valid, names = batch
    c = ctx(model)
    R = torch.stack([camera_to_world_rotation(c.sample_dir(n)) for n in names])
    upright = torch.einsum('bij,bnj->bni', R.to(pc.device), pc.float())
    # Straight to tokens: pointmae.features would add its own fixed sorghum
    # rotation on top of this one.
    return _pm.pool(model.tokens(upright), c.feature)
