"""Shared body of the *_upright arm rows: one of OUR trained maize arms, frozen,
fed gravity-aligned point clouds. A SENSITIVITY row, like pointmae_upright.

WHY IT EXISTS. On maize, Point-MAE given each view's camera pose
(pointmae_upright) beats maize_4m on 8 of 11 targets. Its rotation is camera
pose that no arm ever received, so those numbers compare unlike inputs. These
rows give our arms the same rotation and nothing else: same checkpoint,
strict load, feature step and plants. That makes them the like-for-like
partner of pointmae_upright.

WHAT IT DOES NOT TEST. Each arm was pretrained on camera-frame clouds, so an
upright cloud is out of its training distribution, and in the multimodal arms it
no longer lines up with the camera-frame RGB and depth that the model learned to
fuse it with. The probe is refitted on the rotated features. A gain here means
the frozen encoder can use orientation even so; no gain says nothing about an
arm PRETRAINED on upright clouds, which is a different run.

Model and feature step: linear_probe_maize.build_model on the run's own
config.json (the probe's builder, strict load, eval mode), then
random_init.features: forward_encoder_select on the non-text modalities with the
params zeroed and source_mask_ratio 0.0, CLS or mean of the non-CLS tokens.
That is linear_probe's own step; --check-parity tests that path. The only
change is the rotation, taken from pointmae_upright.camera_to_world_rotation,
which rejects anything that is not a proper rotation. The loader has already
centred and unit-scaled the cloud, and a rotation keeps both.
"""

from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
CODE_DEPS = ('embodied_mae.py', 'embodied_mae_4m.py', 'embodied_mae_4m_maize.py')
FEATURES = ('cls', 'mean')
INPUTS = ('rgb', 'depth', 'pc')
USES_CAMERA_POSE = True
MODEL_SIZE = 'base'
EPOCH = 600
CKPT = 'checkpoints/checkpoint_epoch_600.pth'


def source(run):
    return (f'this repo: outputs/{run}/{CKPT}, strict load via linear_probe_maize.build_model; '
            f'PC gravity-aligned by each VIEW\'s camera_pose.json cameraToWorld rotation '
            f'(camera pose the arm never saw in training)')


def build(run, device, inputs):
    import linear_probe_maize as P
    model, cfg, epoch = P.build_model(REPO / 'outputs' / run, CKPT, device)
    if epoch != EPOCH:
        raise RuntimeError(f'{run}: {CKPT} is epoch {epoch}, expected {EPOCH}')
    # forward_encoder_select only ever sees the arm's non-text modalities
    # (the framework passes None in every slot not in the row's INPUTS)
    want = {m for m in model.active_modalities if m != 'text'}
    if want != set(inputs):
        raise RuntimeError(f'{run}: active {model.active_modalities} does not match INPUTS {inputs}')
    print(f'  {run}: epoch {epoch}, active {model.active_modalities}, PC rotated upright per view')
    return model


def features(model, batch):
    from . import ctx
    from . import random_init as _ri
    from .pointmae_upright import camera_to_world_rotation
    c = ctx(model)
    if c.species != 'maize':
        raise RuntimeError('the *_upright arm rows are maize-only: every sorghum view 00 '
                           'is one fixed camera, so there is nothing to align')
    rgb, depth, pc, params, text_valid, names = batch
    R = torch.stack([camera_to_world_rotation(c.sample_dir(n)) for n in names])
    upright = torch.einsum('bij,bnj->bni', R.to(pc.device), pc.float())
    return _ri.features(model, (rgb, depth, upright, params, text_valid, names))
