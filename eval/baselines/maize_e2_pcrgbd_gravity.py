"""maize_e2_pcrgbd_gravity -- maize_e2_pcrgbd at epoch 600, frozen, fed levelled clouds (PC+RGB+depth arm).

A SENSITIVITY row, maize only: the like-for-like partner of pointmae_gravity. See
_arm_gravity.py for what it tests and what it does not. Caption it
"maize_e2_pcrgbd + gravity (up axis known)", never as a plain arm row.
"""

from . import _arm_gravity as _G

NAME = 'maize_e2_pcrgbd_gravity'
RUN = 'maize_e2_pcrgbd'
INPUTS = ('rgb', 'depth', 'pc')
SOURCE = _G.source(RUN)
MODEL_SIZE = _G.MODEL_SIZE
EPOCH = _G.EPOCH
FEATURES = _G.FEATURES
USES_CAMERA_POSE = _G.USES_CAMERA_POSE
CODE_DEPS = _G.CODE_DEPS


def build(device, cache_dir):
    """`cache_dir` is unused: the weights are this repo's own checkpoint."""
    return _G.build(RUN, device, INPUTS)


features = _G.features
