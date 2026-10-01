"""maize_4m_gravity -- maize_4m at epoch 600, frozen, fed levelled clouds (four-modality arm).

A SENSITIVITY row, maize only: the like-for-like partner of pointmae_gravity. See
_arm_gravity.py for what it tests and what it does not. Caption it
"maize_4m + gravity (up axis known)", never as a plain arm row.
"""

from . import _arm_gravity as _G

NAME = 'maize_4m_gravity'
RUN = 'maize_4m'
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
