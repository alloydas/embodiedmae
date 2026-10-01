"""maize_4m_upright -- maize_4m at epoch 600, frozen, fed gravity-aligned clouds.

A SENSITIVITY row, the like-for-like partner of pointmae_upright. See
_arm_upright.py for what it tests and what it does not. Caption it
"maize_4m + camera pose (gravity-aligned input)", never as a plain arm row.
"""

from . import _arm_upright as _A

NAME = 'maize_4m_upright'
RUN = 'maize_4m'
INPUTS = ('rgb', 'depth', 'pc')
SOURCE = _A.source(RUN)
MODEL_SIZE = _A.MODEL_SIZE
EPOCH = _A.EPOCH
FEATURES = _A.FEATURES
USES_CAMERA_POSE = _A.USES_CAMERA_POSE
CODE_DEPS = _A.CODE_DEPS


def build(device, cache_dir):
    """`cache_dir` is unused: the weights are this repo's own checkpoint."""
    return _A.build(RUN, device, INPUTS)


features = _A.features
