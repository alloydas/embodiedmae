"""maize_e2_pcrgbd_levelled -- the PC+RGB+depth arm PRETRAINED on levelled clouds, probed on them.

Not a sensitivity row in the sense of the *_gravity rows: this arm was trained
by train_maize_4m_gravity.py on clouds levelled exactly as _arm_gravity.features
levels them here, so the probe hands it its own training distribution. It is the
like-for-like partner of pointmae_gravity, and its camera-frame twin is
maize_e2_pcrgbd (same seeds, views and point permutations; only the frame
differs). Caption it "PC + RGB + depth, pretrained with gravity (up axis known)".
"""

import json

from . import _arm_gravity as _G

NAME = 'maize_e2_pcrgbd_levelled'
RUN = 'maize_e2_pcrgbd_levelled'
INPUTS = ('rgb', 'depth', 'pc')
SOURCE = (f'this repo: outputs/{RUN}/{_G.CKPT}, trained on levelled clouds '
          f'(train_maize_4m_gravity.py), strict load via linear_probe_maize.build_model; '
          f'PC levelled per view as in training')
MODEL_SIZE = _G.MODEL_SIZE
EPOCH = _G.EPOCH
FEATURES = _G.FEATURES
USES_CAMERA_POSE = True        # the levelling reads each view's camera_pose.json
CODE_DEPS = _G.CODE_DEPS


def build(device, cache_dir):
    """`cache_dir` is unused: the weights are this repo's own checkpoint."""
    cfg = json.loads((_G._A.REPO / 'outputs' / RUN / 'config.json').read_text())
    if cfg.get('pc_frame') != 'gravity':
        raise RuntimeError(f'{RUN}: config.json records pc_frame {cfg.get("pc_frame")!r}, '
                           f'expected gravity (train_maize_4m_gravity.py)')
    return _G.build(RUN, device, INPUTS)


features = _G.features
