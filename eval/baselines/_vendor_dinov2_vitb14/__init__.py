"""DINOv2 ViT-B/14 model code, vendored from facebookresearch/dinov2@7764ea0f.

Apache-2.0: see LICENSE here (upstream's, verbatim) and the header of
vision_transformer.py for the exact upstream files, their sha256, and what was
removed. Not a baseline itself: eval/baselines/dinov2_vitb14.py is the adapter.
"""

from .vision_transformer import DinoVisionTransformer, vit_base_14

UPSTREAM_REPO = 'https://github.com/facebookresearch/dinov2'
UPSTREAM_COMMIT = '7764ea0f912e53c92e82eb78a2a1631e92725fc8'

__all__ = ['DinoVisionTransformer', 'vit_base_14', 'UPSTREAM_REPO', 'UPSTREAM_COMMIT']
