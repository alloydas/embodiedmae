"""Vendored MultiMAE encoder (RGB + depth), pure torch.

Upstream: https://github.com/EPFL-VILAB/MultiMAE at commit
66910f5b5ba236f5e731883db85fe4f24ee01106 (2022-12-13), (c) EPFL VILAB.
Licensed CC BY-NC 4.0: see LICENSE in this directory, copied verbatim from
that commit. The port in multivit.py is an adaptation of upstream code, and the
same licence covers it (attribution required, non-commercial use only).

Why vendor and not import. The upstream package cannot be imported here, for
three reasons:
  * `multimae/multimae.py:28` imports `utils.registry`, and upstream
    `utils/__init__.py` pulls in `utils/native_scaler.py:11`,
    `from torch._six import inf`. That module was removed in torch 2, so a
    clean clone fails at import under det (torch 2.5) and det_cu128 (2.11).
  * With the repo root on sys.path (every eval/ script's shim), this repo's
    `utils.py` shadows upstream's `utils/` package, so the import fails even
    with that patched.
  * det_cu128, the env for the RTX PRO 6000 nodes, has no einops or timm.
"""

from .multivit import MultiViTEncoder, build_2d_sincos_posemb

__all__ = ['MultiViTEncoder', 'build_2d_sincos_posemb']
