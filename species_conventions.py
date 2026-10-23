"""Render conventions per species, and the depth adapter for cross-species inputs.

The encoder reads the encoded depth value raw, so a convention mismatch feeds it values
~10x out of range (seen on the sorghum row renderer, job 17070980). Conventions, measured
against the camera-frame clouds:

    sorghum_std   code = (z - 0.0238) / 50          fg ~ 0.03-0.07
    maize_std     code = (z - near) / (far - near)  per-plant near/far (camera_pose.json);
                                                    fg lands in ~[0.26, 0.74]
    maize_row     code = (z - 0.05) / (3.0 - 0.05)  the maize row renderer
    sorghum_row   code = (z - 0.05) / (4.5 - 0.05)  the sorghum row renderer (decode only:
                                                    sorghum models are FED sorghum_std)

Decoders map an encoded image to METRES under the render's own convention; encoders map
metres into the TARGET MODEL's training convention. `maize_std` has no global inverse, so
its encoder maps each image's robust foreground span onto [0.26, 0.74], the band the
per-plant near/far encoding produces on the maize renders. RGB and point clouds need no
adapter (ImageNet normalisation; unit-sphere frame), but the cloud CONTENT differs:
sorghum models trained on complete plants (67 % of points occluded from the camera),
maize on single-camera captures (7 %) -- a domain gap, not a code transform.
"""
import json
from pathlib import Path

import numpy as np
import torch

SORGHUM_STD = dict(offset=0.0238, scale=50.0)            # code = (z - offset) / scale
ROW_NEARFAR = dict(sorghum=(0.05, 4.5), maize=(0.05, 3.0))
MAIZE_BAND = (0.26, 0.74)


def depth_decoder(convention, folder=None):
    """encoded [0,1] -> metres. 'maize_std' needs the render folder (camera_pose.json)."""
    if convention == 'sorghum_std':
        return lambda c: c * SORGHUM_STD['scale'] + SORGHUM_STD['offset']
    if convention == 'maize_std':
        cp = json.loads((Path(folder) / 'camera_pose.json').read_text())
        near, far = float(cp['near']), float(cp['far'])
        return lambda c: c * (far - near) + near
    if convention in ('sorghum_row', 'maize_row'):
        near, far = ROW_NEARFAR[convention.split('_')[0]]
        return lambda c: c * (far - near) + near
    raise ValueError(f'unknown depth convention {convention!r}')


def depth_encoder(convention):
    """metres -> encoded, in the TARGET model's training convention.
    Returns f(z_fg) applied to the foreground values of ONE image."""
    if convention == 'sorghum_std':
        return lambda z: (z - SORGHUM_STD['offset']) / SORGHUM_STD['scale']
    if convention == 'maize_row':
        near, far = ROW_NEARFAR['maize']
        return lambda z: (z - near) / (far - near)
    if convention == 'maize_std':
        lo, hi = MAIZE_BAND
        def f(z):
            if isinstance(z, np.ndarray):
                z0, z1 = np.quantile(z, 0.01), np.quantile(z, 0.99)
            else:
                z0, z1 = torch.quantile(z, 0.01), torch.quantile(z, 0.99)
            span = (z1 - z0) if float(z1 - z0) > 1e-6 else 1.0
            return lo + (hi - lo) * (z - z0) / span
        return f
    raise ValueError(f'unknown target depth convention {convention!r}')


def model_depth_convention(species, run_cfg=None):
    """What a given model was trained to see. Sorghum models (standard and row fine-tunes
    alike) see sorghum_std -- the row loader re-encodes into it. Maize ROW models trained
    on the raw row encoding; every other maize model on maize_std."""
    if species == 'sorghum':
        return 'sorghum_std'
    if run_cfg and run_cfg.get('trainer') == 'row':
        return 'maize_row'
    return 'maize_std'


def reencode_depth(depth, decode, encode, fg_thresh=1e-3):
    """Re-encode ONE image's depth (torch tensor or ndarray, any shape, 0 = background)
    from its render convention into a model's convention. Background stays 0; foreground
    is clipped to [1e-6, 1] (bilinear-resize bleed at silhouette edges decodes to nonsense
    below fg_thresh and is left at its tiny value rather than trusted)."""
    is_torch = torch.is_tensor(depth)
    d = depth.clone() if is_torch else depth.copy()
    fg = d > fg_thresh
    if not bool(fg.any()):
        return d
    enc = encode(decode(d[fg]))
    if is_torch:
        d[fg] = enc.clamp(1e-6, 1.0).to(d.dtype)
    else:
        d[fg] = np.clip(enc, 1e-6, 1.0).astype(d.dtype)
    return d
