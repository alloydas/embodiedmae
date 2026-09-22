#!/usr/bin/env python3
"""EmbodiedMAE-4M for MAIZE — RGB + Depth + PointCloud + procedural params.

Maize and sorghum are kept as separate pipelines on purpose. This file owns
everything maize-specific; `embodied_mae_4m.py` is never modified and never
learns about maize.

What is NOT duplicated: the encoder/decoder machinery is genuinely identical
across species, and `N_PARAMS` couples to it at exactly two construction
points — `ParamEmbed(N_PARAMS, embed_dim)` (embodied_mae_4m.py:491) and
`nn.Linear(decoder_embed_dim, N_PARAMS)` (:558). Everything else reads widths
off the tensors. So this subclasses `EmbodiedMAE4M` and rebuilds those two
modules at maize width, rather than cloning 1,100 lines that would drift.
`max_leaves` is already a constructor argument and needs no surgery.

──────────────────────────────────────────────────────────────────────────────
PARAMETER LAYOUT — N_PARAMS = 14, MAX_LEAVES = 28
──────────────────────────────────────────────────────────────────────────────
Derived from all 2,250 plants / 27,278 leaves of the complete `test` split. The
source XML carries 48 attributes and **31 of them are dropped**, which matters:
keeping them would repeat the sorghum failure where constant fields inflated the
reported parameter accuracy (see the param-metric note in CLAUDE.md).

Dropped, and why:
  * 26 are CONSTANT across every plant — the whole `<Tassel>` block bar
    `matureDroopStrength`, `<Tiller>@type/@alpha`, `<plant>@species/@phenotypeId`,
    and 9 of the 26 `<leaf>` attributes (splinePoints, surfaceNoise*, midrib*,
    ligule*, sheathOuterScale). Maize is worse than sorghum here, not better.
  * 3 are DETERMINISTIC RESTATEMENTS a constancy check cannot see:
    `waveRAmp` is bit-identical to `waveLAmp` (max|diff| = 0.0 over 27,278
    records), `waveRFreq` to `waveLFreq`, and `waveRPhase == waveLPhase + 1.57`
    (the generator's literal 1.57, not pi/2).
  * `<Tassel>@seed` == the integer plant id for 2,250/2,250 plants. It is an
    IDENTITY LEAK, not a phenotype, and because splits are assigned by plant id
    it leaks split membership too.
  * `<leaf>@id` is 0..n-1 in document order; the token index already carries it.

`leafAzimuthDeg` is stored as `azJitterDeg`, NOT raw and NOT sin/cos:
    leafAzimuthDeg == (180.0 * leaf_index + U(-15, +15)) mod 360
i.e. distichous phyllotaxis — leaves alternate 180° with a small jitter.
  * Raw linear is wrong: 34.3 % of leaves sit within 10° of the 0/360 seam, so
    two physically identical leaves land a full range apart and Smooth-L1
    rewards predicting 180°, the one direction no leaf points.
  * sin/cos is ALSO wrong here: cos(az) is ±1 by leaf parity, so a decoder that
    ignores every input modality scores R² = 0.9999 on that slot. That is the
    sorghum roll_angle bug's twin — a free win that inflates the metric.
  * `azJitterDeg = wrap180(az - 180*leaf_index)` is exactly invertible, has no
    seam, and scores R² = 0.0007 from token position alone.

`waveLPhase` IS genuinely circular (radians, span 10.62 > 2π) with no
index-relative trick available, so it takes a sin/cos pair and decodes by atan2.
`leafTwist`, `leafAngle`, `droopiness`, `stemInclinationDeg` are in degrees but
do NOT wrap — they are bounded physical quantities. Do not "fix" them.

Round-trip raw → norm → raw was validated at max abs error 8.5e-14 (leaf) and
2.2e-16 (plant), with ZERO values clipped at either end of [0, 1].
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np
import torch
import torch.nn as nn

from embodied_mae_4m import EmbodiedMAE4M, ParamEmbed

# ── Numeric parameter layout ─────────────────────────────────────────────────

N_PARAMS = 14      # float vector per token (sorghum uses 9)
MAX_LEAVES = 28    # observed 4..22 leaves/plant; headroom (sorghum uses 24)

PLANT_FIELDS = ('leafCount', 'stemRadius', 'stemShrink',
                'tasselMatureDroopStrength', 'stemInternodeSum')
LEAF_FIELDS = ('distance', 'leafLength', 'leafWidth', 'leafAngle', 'droopiness',
               'stemInclinationDeg', 'widthTaper', 'leafTwist', 'leafCurl',
               'waveLAmp', 'waveLFreq', 'azJitterDeg',
               'sinWavePhase', 'cosWavePhase')

# norm = clip((raw + SHIFT) / SCALE, 0, 1)   ·   raw = norm * SCALE - SHIFT
# Same convention as sorghum so the loss and decoder need no changes.
# Defined ONCE. (embodied_mae_4m.py defines its equivalents twice, lines 46-97
# and 245-306, with the second shadowing the first — editing only the first is a
# silent no-op. Do not reintroduce that here.)
_PLANT_SCALE = np.array([32.0, 0.03, 0.02, 3.5, 2.0] + [1.0] * 9, np.float32)
_PLANT_SHIFT = np.array([0.0, 0.0, 0.0, 0.0, 0.0] + [0.0] * 9, np.float32)

_LEAF_SCALE = np.array([0.40, 0.85, 0.15, 180.0, 120.0, 4.0, 2.0,
                        600.0, 1.7, 0.04, 60.0, 40.0, 2.0, 2.0], np.float32)
_LEAF_SHIFT = np.array([0.00, -0.05, 0.00, 90.0, 150.0, 2.0, -2.0,
                        300.0, 0.5, 0.00, 12.0, 20.0, 1.0, 1.0], np.float32)


def _wrap180(deg):
    """Map an angle difference into (-180, 180]."""
    return (deg + 180.0) % 360.0 - 180.0


def _plant_to_params(tassel, tiller, leaves) -> np.ndarray:
    """Plant token: 5 informative slots, zero-padded to N_PARAMS."""
    raw = np.zeros(N_PARAMS, dtype=np.float32)
    raw[0] = float(len(leaves))
    raw[1] = float(tiller.get('radius', 0.0))
    raw[2] = float(tiller.get('stemShrink', 0.0))
    raw[3] = float(tassel.get('matureDroopStrength', 0.0)) if tassel is not None else 0.0
    # stem_internodeSum is the one plant-level field that is not a raw XML
    # attribute: it aggregates the per-leaf `distance`, and plant_scores.csv's
    # `stem_internodeSum` column is exactly this sum (pearson r = 1.000000).
    raw[4] = float(sum(float(lf.get('distance', 0.0)) for lf in leaves))
    return np.clip((raw + _PLANT_SHIFT) / _PLANT_SCALE, 0.0, 1.0)


def _leaf_to_params(leaf, leaf_index: int) -> np.ndarray:
    """Leaf token. `leaf_index` is needed to de-trend the azimuth."""
    phase = float(leaf.get('waveLPhase', 0.0))
    az = float(leaf.get('leafAzimuthDeg', 0.0))
    raw = np.array([
        float(leaf.get('distance', 0.0)),
        float(leaf.get('leafLength', 0.0)),
        float(leaf.get('leafWidth', 0.0)),
        float(leaf.get('leafAngle', 0.0)),
        float(leaf.get('droopiness', 0.0)),
        float(leaf.get('stemInclinationDeg', 0.0)),
        float(leaf.get('widthTaper', 0.0)),
        float(leaf.get('leafTwist', 0.0)),
        float(leaf.get('leafCurl', 0.0)),
        float(leaf.get('waveLAmp', 0.0)),
        float(leaf.get('waveLFreq', 0.0)),
        _wrap180(az - 180.0 * leaf_index),     # azJitterDeg
        math.sin(phase),
        math.cos(phase),
    ], dtype=np.float32)
    return np.clip((raw + _LEAF_SHIFT) / _LEAF_SCALE, 0.0, 1.0)


def _params_to_plant_text(params: np.ndarray) -> str:
    p = np.clip(np.asarray(params, dtype=np.float32), 0.0, 1.0)
    raw = p * _PLANT_SCALE - _PLANT_SHIFT
    n, r, sh, td, isum = raw[:5]
    return (f"n={n:.1f} r={r:.5f} shrink={sh:.5f} "
            f"droop={td:.3f} internode={isum:.4f}")


def _params_to_leaf_text(params: np.ndarray) -> str:
    p = np.clip(np.asarray(params, dtype=np.float32), 0.0, 1.0)
    raw = p * _LEAF_SCALE - _LEAF_SHIFT
    (d, ln, w, ang, drp, inc, tap, tw, cur, wa, wf, azj, s, c) = raw
    phase = math.atan2(s, c)          # circular decode
    return (f"d={d:.4f} len={ln:.4f} w={w:.4f} ang={ang:+06.2f} "
            f"droop={drp:+07.2f} twist={tw:+07.2f} curl={cur:+.3f} "
            f"wAmp={wa:.5f} wFreq={wf:+06.2f} azJit={azj:+06.2f} ph={phase:+.3f}")


def load_spline_params(xml_path, max_leaves: int = MAX_LEAVES):
    """Parse a maize spline XML into the 4M text-modality tensors.

    Returns
      valid        : (1+max_leaves,)            float32 — 1=real token, 0=pad
      param_floats : (1+max_leaves, N_PARAMS)   float32 — encoder input + target

    Same signature and return order as the sorghum loader so the dataset and
    training loop are unchanged. ElementTree parses these in ~0.26 ms against
    ~85 ms for a sorghum YAML — about 300x cheaper, which materially changes the
    dataloader budget (sorghum's YAML parse was 72 % of per-item cost).
    """
    root = ET.parse(xml_path).getroot()
    tassel = root.find('Tassel')
    tiller = root.find('Tiller')
    if tiller is None:
        raise ValueError(f'{xml_path}: no <Tiller>')
    leaves = tiller.findall('./leaves/leaf')

    n_tokens = 1 + max_leaves
    valid = np.zeros(n_tokens, dtype=np.float32)
    param_floats = np.zeros((n_tokens, N_PARAMS), dtype=np.float32)

    valid[0] = 1.0
    param_floats[0] = _plant_to_params(tassel, tiller, leaves)

    for i, leaf in enumerate(leaves[:max_leaves]):
        valid[1 + i] = 1.0
        param_floats[1 + i] = _leaf_to_params(leaf, i)

    return torch.from_numpy(valid), torch.from_numpy(param_floats)


# ── Model ────────────────────────────────────────────────────────────────────

class EmbodiedMAE4MMaize(EmbodiedMAE4M):
    """EmbodiedMAE4M at maize parameter width.

    Rebuilds only the two modules whose shape depends on N_PARAMS. The encoder,
    decoder, masking, losses and the whole forward path are inherited unchanged
    — they read widths off the tensors.
    """

    def __init__(self, *args, max_leaves: int = MAX_LEAVES, **kwargs):
        super().__init__(*args, max_leaves=max_leaves, **kwargs)

        if 'text' in self.active_modalities:
            # Encoder side: embodied_mae_4m.py:491 built this at sorghum width.
            embed_dim = self.param_embed.proj.out_features
            self.param_embed = ParamEmbed(N_PARAMS, embed_dim)

            # Decoder side: the last Linear of decoder_pred_params
            # (embodied_mae_4m.py:553-559) emits one float per parameter slot.
            head = self.decoder_pred_params
            last = head[-1]
            if not isinstance(last, nn.Linear):
                raise RuntimeError(
                    f'decoder_pred_params tail is {type(last).__name__}, not '
                    'nn.Linear — the head layout changed upstream and this '
                    'subclass needs updating rather than silently mis-sizing.')
            head[-1] = nn.Linear(last.in_features, N_PARAMS)

        self.n_params = N_PARAMS

    def decode_params_to_text(self, params):
        """(n_tokens, N_PARAMS) → list[str], for the visualisation grid."""
        arr = params.detach().cpu().numpy() if torch.is_tensor(params) else np.asarray(params)
        out = [_params_to_plant_text(arr[0])]
        out.extend(_params_to_leaf_text(row) for row in arr[1:])
        return out


def embodied_mae_4m_maize_small(**kw):
    return EmbodiedMAE4MMaize(embed_dim=384, depth=12, num_heads=6,
                              decoder_embed_dim=192, decoder_depth=4,
                              decoder_num_heads=3, **kw)


def embodied_mae_4m_maize_base(**kw):
    return EmbodiedMAE4MMaize(embed_dim=768, depth=12, num_heads=12,
                              decoder_embed_dim=512, decoder_depth=8,
                              decoder_num_heads=16, **kw)


def embodied_mae_4m_maize_large(**kw):
    return EmbodiedMAE4MMaize(embed_dim=1024, depth=24, num_heads=16,
                              decoder_embed_dim=512, decoder_depth=8,
                              decoder_num_heads=16, **kw)


if __name__ == '__main__':
    m = embodied_mae_4m_maize_base()
    n_text = 1 + m.max_leaves
    print(f'EmbodiedMAE-4M-Maize-Base  params={sum(p.numel() for p in m.parameters()):,}')
    print(f'  N_PARAMS={N_PARAMS}  max_leaves={m.max_leaves}  text tokens={n_text}')
    B = 2
    out = m(torch.randn(B, 3, 224, 224), torch.randn(B, 1, 224, 224),
            torch.randn(B, 8192, 3), torch.rand(B, n_text, N_PARAMS),
            torch.ones(B, n_text), mask_ratio=0.8)
    print('  forward OK:', type(out).__name__)
