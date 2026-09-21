"""
EmbodiedMAE-4M: RGB + Depth + PointCloud + Spline Parameters

Spline / parameter modality (encoder → decoder design):
  ENCODER INPUT  : N_PARAMS-dim float vector per token → ParamEmbed
                   (Linear → GELU → LayerNorm)
  DECODER OUTPUT : direct float regression of the same numeric values
                   (Smooth-L1 loss, all params normalised to [0, 1])
  DISPLAY        : predicted floats are un-normalised and formatted as text
                   for visualisation only (decode_params_to_text)

Parameter layout (N_PARAMS = 9 per token):
  Plant token (idx 0) : [sl, sd_x, sd_y, sd_z, ps_x, ps_y, ps_z, pa, pr]
  Leaf  token (idx 1+): [sp, ln,   ra,   ba,   wf,   wp0,  wp1,  0,  0 ]
  All values normalised to [0, 1] — see _PLANT_SCALE / _LEAF_SCALE below.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from embodied_mae import (
    PatchEmbed,
    PointCloudEmbed,
    TransformerBlock,
    get_2d_sincos_pos_embed,
    chamfer_distance,
    qal_loss,
    earth_movers_distance,
)


# ── Numeric parameter layout ──────────────────────────────────────────────────

N_PARAMS = 9   # fixed-size float vector per token

# Canonical modality order. Token layout, loss order and state_dict keys all
# follow it, so an arm's layout never depends on the order the caller wrote
# active_modalities in. Same vocabulary as train_sorghum_4m_distill.py:83.
MODALITIES = ('rgb', 'depth', 'pc', 'text')

# norm = (raw + shift) / scale  →  raw = norm * scale - shift
# All outputs land in [0, 1] for the expected data ranges.
_PLANT_SCALE = np.array([3.0, 2.0, 2.0, 2.0,  1.0, 1.0, 1.0, 50.0, 0.02], np.float32)
_PLANT_SHIFT = np.array([0.0, 1.0, 1.0, 1.0,  0.0, 0.0, 0.0,  0.0,  0.0], np.float32)
# [sl,   sdx,  sdy,  sdz,  psx, psy, psz, pa,  pr ]

_LEAF_SCALE  = np.array([1.0, 1.0, 360.0, 180.0, 0.1, 360.0, 360.0, 1.0, 1.0], np.float32)
_LEAF_SHIFT  = np.array([0.0, 0.0,   0.0,   0.0, 0.0,   0.0,   0.0, 0.0, 0.0], np.float32)
# [sp,  ln,  ra,   ba,   wf,  wp0,  wp1, (unused×2)]


def _plant_to_params(plant: dict) -> np.ndarray:
    p  = plant['Parameters']
    sd = p['stem_direction']
    ps = p['panicle_size']
    raw = np.array([
        float(p['stem_length']),
        float(sd[0]), float(sd[1]), float(sd[2]),
        float(ps[0]), float(ps[1]), float(ps[2]),
        float(p['panicle_seed_amount']),
        float(p['panicle_seed_radius']),
    ], dtype=np.float32)
    return np.clip((raw + _PLANT_SHIFT) / _PLANT_SCALE, 0.0, 1.0)


def _leaf_to_params(leaf: dict) -> np.ndarray:
    wps = leaf['waviness_period_start']
    raw7 = np.array([
        float(leaf['starting_point']),
        float(leaf['length']),
        float(leaf['roll_angle']),
        float(leaf['branching_angle']),
        float(leaf['waviness_frequency']),
        float(wps[0]), float(wps[1]),
    ], dtype=np.float32)
    norm7 = np.clip((raw7 + _LEAF_SHIFT[:7]) / _LEAF_SCALE[:7], 0.0, 1.0)
    return np.concatenate([norm7, [0.0, 0.0]])   # pad to N_PARAMS=9


def _params_to_plant_text(params: np.ndarray) -> str:
    p   = np.clip(params, 0.0, 1.0)
    raw = p * _PLANT_SCALE - _PLANT_SHIFT
    sl, sdx, sdy, sdz, psx, psy, psz, pa, pr = raw
    return (f"sl={sl:.4f} sd={sdx:+.3f},{sdy:+.3f},{sdz:+.3f} "
            f"ps={psx:.3f},{psy:.3f},{psz:.3f} pa={int(round(pa))} pr={pr:.4f}")


def _params_to_leaf_text(params: np.ndarray) -> str:
    p    = np.clip(params[:7], 0.0, 1.0)
    raw7 = p * _LEAF_SCALE[:7] - _LEAF_SHIFT[:7]
    sp, ln, ra, ba, wf, wp0, wp1 = raw7
    return (f"sp={sp:.4f} ln={ln:.4f} ra={ra:06.2f} ba={ba:06.2f} "
            f"wf={wf:.6f} wp={wp0:06.2f},{wp1:06.2f}")


def _bounded_proportional_allocation(total, weights, lower, upper):
    """Allocate an integer total proportionally while respecting bounds.

    The continuous allocation is a capped water-filling solution.  Largest
    remainders then convert it to integers without changing ``total``.  This is
    deliberately a pure-Python helper: modality counts are shared by the whole
    batch, tiny, and easy to unit-test independently of model construction.
    """
    n_items = len(weights)
    if n_items == 0 or len(lower) != n_items or len(upper) != n_items:
        raise ValueError("weights, lower, and upper must have the same non-zero length")
    if not isinstance(total, int) or isinstance(total, bool):
        raise TypeError("total must be an integer")
    if any(not isinstance(value, int) or isinstance(value, bool)
           for value in (*lower, *upper)):
        raise TypeError("allocation bounds must be integers")
    if any(lo < 0 or hi < lo for lo, hi in zip(lower, upper)):
        raise ValueError("allocation bounds must satisfy 0 <= lower <= upper")
    if total < sum(lower) or total > sum(upper):
        raise ValueError("total is outside the feasible allocation bounds")

    weights = tuple(float(value) for value in weights)
    if any(not math.isfinite(value) or value < 0 for value in weights):
        raise ValueError("allocation weights must be finite and non-negative")
    if sum(weights) <= 0:
        raise ValueError("at least one allocation weight must be positive")

    continuous = [float(value) for value in lower]
    remaining = float(total - sum(lower))
    active = {index for index, (lo, hi) in enumerate(zip(lower, upper))
              if lo < hi}

    # Redistribute the share of every saturated item across all items that
    # still have capacity.  Zero-weight survivors use a uniform fallback once
    # all positive-weight items have saturated.
    while remaining > 1e-12:
        if not active:
            raise RuntimeError("allocation exhausted capacity before reaching total")
        active_weight = sum(weights[index] for index in active)
        if active_weight > 0:
            shares = {
                index: remaining * weights[index] / active_weight
                for index in active
            }
        else:
            uniform_share = remaining / len(active)
            shares = {index: uniform_share for index in active}

        saturated = [
            index for index in active
            if shares[index] >= upper[index] - continuous[index] - 1e-12
        ]
        if saturated:
            for index in saturated:
                capacity = upper[index] - continuous[index]
                continuous[index] = float(upper[index])
                remaining -= capacity
                active.remove(index)
            remaining = max(remaining, 0.0)
        else:
            for index in active:
                continuous[index] += shares[index]
            remaining = 0.0

    allocation = [int(math.floor(value)) for value in continuous]
    remainder = total - sum(allocation)
    while remainder > 0:
        candidates = [index for index in range(n_items)
                      if allocation[index] < upper[index]]
        if not candidates:
            raise RuntimeError("integer allocation exhausted capacity")
        index = max(
            candidates,
            key=lambda item: (
                continuous[item] - allocation[item], weights[item], -item,
            ),
        )
        allocation[index] += 1
        remainder -= 1

    return tuple(allocation)


def _visible_token_counts(lengths, weights, mask_ratio_total,
                          min_mask_ratio=0.25):
    """Return bounded visible-token counts whose sum matches the global ratio.

    At least one token stays visible in every modality.  ``min_mask_ratio`` is
    treated as a per-modality upper bound on visibility whenever those bounds
    can accommodate the requested global total.  If they cannot, the minimum
    masking constraint is relaxed only by the unavoidable overflow, which is
    redistributed according to ``weights``.
    """
    lengths = tuple(lengths)
    if not lengths:
        raise ValueError("at least one modality is required")
    if any(not isinstance(length, int) or isinstance(length, bool) or length <= 0
           for length in lengths):
        raise ValueError("all modality lengths must be positive integers")
    if len(weights) != len(lengths):
        raise ValueError("weights and modality lengths must have the same length")

    try:
        mask_ratio_total = float(mask_ratio_total)
        min_mask_ratio = float(min_mask_ratio)
    except (TypeError, ValueError) as exc:
        raise TypeError("mask ratios must be real numbers") from exc
    if not math.isfinite(mask_ratio_total) or not 0.0 <= mask_ratio_total <= 1.0:
        raise ValueError("mask_ratio_total must be finite and in [0, 1]")
    if not math.isfinite(min_mask_ratio) or not 0.0 <= min_mask_ratio <= 1.0:
        raise ValueError("min_mask_ratio must be finite and in [0, 1]")

    total_tokens = sum(lengths)
    # Preserve the model's existing floor convention for a fractional token.
    total_visible = int(math.floor(total_tokens * (1.0 - mask_ratio_total)))
    minimum_visible = len(lengths)
    if total_visible < minimum_visible:
        raise ValueError(
            f"mask_ratio_total leaves {total_visible} visible tokens, but at "
            f"least {minimum_visible} are required to keep one per modality"
        )

    lower = (1,) * len(lengths)
    constrained_upper = tuple(
        max(1, int(math.floor(
            length * (1.0 - min_mask_ratio) + 1e-12
        )))
        for length in lengths
    )

    if total_visible <= sum(constrained_upper):
        upper = constrained_upper
        allocation_lower = lower
    else:
        # The global ratio has priority.  Starting at every constrained upper
        # bound makes the amount of min-mask relaxation exactly the unavoidable
        # overflow, rather than relaxing a modality that could stay bounded.
        # Keep at least one reconstruction target per modality whenever the
        # global budget contains enough masked tokens to do so.  Besides being
        # a better MAE objective, this prevents empty masked-loss denominators.
        total_masked = total_tokens - total_visible
        if total_masked >= len(lengths) and all(length > 1 for length in lengths):
            upper = tuple(length - 1 for length in lengths)
        else:
            upper = lengths
        allocation_lower = constrained_upper

    return _bounded_proportional_allocation(
        total_visible, weights, allocation_lower, upper)

# norm = (raw + shift) / scale  →  raw = norm * scale - shift
# All outputs land in [0, 1] for the expected data ranges.
_PLANT_SCALE = np.array([3.0, 2.0, 2.0, 2.0,  1.0, 1.0, 1.0, 50.0, 0.02], np.float32)
_PLANT_SHIFT = np.array([0.0, 1.0, 1.0, 1.0,  0.0, 0.0, 0.0,  0.0,  0.0], np.float32)
# [sl,   sdx,  sdy,  sdz,  psx, psy, psz, pa,  pr ]

_LEAF_SCALE  = np.array([1.0, 1.0, 360.0, 180.0, 0.1, 360.0, 360.0, 1.0, 1.0], np.float32)
_LEAF_SHIFT  = np.array([0.0, 0.0,   0.0,   0.0, 0.0,   0.0,   0.0, 0.0, 0.0], np.float32)
# [sp,  ln,  ra,   ba,   wf,  wp0,  wp1, (unused×2)]

def _plant_to_params(plant: dict) -> np.ndarray:
    p  = plant['Parameters']
    sd = p['stem_direction']
    ps = p['panicle_size']
    raw = np.array([
        float(p['stem_length']),
        float(sd[0]), float(sd[1]), float(sd[2]),
        float(ps[0]), float(ps[1]), float(ps[2]),
        float(p['panicle_seed_amount']),
        float(p['panicle_seed_radius']),
    ], dtype=np.float32)
    return np.clip((raw + _PLANT_SHIFT) / _PLANT_SCALE, 0.0, 1.0)


def _leaf_to_params(leaf: dict) -> np.ndarray:
    wps = leaf['waviness_period_start']
    raw7 = np.array([
        float(leaf['starting_point']),
        float(leaf['length']),
        float(leaf['roll_angle']),
        float(leaf['branching_angle']),
        float(leaf['waviness_frequency']),
        float(wps[0]), float(wps[1]),
    ], dtype=np.float32)
    norm7 = np.clip((raw7 + _LEAF_SHIFT[:7]) / _LEAF_SCALE[:7], 0.0, 1.0)
    return np.concatenate([norm7, [0.0, 0.0]])   # pad to N_PARAMS=9


def _params_to_plant_text(params: np.ndarray) -> str:
    p   = np.clip(params, 0.0, 1.0)
    raw = p * _PLANT_SCALE - _PLANT_SHIFT
    sl, sdx, sdy, sdz, psx, psy, psz, pa, pr = raw
    return (f"sl={sl:.4f} sd={sdx:+.3f},{sdy:+.3f},{sdz:+.3f} "
            f"ps={psx:.3f},{psy:.3f},{psz:.3f} pa={int(round(pa))} pr={pr:.4f}")


def _params_to_leaf_text(params: np.ndarray) -> str:
    p    = np.clip(params[:7], 0.0, 1.0)
    raw7 = p * _LEAF_SCALE[:7] - _LEAF_SHIFT[:7]
    sp, ln, ra, ba, wf, wp0, wp1 = raw7
    return (f"sp={sp:.4f} ln={ln:.4f} ra={ra:06.2f} ba={ba:06.2f} "
            f"wf={wf:.6f} wp={wp0:06.2f},{wp1:06.2f}")


# ── Data loading ──────────────────────────────────────────────────────────────


# ── Data loading ──────────────────────────────────────────────────────────────

def load_spline_params(yml_path, max_leaves: int = 24):
    """Parse a spline YAML into:
      valid        : (1+max_leaves,)            float32 — 1=real token, 0=pad
      param_floats : (1+max_leaves, N_PARAMS)   float32 — encoder input + target
    """
    import yaml
    # The spline YAMLs are ~258 KB and get re-parsed on every __getitem__, which
    # made pure-python yaml.safe_load the single largest cost in the whole data
    # pipeline (~729 ms/sample vs ~11 ms for the point cloud). libyaml's C loader
    # parses the same documents ~8.6x faster; fall back if it is unavailable.
    try:
        from yaml import CSafeLoader as _Loader
    except ImportError:
        _Loader = yaml.SafeLoader
    with open(yml_path) as f:
        data = yaml.load(f, Loader=_Loader)

    plant  = data['Sorghums'][0]
    # Some leaves are geometry-only (Center/Left/Right Points but no procedural
    # params) — skip them so they don't become param tokens and crash _leaf_to_params.
    _REQ_LEAF = ('starting_point', 'length', 'roll_angle', 'branching_angle',
                 'waviness_frequency', 'waviness_period_start')
    leaves = [lf for lf in plant['Leaves'] if all(k in lf for k in _REQ_LEAF)]

    n_tokens     = 1 + max_leaves
    valid        = np.zeros(n_tokens,            dtype=np.float32)
    param_floats = np.zeros((n_tokens, N_PARAMS), dtype=np.float32)

    valid[0]        = 1.0
    param_floats[0] = _plant_to_params(plant)

    for i, leaf in enumerate(leaves[:max_leaves]):
        valid[1 + i]        = 1.0
        param_floats[1 + i] = _leaf_to_params(leaf)

    return torch.from_numpy(valid), torch.from_numpy(param_floats)


# ── Param embedder (encoder input) ───────────────────────────────────────────

class ParamEmbed(nn.Module):
    """Embed an N_PARAMS-dim float vector per token into the shared embed space.
    Pipeline: floats → Linear → GELU → LayerNorm.
    """

    def __init__(self, n_params: int = N_PARAMS, embed_dim: int = 768):
        super().__init__()
        self.proj = nn.Linear(n_params, embed_dim)
        self.act  = nn.GELU()
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """params : (B, n_tokens, N_PARAMS)  →  (B, n_tokens, D)"""
        return self.norm(self.act(self.proj(params)))


# ── Main model ────────────────────────────────────────────────────────────────

class EmbodiedMAE4M(nn.Module):
    """Multi-modal MAE: RGB | Depth | PointCloud | Parametric.

    Param modality:
      encoder reads N_PARAMS-dim float vector per token (ParamEmbed)
      decoder predicts N_PARAMS float values per masked token (Smooth-L1)
      at display time floats are formatted back to human-readable text
    """

    def __init__(
        self,
        img_size:            int   = 224,
        patch_size:          int   = 16,
        in_chans_rgb:        int   = 3,
        in_chans_depth:      int   = 1,
        num_pc_tokens:       int   = 196,
        pc_group_size:       int   = 32,
        target_points:       int   = 10000,
        pc_loss_weight:      float = 50.0,
        max_leaves:          int   = 24,
        spline_loss_weight:  float = 5.0,
        embed_dim:           int   = 768,
        depth:               int   = 12,
        num_heads:           int   = 12,
        decoder_embed_dim:   int   = 512,
        decoder_depth:       int   = 8,
        decoder_num_heads:   int   = 16,
        mlp_ratio:           float = 4.0,
        norm_pix_loss:       bool  = True,
        dirichlet_alpha:     float = 1.0,
        depth_norm_type:     str   = 'minmax',
        pc_loss_name:        str   = 'chamfer',
        qal_threshold:       float = 0.01,
        qal_alpha:           float = 100.0,
        qal_use_squared:     bool  = False,
        active_modalities=None,
        text_mask_ratio=None,
    ):
        super().__init__()

        # ── Modality gate (E2 ablation) ───────────────────────────────────
        # `active_modalities` selects which token streams exist AT ALL. An
        # inactive modality gets no embedder, no modality/positional embedding,
        # no decoder head and no loss term — it is absent, not merely masked.
        #
        # Why absent rather than masked: forcing a modality to zero visible
        # tokens still reconstructs it (forward_decoder restores every position
        # from mask tokens) and still adds its term to `total`, so a "PC-only"
        # model would keep learning the spline params and the ablation would
        # measure nothing. Zero-length slices also produce zero-valued grads
        # rather than None, so AdamW would go on weight-decaying a dead
        # embedder. That masked-but-present path is cross-modal distillation,
        # which is what forward_encoder_select / `visible=` are for.
        #
        # Default None == all four, and that path is bit-identical to the
        # pre-gate model, so existing checkpoints load unchanged.
        if active_modalities is None:
            active_modalities = MODALITIES
        active = tuple(str(m).lower() for m in active_modalities)
        unknown = [m for m in active if m not in MODALITIES]
        if unknown:
            raise ValueError(
                f"unknown modality/modalities {unknown}; "
                f"expected a subset of {list(MODALITIES)}")
        if len(set(active)) != len(active):
            raise ValueError(f"duplicate entries in active_modalities: {active}")
        if not active:
            raise ValueError("active_modalities must name at least one modality")
        if 'pc' not in active:
            # Every E2 arm is PC-anchored, and the PC loss is the only term
            # computed over the whole cloud rather than over masked tokens.
            raise ValueError("'pc' must be active: it anchors the reconstruction target")
        # Normalise to canonical order so the token layout never depends on the
        # order the caller happened to write the list in.
        self.active_modalities = tuple(m for m in MODALITIES if m in active)
        self.n_active = len(self.active_modalities)
        self.text_mask_ratio = text_mask_ratio

        self.img_size           = img_size
        self.patch_size         = patch_size
        self.num_patches        = (img_size // patch_size) ** 2
        self.num_pc_tokens      = num_pc_tokens
        self.max_leaves         = max_leaves
        self.n_text_tokens      = 1 + max_leaves
        self.embed_dim          = embed_dim
        self.norm_pix_loss      = norm_pix_loss
        self.dirichlet_alpha    = dirichlet_alpha
        self.pc_loss_weight     = pc_loss_weight
        self.spline_loss_weight = spline_loss_weight
        self.depth_norm_type    = depth_norm_type
        if pc_loss_name not in {'chamfer', 'qal_loss'}:
            raise ValueError(
                f"Unsupported point-cloud loss {pc_loss_name!r}; "
                "expected 'chamfer' or 'qal_loss'")
        if qal_threshold < 0:
            raise ValueError("qal_threshold must be non-negative")
        if qal_alpha <= 0:
            raise ValueError("qal_alpha must be positive")
        self.pc_loss_name       = pc_loss_name
        self.qal_threshold      = qal_threshold
        self.qal_alpha          = qal_alpha
        self.qal_use_squared    = qal_use_squared
        self.target_points      = target_points
        self.points_per_token   = target_points // num_pc_tokens

        # Token count per modality — the single source of truth for every layout
        # computation below (encoder cat, decoder restore, head split).
        self._token_len = {
            'rgb':   self.num_patches,
            'depth': self.num_patches,
            'pc':    num_pc_tokens,
            'text':  self.n_text_tokens,
        }

        # ── Embedders (active modalities only) ────────────────────────────
        # Attribute NAMES are unchanged so state_dict keys match the pre-gate
        # model; an inactive modality simply contributes no keys.
        act = self.active_modalities
        self.rgb_embed   = (PatchEmbed(img_size, patch_size, in_chans_rgb, embed_dim)
                            if 'rgb' in act else None)
        self.depth_embed = (PatchEmbed(img_size, patch_size, in_chans_depth, embed_dim)
                            if 'depth' in act else None)
        self.pc_embed    = (PointCloudEmbed(num_pc_tokens, pc_group_size, embed_dim)
                            if 'pc' in act else None)
        self.param_embed = (ParamEmbed(N_PARAMS, embed_dim)
                            if 'text' in act else None)

        # ── Modality embeddings ───────────────────────────────────────────
        def _mod_embed(name):
            return (nn.Parameter(torch.zeros(1, 1, embed_dim))
                    if name in act else None)
        self.modality_embed_rgb   = _mod_embed('rgb')
        self.modality_embed_depth = _mod_embed('depth')
        self.modality_embed_pc    = _mod_embed('pc')
        self.modality_embed_text  = _mod_embed('text')

        # ── Positional embeddings ─────────────────────────────────────────
        # pos_embed_2d is SHARED by rgb and depth, so it exists if either does.
        self._needs_2d = ('rgb' in act) or ('depth' in act)
        self.pos_embed_2d   = (nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)
            if self._needs_2d else None)
        self.pos_embed_pc   = (nn.Parameter(
            torch.zeros(1, num_pc_tokens, embed_dim), requires_grad=True)
            if 'pc' in act else None)
        self.pos_embed_text = (nn.Parameter(
            torch.zeros(1, self.n_text_tokens, embed_dim), requires_grad=True)
            if 'text' in act else None)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # ── Encoder ───────────────────────────────────────────────────────
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True)
            for _ in range(depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # ── Decoder ───────────────────────────────────────────────────────
        self.decoder_embed  = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token     = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed_2d   = (nn.Parameter(
            torch.zeros(1, self.num_patches, decoder_embed_dim), requires_grad=False)
            if self._needs_2d else None)
        self.decoder_pos_embed_pc   = (nn.Parameter(
            torch.zeros(1, num_pc_tokens, decoder_embed_dim), requires_grad=False)
            if 'pc' in act else None)
        self.decoder_pos_embed_text = (nn.Parameter(
            torch.zeros(1, self.n_text_tokens, decoder_embed_dim), requires_grad=False)
            if 'text' in act else None)

        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        # ── Prediction heads (active modalities only) ─────────────────────
        self.decoder_pred_rgb   = (
            nn.Linear(decoder_embed_dim, patch_size**2 * in_chans_rgb)
            if 'rgb' in act else None)
        self.decoder_pred_depth = (
            nn.Linear(decoder_embed_dim, patch_size**2 * in_chans_depth)
            if 'depth' in act else None)

        # Param head: 2-layer MLP — produces N_PARAMS normalised floats per token
        self.decoder_pred_params = (nn.Sequential(
            nn.Linear(decoder_embed_dim, decoder_embed_dim),
            nn.GELU(),
            nn.LayerNorm(decoder_embed_dim),
            nn.Linear(decoder_embed_dim, N_PARAMS),
        ) if 'text' in act else None)

        # PC folding decoder
        self.decoder_pc_proj = (nn.Linear(decoder_embed_dim, 512)
                                if 'pc' in act else None)
        self.decoder_pc_fold = (nn.Sequential(
            nn.Linear(514, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 3),
        ) if 'pc' in act else None)

        self.initialize_weights()

    # ── Weights ───────────────────────────────────────────────────────────────

    def initialize_weights(self):
        # pos_embed_2d / decoder_pos_embed_2d exist only if rgb or depth is active.
        if self.pos_embed_2d is not None:
            pos_2d = get_2d_sincos_pos_embed(self.embed_dim, int(self.num_patches ** .5))
            self.pos_embed_2d.data.copy_(torch.from_numpy(pos_2d).float().unsqueeze(0))

            dec_dim = self.decoder_blocks[0].norm1.normalized_shape[0]
            dec_2d  = get_2d_sincos_pos_embed(dec_dim, int(self.num_patches ** .5))
            self.decoder_pos_embed_2d.data.copy_(
                torch.from_numpy(dec_2d).float().unsqueeze(0))

        for p in (self.pos_embed_pc, self.pos_embed_text,
                  self.decoder_pos_embed_pc, self.decoder_pos_embed_text):
            if p is not None:
                nn.init.normal_(p, std=.02)
        for e in (self.modality_embed_rgb, self.modality_embed_depth,
                  self.modality_embed_pc,  self.modality_embed_text):
            if e is not None:
                nn.init.normal_(e, std=.02)
        nn.init.normal_(self.cls_token,  std=.02)
        nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias,   0)
            nn.init.constant_(m.weight, 1.0)

    # ── Dirichlet masking ─────────────────────────────────────────────────────

    def _embed_active(self, x_rgb, x_depth, x_pc, x_param):
        """Embed ONLY the active modalities -> {name: (B, L, D)}.

        An inactive modality is not embedded at all (its embedder is None), so a
        reduced arm costs what it should instead of computing and discarding.
        """
        out = {}
        if 'rgb' in self.active_modalities:
            out['rgb'] = (self.rgb_embed(x_rgb)
                          + self.pos_embed_2d + self.modality_embed_rgb)
        if 'depth' in self.active_modalities:
            out['depth'] = (self.depth_embed(x_depth)
                            + self.pos_embed_2d + self.modality_embed_depth)
        if 'pc' in self.active_modalities:
            out['pc'] = (self.pc_embed(x_pc)
                         + self.pos_embed_pc + self.modality_embed_pc)
        if 'text' in self.active_modalities:
            out['text'] = (self.param_embed(x_param)
                           + self.pos_embed_text + self.modality_embed_text)
        return out

    def random_masking_dirichlet(self, embeds,
                                 mask_ratio_total=0.75, min_mask_ratio=0.25):
        """Mask random positions with an exact, bounded global token budget.

        `embeds` is the {name: (B, L, D)} dict from _embed_active. One Dirichlet
        draw allocates visible-token counts across the ACTIVE modalities for the
        batch; positions are shuffled independently for every sample.

        When `self.text_mask_ratio` is not None the text stream is taken out of
        the Dirichlet and masked independently at that rate. Reason: text is 25
        tokens against 196 per vision stream, and one length-agnostic global
        budget hands it a disproportionate share. Measured over 4000 draws at
        mask_ratio 0.80, the four-modality arm came out text 59% visible while
        its vision streams fell from 20% to 18% -- distorting the +params arm in
        both directions at once (an easier param pretext task AND fewer vision
        tokens than the PC+RGB+D arm it is compared against). Splitting the
        budget puts every stream at 20%, and leaves arms without text
        bit-identical to the pre-gate model.

        Returns {name: (x_vis, mask, ids_rest)} for the active modalities.
        """
        names = self.active_modalities
        missing = [n for n in names if n not in embeds]
        if missing:
            raise KeyError(f"embeds is missing active modalities {missing}")

        first = embeds[names[0]]
        B, _, embed_dim = first.shape
        device, dtype = first.device, first.dtype
        for name in names:
            t = embeds[name]
            if not isinstance(t, torch.Tensor):
                raise TypeError(f"embeds[{name!r}] must be a torch.Tensor")
            if t.ndim != 3:
                raise ValueError(
                    f"embeds[{name!r}] must have shape (B, L, D), got {tuple(t.shape)}")
            if t.shape[0] != B:
                raise ValueError("all modalities must have the same batch size")
            if t.shape[2] != embed_dim:
                raise ValueError("all modalities must have the same embedding dimension")
            if t.device != device:
                raise ValueError("all modalities must be on the same device")
            if t.dtype != dtype:
                raise ValueError("all modalities must have the same dtype")
            if t.shape[1] <= 0:
                raise ValueError(f"embeds[{name!r}] has no tokens")

        split_text = self.text_mask_ratio is not None and 'text' in names
        budget_names = tuple(n for n in names if not (split_text and n == 'text'))

        lengths = tuple(embeds[n].shape[1] for n in budget_names)
        alpha = torch.full((len(budget_names),), self.dirichlet_alpha)
        proportions = torch.distributions.Dirichlet(alpha).sample().tolist()
        counts = _visible_token_counts(
            lengths, proportions, mask_ratio_total, min_mask_ratio)
        nv = dict(zip(budget_names, counts))

        if split_text:
            L_text = embeds['text'].shape[1]
            # Same floor convention as _visible_token_counts, and never zero: a
            # text stream with no visible tokens is a different pretext task.
            # +1e-12 matches the convention in _visible_token_counts' upper
            # bound. Without it, 1.0-0.80 == 0.19999999999999996 in binary
            # float, so 25 tokens at text_mask_ratio 0.80 floors to 4 visible
            # rather than 5 — an off-by-one that is 20% of this short stream.
            nv['text'] = max(1, int(math.floor(
                L_text * (1.0 - self.text_mask_ratio) + 1e-12)))

        def _mask(x, nv_, L_):
            noise    = torch.rand(B, L_, device=x.device)
            ids_shuf = torch.argsort(noise, dim=1)
            ids_rest = torch.argsort(ids_shuf, dim=1)
            ids_keep = ids_shuf[:, :nv_]
            x_vis    = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[2]))
            mask     = torch.ones(B, L_, device=x.device)
            mask[:, :nv_] = 0
            mask     = torch.gather(mask, 1, ids_rest)
            return x_vis, mask, ids_rest

        return {n: _mask(embeds[n], nv[n], embeds[n].shape[1]) for n in names}

    # -- Encoder ---------------------------------------------------------------

    def _run_encoder(self, parts):
        """Concatenate visible tokens (canonical order) + CLS, run the trunk."""
        x   = torch.cat(parts, dim=1)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x   = torch.cat([cls, x], dim=1)
        for blk in self.encoder_blocks:
            x = blk(x)
        return self.encoder_norm(x)

    def _pack(self, per_mod, latent):
        """Pack per-modality (mask, ids_rest, n_vis) into the legacy 13-tuple.

        Inactive modalities yield None / 0 so the positional signature that
        generate_crossmodal.py:99 and train_sorghum_4m_distill.py rely on keeps
        working unchanged.
        """
        m  = tuple(per_mod[n][1] if n in per_mod else None for n in MODALITIES)
        r  = tuple(per_mod[n][2] if n in per_mod else None for n in MODALITIES)
        lv = tuple(per_mod[n][0].shape[1] if n in per_mod else 0 for n in MODALITIES)
        return (latent,) + m + r + lv

    def forward_encoder(self, x_rgb, x_depth, x_pc, x_param, mask_ratio=0.75):
        embeds = self._embed_active(x_rgb, x_depth, x_pc, x_param)
        masked = self.random_masking_dirichlet(embeds, mask_ratio)
        latent = self._run_encoder([masked[n][0] for n in self.active_modalities])
        return self._pack(masked, latent)

    # -- Encoder with explicit modality visibility (cross-modal) ---------------

    def forward_encoder_select(self, x_rgb, x_depth, x_pc, x_param, visible,
                               source_mask_ratio=0.0):
        """Encoder where each ACTIVE modality is EITHER fully visible OR fully
        masked (0 visible tokens), instead of Dirichlet token-level masking.

        `visible` : iterable subset of the active modalities. Those keep all
        their tokens; the rest contribute 0 encoder tokens and are entirely
        reconstructed by the decoder. This is the training-time analogue of
        generate_crossmodal.py's probe, and the probe path for E2 -- an arm is
        probed with visible = {rgb, depth, pc} & arm, never text, so no arm
        reads the parameter stream at probe time.

        `source_mask_ratio` : 0.0 (default) keeps the historical behaviour -- a
        visible modality is handed over WHOLE, every token, and its own
        reconstruction loss is identically zero because nothing of it is masked.
        Above 0.0, the visible modalities are additionally masked at the token
        level at that rate, so the student must infer the absent modalities from
        a PARTIAL source. This makes the cross-modal task strictly harder and
        matches how the model is probed at evaluation time (sweeps/rgb_mask_sweep.py
        blanks 0-95% of RGB patches). The masked source tokens also become real
        reconstruction targets, so the source's own loss stops being 0.

        Returns the same 13-tuple as forward_encoder. mask_* is 1 everywhere for
        a fully masked modality, 0 everywhere for a fully visible one, and a
        genuine per-token 0/1 mask for a visible modality when
        source_mask_ratio > 0. None for inactive modalities.
        """
        if not 0.0 <= float(source_mask_ratio) < 1.0:
            raise ValueError(
                f"source_mask_ratio must be in [0, 1), got {source_mask_ratio}")
        visible = set(visible)
        unknown = visible - set(self.active_modalities)
        if unknown:
            raise ValueError(
                f"visible names {sorted(unknown)} are not active modalities "
                f"{list(self.active_modalities)}")

        embeds = self._embed_active(x_rgb, x_depth, x_pc, x_param)
        first  = embeds[self.active_modalities[0]]
        B, dev = first.shape[0], first.device

        def _sel(name):
            e = embeds[name]
            L = e.shape[1]
            if name not in visible:
                ids_rest = torch.arange(L, device=dev).unsqueeze(0).expand(B, L)
                return e[:, :0], torch.ones(B, L, device=dev), ids_rest
            if source_mask_ratio <= 0.0:
                ids_rest = torch.arange(L, device=dev).unsqueeze(0).expand(B, L)
                return e, torch.zeros(B, L, device=dev), ids_rest
            # Token-level masking of a VISIBLE source. Same construction as
            # random_masking_dirichlet._mask so the two regimes agree; +1e-12
            # for the same binary-float floor reason, and at least one token
            # survives so the encoder never sees an empty source.
            n_keep = max(1, int(math.floor(L * (1.0 - source_mask_ratio) + 1e-12)))
            noise    = torch.rand(B, L, device=dev)
            ids_shuf = torch.argsort(noise, dim=1)
            ids_rest = torch.argsort(ids_shuf, dim=1)
            ids_keep = ids_shuf[:, :n_keep]
            x_vis = torch.gather(e, 1, ids_keep.unsqueeze(-1).expand(-1, -1, e.shape[2]))
            mask = torch.ones(B, L, device=dev)
            mask[:, :n_keep] = 0
            mask = torch.gather(mask, 1, ids_rest)
            return x_vis, mask, ids_rest

        per_mod = {n: _sel(n) for n in self.active_modalities}
        latent  = self._run_encoder([per_mod[n][0] for n in self.active_modalities])
        return self._pack(per_mod, latent)
    # ── Decoder ───────────────────────────────────────────────────────────────

    def forward_decoder(self, x, rr, rd, rp, rt, lr_, ld_, lp_, lt_,
                        return_features=False):
        """Restore masked positions and predict, for ACTIVE modalities only.

        The signature is positional and unchanged so generate_crossmodal.py:99
        and train_sorghum_4m_distill.py keep working; entries belonging to an
        inactive modality are ignored (pass None / 0). An inactive modality's
        prediction comes back as None.
        """
        ids_rest = {'rgb': rr,  'depth': rd,  'pc': rp,  'text': rt}
        n_vis    = {'rgb': lr_, 'depth': ld_, 'pc': lp_, 'text': lt_}
        act      = self.active_modalities

        x    = self.decoder_embed(x)
        x_nc = x[:, 1:, :]
        B    = x.shape[0]

        # Slice the encoder output back into per-modality visible blocks by
        # walking the active modalities in canonical order. This replaces the
        # old fixed rgb|depth|pc|text offsets, which assumed all four existed.
        vis = {}
        off = 0
        for name in act:
            k = n_vis[name]
            if k is None:
                raise ValueError(f"n_vis for active modality {name!r} is None")
            vis[name] = x_nc[:, off: off + k]
            off += k

        def _restore(x_vis, rest, n_tok):
            pad    = self.mask_token.repeat(B, n_tok - x_vis.shape[1], 1)
            x_full = torch.cat([x_vis, pad], dim=1)
            return torch.gather(x_full, 1,
                                rest.unsqueeze(-1).expand(-1, -1, x_full.shape[2]))

        dec_pos = {'rgb':   self.decoder_pos_embed_2d,
                   'depth': self.decoder_pos_embed_2d,
                   'pc':    self.decoder_pos_embed_pc,
                   'text':  self.decoder_pos_embed_text}

        parts = []
        for name in act:
            rest = ids_rest[name]
            if rest is None:
                raise ValueError(f"ids_rest for active modality {name!r} is None")
            parts.append(_restore(vis[name], rest, self._token_len[name])
                         + dec_pos[name])

        x = torch.cat(parts, dim=1)
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # Token-aligned decoder features (B, L_total, decoder_dim). The layout is
        # the ACTIVE modalities in canonical order and is IDENTICAL regardless of
        # which tokens were masked -- so these line up position-for-position
        # between any two forward passes of the SAME arm (e.g. a full-modal
        # teacher and a single-modal student). Used as the feature-distillation
        # target. NOTE: features are only comparable between models sharing the
        # same active_modalities, since the layout follows it.
        feats = x

        # Per-modality token spans in the decoder sequence.
        span = {}
        off = 0
        for name in act:
            L = self._token_len[name]
            span[name] = (off, off + L)
            off += L

        def _slice(name):
            a, b = span[name]
            return x[:, a:b]

        pred_rgb = pred_depth = pred_pc = pred_params = None

        if 'rgb' in act:
            pred_rgb = self.decoder_pred_rgb(_slice('rgb'))
        if 'depth' in act:
            pred_depth = self.decoder_pred_depth(_slice('depth'))
        if 'text' in act:
            pred_params = self.decoder_pred_params(_slice('text'))

        if 'pc' in act:
            x_pc_dec     = _slice('pc')
            B2, N_tok, _ = x_pc_dec.shape
            pc_feat      = self.decoder_pc_proj(x_pc_dec)
            gs  = int(np.ceil(np.sqrt(self.points_per_token)))
            g1  = torch.linspace(-1, 1, gs, device=x.device)
            gy, gx = torch.meshgrid(g1, g1, indexing='ij')
            grid = torch.stack([gx, gy], dim=-1).reshape(-1, 2)[:self.points_per_token]
            grid = grid.unsqueeze(0).unsqueeze(0).expand(B2, N_tok, -1, -1)
            pc_feat_e = pc_feat.unsqueeze(2).expand(-1, -1, self.points_per_token, -1)
            fold_in   = torch.cat([pc_feat_e, grid], dim=-1)
            pred_pc   = self.decoder_pc_fold(
                fold_in.reshape(B2 * N_tok * self.points_per_token, -1)
            ).reshape(B2, N_tok * self.points_per_token, 3)

            if pred_pc.shape[1] > self.target_points:
                pred_pc = pred_pc[:, :self.target_points]
            elif pred_pc.shape[1] < self.target_points:
                pad = self.target_points - pred_pc.shape[1]
                pred_pc = torch.cat([pred_pc, pred_pc[:, :pad]], dim=1)

        if return_features:
            return pred_rgb, pred_depth, pred_pc, pred_params, feats
        return pred_rgb, pred_depth, pred_pc, pred_params

    # -- Loss ------------------------------------------------------------------

    def forward_loss(self,
                     imgs_rgb, imgs_depth, pc, param_floats, text_valid,
                     pred_rgb, pred_depth, pred_pc, pred_params,
                     mask_rgb, mask_depth, mask_pc, mask_text):
        """Loss over ACTIVE modalities only.

        `total` sums only the active terms -- an inactive modality contributes
        nothing and is not reconstructed at all. Inactive components come back as
        a zero scalar rather than None so the 5-tuple signature and every
        downstream .item() / logging call keep working unchanged.

        IMPORTANT for the E2 ablation: because `total` sums a different number of
        terms per arm, it is NOT comparable across arms and must not be used for
        best-model selection in a cross-arm comparison. Use the per-modality
        components, or the arm-invariant Chamfer computed in evaluate().
        """
        act  = self.active_modalities
        dev  = pc.device if isinstance(pc, torch.Tensor) else imgs_rgb.device

        def zero():
            return torch.zeros((), device=dev)

        if 'rgb' in act:
            tgt_rgb = self.patchify(imgs_rgb, self.patch_size, imgs_rgb.shape[1])
            if self.norm_pix_loss:
                m = tgt_rgb.mean(-1, keepdim=True)
                v = tgt_rgb.var(-1,  keepdim=True)
                tgt_rgb = (tgt_rgb - m) / (v + 1e-6) ** .5
            l = ((pred_rgb - tgt_rgb) ** 2).mean(-1)
            # clamp(min=1): when RGB is the fully-visible cross-modal source,
            # mask_rgb is all-zero -> guard against 0/0 (yields 0 loss, no NaN).
            loss_rgb = (l * mask_rgb).sum() / mask_rgb.sum().clamp(min=1)
        else:
            loss_rgb = zero()

        if 'depth' in act:
            tgt_d = self.patchify(imgs_depth, self.patch_size, imgs_depth.shape[1])
            B_d, N_d = tgt_d.shape[0], tgt_d.shape[1]
            tf = tgt_d.reshape(B_d, -1)
            if self.depth_norm_type == 'minmax':
                t_min = tf.min(1, keepdim=True).values
                t_max = tf.max(1, keepdim=True).values
                tf = (tf - t_min) / (t_max - t_min).clamp(min=1e-6)
            elif self.depth_norm_type == 'standard':
                tf = (tf - tf.mean(1, keepdim=True)) / tf.std(1, keepdim=True).clamp(min=1e-6)
            tgt_d = tf.reshape(B_d, N_d, -1)
            l = ((pred_depth - tgt_d) ** 2).mean(-1)
            loss_depth = (l * mask_depth).sum() / mask_depth.sum().clamp(min=1)
        else:
            loss_depth = zero()

        # PC is always active -- enforced in __init__.
        if self.pc_loss_name == 'qal_loss':
            loss_pc = qal_loss(
                pred_pc, pc,
                threshold=self.qal_threshold,
                alpha=self.qal_alpha,
                use_squared=self.qal_use_squared,
            )
        else:
            loss_pc = chamfer_distance(pred_pc, pc)

        if 'text' in act:
            # Smooth-L1 (Huber) on normalised float params. All targets and
            # predictions are in [0, 1] -> well-conditioned gradients. Loss is
            # only over tokens that are (a) masked and (b) real (text_valid=1).
            B, L, P = pred_params.shape                    # P = N_PARAMS
            per_pos = F.smooth_l1_loss(
                pred_params.reshape(B * L, P),
                param_floats.reshape(B * L, P),
                reduction='none', beta=0.1,
            ).mean(-1).reshape(B, L)
            eff_mask  = mask_text * text_valid
            loss_text = (per_pos * eff_mask).sum() / eff_mask.sum().clamp(min=1)
        else:
            loss_text = zero()

        total = loss_pc * self.pc_loss_weight
        if 'rgb' in act:
            total = total + loss_rgb
        if 'depth' in act:
            total = total + loss_depth
        if 'text' in act:
            total = total + loss_text * self.spline_loss_weight

        return total, loss_rgb, loss_depth, loss_pc, loss_text
    def patchify(self, imgs, patch_size, in_chans):
        p = patch_size
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], in_chans, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        return x.reshape(imgs.shape[0], h * w, p ** 2 * in_chans)

    def decode_params_to_text(self, pred_params: torch.Tensor) -> list[list[str]]:
        """Decode predicted float params back to human-readable text.
        pred_params : (B, n_text_tokens, N_PARAMS)
        returns     : list[list[str]]  (B, n_text_tokens)
                      [] when the text modality is inactive (no param head).
        """
        if 'text' not in self.active_modalities or pred_params is None:
            return []
        p = pred_params.clamp(0, 1).detach().cpu().numpy()
        result = []
        for b in range(p.shape[0]):
            row = []
            for ti in range(p.shape[1]):
                if ti == 0:
                    row.append(_params_to_plant_text(p[b, ti]))
                else:
                    row.append(_params_to_leaf_text(p[b, ti]))
            result.append(row)
        return result

    # ── Full forward ──────────────────────────────────────────────────────────

    def forward(self, imgs_rgb, imgs_depth, pc, param_floats, text_valid,
                mask_ratio: float = 0.75, visible=None, return_features: bool = False,
                source_mask_ratio: float = 0.0):
        """
        imgs_rgb    : (B, 3, H, W)
        imgs_depth  : (B, 1, H, W)
        pc          : (B, N, 3)
        param_floats: (B, 1+max_leaves, N_PARAMS)  float32 — encoder input + target
        text_valid  : (B, 1+max_leaves)            float32 — 1=real token, 0=pad

        visible : None  → standard Dirichlet token masking (original behaviour).
                  iterable subset of {'rgb','depth','pc','text'} → cross-modal mode:
                  those modalities are fully visible, the rest fully masked and
                  reconstructed from them.
        return_features : if True, append (decoder_feats, cls_latent) to the output
                  for distillation. decoder_feats is token-aligned across masking
                  regimes; cls_latent is the encoder CLS token (B, embed_dim).
        """
        if visible is None:
            (latent,
             mr, md, mp, mt,
             rr, rd, rp, rt,
             lr_, ld_, lp_, lt_) = self.forward_encoder(
                imgs_rgb, imgs_depth, pc, param_floats, mask_ratio)
        else:
            (latent,
             mr, md, mp, mt,
             rr, rd, rp, rt,
             lr_, ld_, lp_, lt_) = self.forward_encoder_select(
                imgs_rgb, imgs_depth, pc, param_floats, visible,
                source_mask_ratio=source_mask_ratio)

        dec = self.forward_decoder(
            latent, rr, rd, rp, rt, lr_, ld_, lp_, lt_,
            return_features=return_features)
        if return_features:
            pred_rgb, pred_depth, pred_pc, pred_params, feats = dec
        else:
            pred_rgb, pred_depth, pred_pc, pred_params = dec

        total, loss_rgb, loss_depth, loss_pc, loss_text = self.forward_loss(
            imgs_rgb, imgs_depth, pc, param_floats, text_valid,
            pred_rgb, pred_depth, pred_pc, pred_params,
            mr, md, mp, mt)

        out = (total,
               (loss_rgb, loss_depth, loss_pc, loss_text),
               (pred_rgb, pred_depth, pred_pc, pred_params),
               (mr, md, mp, mt))
        if return_features:
            out = out + ((feats, latent[:, 0]),)
        return out


# ── Convenience constructors ─────────────────────────────────────────────────

def embodied_mae_4m_small(**kwargs):
    return EmbodiedMAE4M(embed_dim=384, depth=12, num_heads=6,
                         decoder_embed_dim=192, decoder_depth=4,
                         decoder_num_heads=3, **kwargs)


def embodied_mae_4m_base(**kwargs):
    return EmbodiedMAE4M(embed_dim=768, depth=12, num_heads=12,
                         decoder_embed_dim=512, decoder_depth=8,
                         decoder_num_heads=16, **kwargs)


def embodied_mae_4m_large(**kwargs):
    return EmbodiedMAE4M(embed_dim=1024, depth=24, num_heads=16,
                         decoder_embed_dim=512, decoder_depth=8,
                         decoder_num_heads=16, **kwargs)


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    B     = 2
    rgb   = torch.randn(B, 3, 224, 224, device=device)
    depth = torch.randn(B, 1, 224, 224, device=device)
    pc    = torch.randn(B, 8196, 3, device=device)

    # The four E2 arms, in the order they appear in the submission plan.
    ARMS = [
        ('PC',       ['pc']),
        ('PC+RGB',   ['pc', 'rgb']),
        ('PC+RGB+D', ['pc', 'rgb', 'depth']),
        ('+params',  ['pc', 'rgb', 'depth', 'text']),
    ]

    print(f"{'arm':<10} {'params':>13} {'vis tok':>8} {'total':>9} "
          f"{'rgb':>7} {'depth':>7} {'pc':>10} {'text':>9}")
    print('-' * 80)

    for label, mods in ARMS:
        model = embodied_mae_4m_base(
            target_points=8196,
            active_modalities=mods,
            text_mask_ratio=0.80 if 'text' in mods else None,
        ).to(device)

        n_text     = 1 + model.max_leaves
        params     = torch.rand(B, n_text, N_PARAMS, device=device)
        text_valid = torch.ones(B, n_text, device=device)
        text_valid[:, 22:] = 0

        total, (lr, ld, lp, lt), (pr, pd, pp, ppar), _ = model(
            rgb, depth, pc, params, text_valid, mask_ratio=0.80)

        # Visible tokens the encoder actually saw (excluding CLS).
        with torch.no_grad():
            emb  = model._embed_active(rgb, depth, pc, params)
            msk  = model.random_masking_dirichlet(emb, 0.80)
            ntok = sum(v[0].shape[1] for v in msk.values())

        nparam = sum(q.numel() for q in model.parameters())
        print(f"{label:<10} {nparam:>13,} {ntok:>8} {total.item():>9.4f} "
              f"{lr.item():>7.4f} {ld.item():>7.4f} {lp.item():>10.6f} {lt.item():>9.6f}")

        # An inactive modality must produce no prediction and no loss.
        for name, pred, loss in (('rgb', pr, lr), ('depth', pd, ld), ('text', ppar, lt)):
            if name in mods:
                assert pred is not None, f"{label}: {name} active but pred is None"
            else:
                assert pred is None, f"{label}: {name} inactive but pred is not None"
                assert loss.item() == 0.0, f"{label}: {name} inactive but loss={loss.item()}"
        assert pp is not None, f"{label}: pc prediction missing"

        # decode_params_to_text must degrade gracefully when text is off.
        txt = model.decode_params_to_text(ppar)
        assert (len(txt) == B) if 'text' in mods else (txt == [])

        # The cross-modal probe path must work per arm, and must reject a
        # modality the arm does not have.
        probe_visible = [m for m in ('rgb', 'depth', 'pc') if m in mods]
        model(rgb, depth, pc, params, text_valid, visible=probe_visible)
        if 'text' not in mods:
            try:
                model(rgb, depth, pc, params, text_valid, visible=['text'])
                raise AssertionError(f"{label}: visible=['text'] should have raised")
            except ValueError:
                pass

        # Backward must reach every parameter this arm actually has.
        total.backward()
        missing = [n for n, q in model.named_parameters()
                   if q.requires_grad and q.grad is None]
        assert not missing, f"{label}: no grad for {missing[:4]}"

    print('-' * 80)
    print("All four arms: built, forward+backward, gating asserted.")

    # The all-four arm must stay checkpoint-compatible with the pre-gate model.
    full = embodied_mae_4m_base(target_points=8196)
    sd   = full.state_dict()
    print(f"\nFull arm: {len(sd)} state_dict tensors, "
          f"{sum(q.numel() for q in full.parameters()):,} params "
          f"(default active_modalities={full.active_modalities})")
