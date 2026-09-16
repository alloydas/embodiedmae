"""
Structured masking and procedural occlusion for plant imagery (4M model).

Motivation (team discussion): in a field image the plant of interest sits in
the middle of the frame and is rarely covered, while the frame border is where
neighbouring plants' leaves reach in and hide it. Uniform random masking has no
notion of that. This module adds the two pieces that do:

1. ``structured_mask_scores`` -- per-token ranking scores that replace the
   i.i.d. uniform noise in ``EmbodiedMAE4M.random_masking_dirichlet``. Masked
   tokens become spatially coherent blobs (a smooth random field) biased toward
   the periphery (``center_bias``). Only WHICH tokens are masked changes; HOW
   MANY per modality still comes from the Dirichlet budget.

2. ``ProceduralOcclusion`` -- corrupts the encoder INPUT while the loss keeps
   the clean targets, so reconstruction becomes amodal completion. Occluders
   are procedural leaves that enter from outside the frame and reach toward the
   centre, so the occlusion rate falls off toward the middle by construction.
   The same leaves hide the same region of the RGB, the depth map and the point
   cloud; sensor noise is added on top.

Shared coordinate frame: "image coordinates" are (u, v) in [-1, 1] at pixel /
patch centres, u = column (right), v = row (down). The point cloud is in the
Blender camera frame (x right, y up, looking down -z) and the dataset centres
and unit-normalises it, so a point maps to the image as

    (u, v) ~= PC_TO_IMAGE_SCALE * (x, -y)

PC_TO_IMAGE_SCALE was measured on Sorghum_15K: over ten train views the ratio of
the foreground bbox extent to the normalised PC x/y extent was 0.94-1.56,
median ~1.1 (the high values are plants clipped by the frame). It is
approximate (+-20%) because the normalisation radius and camera distance vary
per plant -- good enough for "the left edge of the image is the left edge of
the cloud", not for pixel-accurate projection.

Everything draws from torch's global RNG on the input's device, so a seeded run
and the fixed-seed validation pass in train_sorghum_4m.evaluate stay
reproducible.
"""

import math
from dataclasses import dataclass, fields

import torch

PC_TO_IMAGE_SCALE = 1.1
SPATIAL_MODALITIES = ('rgb', 'depth', 'pc')

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
# Fallback leaf colour (linear [0, 1] RGB) when a sample has no foreground to
# sample from; measured off the renders, where leaves are a dark olive green.
_DEFAULT_LEAF_RGB = (0.23, 0.28, 0.16)


# ── Small helpers ─────────────────────────────────────────────────────────────

def _as_range(value, name, lo_bound=-math.inf, hi_bound=math.inf):
    """Accept a scalar or a [lo, hi] pair -> (lo, hi) floats, validated."""
    if isinstance(value, (int, float)):
        lo = hi = float(value)
    else:
        try:
            lo, hi = (float(v) for v in value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a number or a [min, max] pair") from exc
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo > hi:
        raise ValueError(f"{name} must be finite with min <= max, got {value!r}")
    if lo < lo_bound or hi > hi_bound:
        raise ValueError(f"{name} must lie in [{lo_bound}, {hi_bound}], got {value!r}")
    return lo, hi


def _uniform(lo, hi, shape, device):
    return lo + (hi - lo) * torch.rand(shape, device=device)


def image_grid(n, device=None):
    """(n*n, 2) image coordinates of an n x n grid's cell centres, row-major.

    Row-major (row * n + col) is the token order of PatchEmbed / patchify.
    """
    c = (torch.arange(n, dtype=torch.float32, device=device) + 0.5) / n * 2 - 1
    v, u = torch.meshgrid(c, c, indexing='ij')
    return torch.stack([u, v], dim=-1).reshape(-1, 2)


def pc_to_image(xyz, scale=PC_TO_IMAGE_SCALE):
    """(..., 3) normalised camera-frame points -> (..., 2) image coordinates."""
    return torch.stack([xyz[..., 0], -xyz[..., 1]], dim=-1) * scale


# ── 1. Structured token masking ───────────────────────────────────────────────

@dataclass(frozen=True)
class StructuredMaskConfig:
    """Config for occlusion-like token masking (``model.structured_mask``).

    prob          : per-sample probability that a TRAINING sample uses the
                    structured ranking; the rest keep uniform random masking so
                    the model still sees the standard pretext task.
    center_bias   : how much more the periphery is masked than the centre, in
                    units of the random field's std. 0 = coherent blobs with no
                    radial prior; negative masks the centre more.
    length_scale  : correlation length of the field in image coordinates
                    ([-1, 1] across the frame). Smaller = finer blobs.
    modalities    : which spatial streams use it (text is never spatial).
    shared_field  : one field per sample for all listed streams, so an
                    "occluder" hides the same region of RGB, depth and cloud.
                    The visible sets then nest across modalities -- faithful to
                    real occlusion, but it withholds the cross-modal hint of
                    "depth sees what RGB lost" on those samples.
    apply_in_eval : also use it in model.eval(). Off by default so validation
                    stays on the uniform regime and comparable across runs.
    """
    prob: float = 0.5
    center_bias: float = 1.5
    length_scale: float = 0.35
    modalities: tuple = SPATIAL_MODALITIES
    shared_field: bool = True
    apply_in_eval: bool = False

    @classmethod
    def from_dict(cls, cfg):
        """None / {} / {'enabled': False} -> None (feature off)."""
        if not cfg:
            return None
        cfg = dict(cfg)
        if not cfg.pop('enabled', True):
            return None
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(cfg) - known)
        if unknown:
            raise ValueError(f"unknown structured_mask keys {unknown}; expected {sorted(known)}")
        if 'modalities' in cfg:
            cfg['modalities'] = tuple(str(m).lower() for m in cfg['modalities'])
        out = cls(**cfg)
        if not 0.0 <= float(out.prob) <= 1.0:
            raise ValueError("structured_mask.prob must be in [0, 1]")
        if not math.isfinite(float(out.center_bias)):
            raise ValueError("structured_mask.center_bias must be finite")
        if not (math.isfinite(float(out.length_scale)) and out.length_scale > 0):
            raise ValueError("structured_mask.length_scale must be positive")
        bad = [m for m in out.modalities if m not in SPATIAL_MODALITIES]
        if bad or not out.modalities:
            raise ValueError(
                f"structured_mask.modalities must be a non-empty subset of "
                f"{list(SPATIAL_MODALITIES)}, got {list(out.modalities)}")
        return out


def sample_field(batch_size, length_scale, device, num_features=32):
    """Draw one smooth 2-D random field per batch element.

    Random Fourier features: an approximate Gaussian process with an RBF kernel
    of the given length scale. Evaluate with ``eval_field``; drawing once and
    evaluating at several token sets is what ``shared_field`` means.
    """
    omega = torch.randn(batch_size, 2, num_features, device=device) / length_scale
    phase = 2 * math.pi * torch.rand(batch_size, 1, num_features, device=device)
    return omega, phase


def eval_field(coords, field):
    """(B, L, 2) coordinates -> (B, L) zero-mean, unit-variance field values."""
    omega, phase = field
    return math.sqrt(2.0 / omega.shape[-1]) * torch.cos(coords @ omega + phase).sum(-1)


def structured_mask_scores(coords, cfg, field):
    """Masking rank scores for tokens at image coordinates ``coords`` (B, L, 2).

    Lower score = kept visible (random_masking_dirichlet keeps the lowest-noise
    tokens). score = center_bias * radius + field, with the radius normalised
    so the frame corners are 1.
    """
    radius = coords.norm(dim=-1) / math.sqrt(2.0)
    return cfg.center_bias * radius + eval_field(coords, field)


# ── 2. Procedural occlusion + sensor noise ────────────────────────────────────

def _sample_leaves(n, device, *, reach, width, bend, jitter, segments, border=1.1):
    """Sample ``n`` leaf centrelines entering the frame [-1, 1]^2 from outside.

    The base sits on a square of half-size ``border`` (just outside the frame)
    and the leaf heads toward the centre +- ``jitter`` radians, curving as a
    quadratic Bezier. A leaf's tip lands at roughly border - reach from the
    centre, so how deep occluders get is set by ``reach``.

    Returns curve (n, segments + 1, 2) and max half-width (n,).
    """
    theta = 2 * math.pi * torch.rand(n, device=device)
    direction = torch.stack([theta.cos(), theta.sin()], dim=-1)
    base = direction * (border / direction.abs().max(dim=-1, keepdim=True).values)

    heading = theta + math.pi + _uniform(-jitter, jitter, (n,), device)
    length = _uniform(*reach, (n,), device)
    step = torch.stack([heading.cos(), heading.sin()], dim=-1)
    tip = base + length[:, None] * step
    normal = torch.stack([-step[:, 1], step[:, 0]], dim=-1)
    ctrl = (0.5 * (base + tip)
            + normal * (_uniform(-bend, bend, (n,), device) * length)[:, None])

    t = torch.linspace(0, 1, segments + 1, device=device)[None, :, None]
    curve = ((1 - t) ** 2 * base[:, None] + 2 * (1 - t) * t * ctrl[:, None]
             + t ** 2 * tip[:, None])
    return curve, _uniform(*width, (n,), device)


def _distance_to_curve(points, curve):
    """points (n, M, 2), curve (n, K+1, 2) -> (distance (n, M), t (n, M)).

    Exact distance to the polyline; t in [0, 1] is the arc parameter (by
    segment index) of the nearest point, 0 at the base and 1 at the tip.
    """
    a = curve[:, :-1]
    ab = curve[:, 1:] - a
    ap = points[:, :, None, :] - a[:, None]
    s = ((ap * ab[:, None]).sum(-1)
         / ab.pow(2).sum(-1).clamp(min=1e-12)[:, None]).clamp(0, 1)
    d2 = (ap - s[..., None] * ab[:, None]).pow(2).sum(-1)
    d2_min, k = d2.min(dim=-1)
    s_k = s.gather(-1, k[..., None]).squeeze(-1)
    return d2_min.sqrt(), (k.to(s.dtype) + s_k) / ab.shape[1]


def _leaf_half_width(half_width, t):
    """Full width where the leaf enters the frame, tapering to a point at the tip."""
    return half_width[:, None] * (1 - t.pow(2)).clamp(min=0).sqrt()


class ProceduralOcclusion:
    """Occlude + add noise to a batch of encoder inputs; targets are untouched.

    Usage (training loop):
        rgb_in, depth_in, pc_in, info = occluder(rgb, depth, pc)
        model(rgb, depth, pc, ..., input_rgb=rgb_in, input_depth=depth_in,
              input_pc=pc_in)

    Parameters (``occlusion:`` section of the YAML)
        prob             per-sample probability of getting occluding leaves.
        num_leaves       [min, max] leaves on an occluded sample.
        reach            [min, max] leaf length in image coordinates; with the
                         base at 1.1, a 1.0 leaf reaches 0.1 from the centre.
                         This is the knob for the centre/border contrast.
        width            [min, max] max half-width in image coordinates
                         (0.1 ~ 22 px wide at 224 px).
        bend             max sideways Bezier offset as a fraction of length.
        heading_jitter_deg  angular spread around "pointing at the centre".
        depth_scale      [min, max] occluder depth as a multiple of the plant's
                         nearest (5th percentile) foreground depth; < 1 puts
                         it in front, as an occluder must be. Leaves then
                         overwrite everything under them.
        depth_quantile   [min, max] in [0, 1], or None (default: use
                         depth_scale). Places each leaf INSIDE the plant's own
                         depth range -- at that quantile of its foreground
                         depth, tilted along its length by up to half the
                         plant's depth spread -- and z-tests it per pixel: the
                         leaf covers the plant only where it is nearer, so
                         neighbour leaves interleave with the plant instead of
                         always floating in front. The point cloud uses the
                         same quantile of its own camera depth, so only points
                         behind the leaf are hidden. Overrides depth_scale.
        rgb_noise_std    max per-sample Gaussian pixel noise, in [0, 1] units.
        depth_noise_std  max per-sample multiplicative noise on foreground depth.
        pc_noise_std     max per-sample Gaussian jitter, unit-sphere units.
        occlude_pc       drop the points behind the leaves (camera x-y
                         silhouette, i.e. an orthographic view). Dropped slots
                         are refilled with duplicates of visible points, the
                         same padding rule the dataset loader uses.
        min_pc_keep      never leave fewer than this fraction of the cloud; a
                         sample whose leaves would hide more keeps its cloud.
        pc_to_image_scale  see module docstring.
        eval_occluded    (read by the training script) also run a second,
                         occluded validation pass and log it under *_occ.

    Noise levels are drawn per sample from U(0, max) so clean-ish inputs stay
    in the mix, and apply to every sample, occluded or not.
    """

    _KEYS = ('prob', 'num_leaves', 'reach', 'width', 'bend', 'heading_jitter_deg',
             'depth_scale', 'depth_quantile', 'rgb_noise_std', 'depth_noise_std',
             'pc_noise_std', 'occlude_pc', 'min_pc_keep', 'pc_to_image_scale',
             'segments')

    # Defaults were tuned on 16 Sorghum_15K val views x 8 draws (all occluded):
    # 8% of the plant covered on average (p90 16%), but 1.8% of plant pixels
    # within r < 0.3 of the centre vs 21% beyond r > 0.6. Longer leaves
    # (reach up to 1.5) add coverage but erase that centre/border contrast.
    def __init__(self, prob=0.5, num_leaves=(4, 10), reach=(0.5, 1.0),
                 width=(0.06, 0.14), bend=0.25, heading_jitter_deg=35.0,
                 depth_scale=(0.8, 0.95), depth_quantile=None, rgb_noise_std=0.02,
                 depth_noise_std=0.01, pc_noise_std=0.005, occlude_pc=True,
                 min_pc_keep=0.5, pc_to_image_scale=PC_TO_IMAGE_SCALE,
                 segments=12):
        self.prob = float(prob)
        if not 0.0 <= self.prob <= 1.0:
            raise ValueError("occlusion.prob must be in [0, 1]")
        lo, hi = _as_range(num_leaves, 'occlusion.num_leaves', 0)
        if lo != int(lo) or hi != int(hi):
            raise ValueError("occlusion.num_leaves must be integers")
        self.num_leaves = (int(lo), int(hi))
        self.reach = _as_range(reach, 'occlusion.reach', 0)
        self.width = _as_range(width, 'occlusion.width', 0)
        self.bend = float(bend)
        self.heading_jitter_deg = float(heading_jitter_deg)
        self.jitter = math.radians(self.heading_jitter_deg)
        self.depth_scale = _as_range(depth_scale, 'occlusion.depth_scale', 0)
        self.depth_quantile = (None if depth_quantile is None else
                               _as_range(depth_quantile, 'occlusion.depth_quantile', 0, 1))
        self.rgb_noise_std = float(rgb_noise_std)
        self.depth_noise_std = float(depth_noise_std)
        self.pc_noise_std = float(pc_noise_std)
        for name in ('bend', 'rgb_noise_std', 'depth_noise_std', 'pc_noise_std'):
            value = getattr(self, name)
            if not (math.isfinite(value) and value >= 0):
                raise ValueError(f"occlusion.{name} must be finite and non-negative")
        self.occlude_pc = bool(occlude_pc)
        self.min_pc_keep = float(min_pc_keep)
        if not 0.0 <= self.min_pc_keep <= 1.0:
            raise ValueError("occlusion.min_pc_keep must be in [0, 1]")
        self.pc_to_image_scale = float(pc_to_image_scale)
        if not self.pc_to_image_scale > 0:
            raise ValueError("occlusion.pc_to_image_scale must be positive")
        self.segments = int(segments)
        if self.segments < 1:
            raise ValueError("occlusion.segments must be at least 1")

    @classmethod
    def from_config(cls, cfg):
        """None / {} / {'enabled': False} -> None (feature off)."""
        if not cfg:
            return None
        cfg = dict(cfg)
        if not cfg.pop('enabled', True):
            return None
        cfg.pop('eval_occluded', None)          # consumed by the training script
        unknown = sorted(set(cfg) - set(cls._KEYS))
        if unknown:
            raise ValueError(f"unknown occlusion keys {unknown}; expected {sorted(cls._KEYS)}")
        return cls(**cfg)

    def __repr__(self):
        body = ', '.join(f"{k}={getattr(self, k)!r}" for k in self._KEYS)
        return f"ProceduralOcclusion({body})"

    # -- per-batch sampling ----------------------------------------------------

    def _sample_batch_leaves(self, B, device):
        """Leaf slots for the batch: curves (B, J, K+1, 2), widths, active (B, J)."""
        occluded = torch.rand(B, device=device) < self.prob
        lo, hi = self.num_leaves
        J = hi
        count = torch.randint(lo, hi + 1, (B,), device=device)
        count = torch.where(occluded, count, torch.zeros_like(count))
        active = torch.arange(J, device=device)[None] < count[:, None]
        if J == 0:
            return None, None, active, occluded
        curve, half_width = _sample_leaves(
            B * J, device, reach=self.reach, width=self.width, bend=self.bend,
            jitter=self.jitter, segments=self.segments)
        return (curve.reshape(B, J, -1, 2), half_width.reshape(B, J),
                active, occluded & (count > 0))

    @torch.no_grad()
    def __call__(self, rgb, depth, pc):
        """rgb (B,3,H,W) ImageNet-normalised, depth (B,1,H,W) in [0,1] with 0 =
        background, pc (B,N,3) unit-normalised. Returns corrupted copies plus an
        info dict of per-sample tensors: 'occluded' flag, 'coverage' (fraction
        of pixels under a leaf), 'plant_coverage' (fraction of the plant's
        foreground pixels under a leaf), 'pc_dropped' (fraction of points
        removed), plus the 'alpha' (B,1,H,W) occluder map and the 'pc_hidden'
        (B,N) mask of removed points."""
        B, _, H, W = rgb.shape
        if H != W:
            raise ValueError("ProceduralOcclusion expects square images")
        device = rgb.device
        fg = depth[:, 0] > 0
        mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
        rgb01 = (rgb * std + mean).clamp(0, 1)
        depth_in = depth.clone()
        pc_in = pc.clone()

        curves, half_widths, active, occluded = self._sample_batch_leaves(B, device)
        cover = torch.zeros(B, H * W, device=device)          # union alpha, for info
        pc_hidden = torch.zeros(B, pc.shape[1], dtype=torch.bool, device=device)

        if curves is not None and bool(active.any()):
            J = curves.shape[1]
            # Leaf colour: the sample's own median foreground colour, so the
            # occluder matches the render's lighting and is not trivially
            # separable from the plant by hue -- real neighbour leaves aren't.
            fg_flat = fg.reshape(B, 1, -1)
            px = rgb01.reshape(B, 3, -1).masked_fill(~fg_flat, float('nan'))
            base_col = px.nanmedian(dim=-1).values
            default = torch.tensor(_DEFAULT_LEAF_RGB, device=device).expand(B, 3)
            base_col = torch.where(base_col.isnan(), default, base_col)
            # Plant depth statistics from SOLID foreground only. Resizing to
            # 224 px blends every silhouette pixel with the 0 background, and
            # thin leaves are mostly silhouette, so the raw 5th percentile sat
            # near 0 and put "in front" leaves at ~1% of the plant's depth --
            # indistinguishable from background in the depth input. Real plant
            # depth spans well under 2x its median, so half the median cleanly
            # separates solid pixels from blended ones.
            flat_depth = depth[:, 0].reshape(B, -1)
            raw_fg = flat_depth.masked_fill(~fg.reshape(B, -1), float('nan'))
            # No foreground at all: 1.0 (the depth ceiling) marks every pixel
            # empty, so leaves still draw and the tables fall back to 0.05.
            median = torch.nan_to_num(raw_fg.nanmedian(dim=-1).values, nan=1.0)
            solid = fg.reshape(B, -1) & (flat_depth > 0.5 * median[:, None])
            fg_depth = flat_depth.masked_fill(~solid, float('nan'))
            interleave = self.depth_quantile is not None
            if interleave:
                # Depth lookup tables at 1% steps: (101, B), image and cloud.
                # The cloud's camera depth is -z (Blender cameras look down -z).
                levels = torch.linspace(0, 1, 101, device=device)
                d_tab = torch.nan_to_num(
                    torch.nanquantile(fg_depth, levels, dim=-1), nan=0.05)
                d_spread = d_tab[95] - d_tab[5]
                p_tab = torch.quantile(-pc[..., 2], levels, dim=1)
                p_spread = p_tab[95] - p_tab[5]
                # Painter's order: far leaves first, so nearer ones end on top.
                q = _uniform(*self.depth_quantile, (B, J), device).sort(
                    dim=1, descending=True).values
                q_idx = (q * 100).round().long()
            else:
                near = torch.nan_to_num(
                    torch.nanquantile(fg_depth, 0.05, dim=-1), nan=0.05)
                # Painter's order: later slots are drawn on top, so make them nearer.
                depth_fac = _uniform(*self.depth_scale, (B, J), device).sort(
                    dim=1, descending=True).values

            grid = image_grid(H, device)
            pixel = 2.0 / H
            rgb_flat = rgb01.reshape(B, 3, -1)
            depth_flat = depth_in.reshape(B, -1)
            pc_uv = pc_to_image(pc, self.pc_to_image_scale)

            for j in range(J):
                on = active[:, j]
                if not bool(on.any()):
                    continue
                curve, hw = curves[:, j], half_widths[:, j]

                dist, t = _distance_to_curve(grid.expand(B, -1, -1), curve)
                w = _leaf_half_width(hw, t)
                alpha = ((w - dist) / pixel + 0.5).clamp(0, 1) * on[:, None]
                if interleave:
                    tilt = _uniform(-0.5, 0.5, (B, 1), device)
                    base = d_tab.gather(0, q_idx[None, :, j]).squeeze(0)
                    leaf_depth = base[:, None] + tilt * (t - 0.5) * d_spread[:, None]
                    # z-test: only where the leaf is nearer than what is there
                    # (plant or an earlier leaf). Background and the blended
                    # silhouette pixels (< half the median) count as empty, so
                    # the plant's anti-aliased outline doesn't poke through.
                    empty = depth_flat < 0.5 * median[:, None]
                    alpha = alpha * (empty | (leaf_depth < depth_flat))
                inside = (dist / w.clamp(min=1e-6)).clamp(max=1)
                shade = 0.8 + 0.2 * (1 - inside.pow(2))           # darker at the rim
                brightness = _uniform(0.7, 1.3, (B, 1), device)
                col = (base_col[:, :, None] * (shade * brightness)[:, None]).clamp(0, 1)
                midrib = (inside < 0.12).to(col.dtype)[:, None]    # pale central vein
                col = col + midrib * 0.5 * ((col * 2.2).clamp(max=1) - col)
                rgb_flat = rgb_flat * (1 - alpha[:, None]) + col * alpha[:, None]

                hard = alpha > 0.5
                if not interleave:
                    slope = _uniform(-0.05, 0.05, (B, 1), device)
                    leaf_depth = (near * depth_fac[:, j])[:, None] * (1 + slope * (t - 0.5))
                depth_flat = torch.where(hard, leaf_depth, depth_flat)
                cover = torch.maximum(cover, alpha)

                if self.occlude_pc:
                    d_pc, t_pc = _distance_to_curve(pc_uv, curve)
                    under = (d_pc < _leaf_half_width(hw, t_pc)) & on[:, None]
                    if interleave:
                        p_base = p_tab.gather(0, q_idx[None, :, j]).squeeze(0)
                        p_leaf = p_base[:, None] + tilt * (t_pc - 0.5) * p_spread[:, None]
                        under &= -pc[..., 2] > p_leaf        # behind the leaf only
                    pc_hidden |= under

            rgb01 = rgb_flat.reshape(B, 3, H, W)
            depth_in = depth_flat.reshape(B, 1, H, W)

        # Refill hidden points with duplicates of visible ones, unless that
        # would leave less than min_pc_keep of the cloud.
        pc_dropped = torch.zeros(B, device=device)
        if self.occlude_pc and bool(pc_hidden.any()):
            keep = ~pc_hidden
            ok = keep.float().mean(dim=1) >= max(self.min_pc_keep, 1e-6)
            pc_hidden &= ok[:, None]
            keep = ~pc_hidden
            if bool(pc_hidden.any()):
                fill = torch.multinomial(keep.float(), pc.shape[1], replacement=True)
                refill = pc.gather(1, fill[..., None].expand(-1, -1, 3))
                pc_in = torch.where(pc_hidden[..., None], refill, pc_in)
            pc_dropped = pc_hidden.float().mean(dim=1)

        # Sensor noise, strength ~ U(0, max) per sample.
        if self.rgb_noise_std > 0:
            s = _uniform(0, self.rgb_noise_std, (B, 1, 1, 1), device)
            rgb01 = (rgb01 + s * torch.randn_like(rgb01)).clamp(0, 1)
        if self.depth_noise_std > 0:
            s = _uniform(0, self.depth_noise_std, (B, 1, 1, 1), device)
            fg_in = depth_in > 0
            depth_in = torch.where(
                fg_in, depth_in * (1 + s * torch.randn_like(depth_in)), depth_in
            ).clamp(0, 1)
        if self.pc_noise_std > 0:
            s = _uniform(0, self.pc_noise_std, (B, 1, 1), device)
            pc_in = pc_in + s * torch.randn_like(pc_in)

        rgb_in = (rgb01 - mean) / std
        fg_flat = fg.reshape(B, -1).float()
        info = {'occluded': occluded,
                'coverage': cover.mean(dim=1),
                'plant_coverage': (cover * fg_flat).sum(1) / fg_flat.sum(1).clamp(min=1),
                'pc_dropped': pc_dropped,
                'alpha': cover.reshape(B, 1, H, W),      # occluder opacity map
                'pc_hidden': pc_hidden}                  # (B, N) points removed
        return rgb_in, depth_in, pc_in, info
