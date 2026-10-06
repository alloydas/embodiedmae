"""
Neighbour-occlusion structured masking for the 4M model (sorghum renders).

Why (lab meetings 2026-09-29): the target plant stands in a row, and what hides
it in the field is its neighbours' leaves, which reach in from the sides, so the
middle of the plant is seen far more often than its outer leaves. Uniform token
masking knows nothing about that. An earlier attempt (the `yongyun` branch's
frame-edge blob field) lost to uniform masking on every PC metric, including
under occlusion, and three things about its design looked responsible:

  1. its bias was toward the FRAME border, which in these renders is background
     (the plant covers 7-17 % of the frame), so much of the mask budget hid
     empty patches;
  2. one field hid the same region in RGB, depth and the cloud on every
     structured sample, including all the background;
  3. leaf tips were masked ~85 % of the time, so the encoder rarely saw them.

This module replaces the synthetic field with a physical occluder. Each sample
borrows another plant from the same batch as its neighbour and places it beside
the target, overlapping the target's side leaves, at a random depth in front of
or behind it. A per-pixel depth test decides which target pixels and which
target points that neighbour would hide; the tokens holding them are masked
FIRST, and the rest of each modality's Dirichlet budget is filled uniformly at
random, independently per modality. So:

  * only plant tokens the neighbour actually hides are forced (fixes 1);
  * the forced set is shared across modalities -- an occluder hides the same
    surface from every camera-derived sensor -- but everything else stays
    independent, so the cross-modal hint survives (fixes 2);
  * the occluder overlaps the side leaves by a random amount and leaves
    interleave in depth, so the outer plant is hidden often but not always
    (fixes 3).

HOW MANY tokens each modality keeps is untouched: the Dirichlet allocation in
EmbodiedMAE4M.random_masking_dirichlet still decides that. Only WHICH tokens
are masked changes, and only on the `prob` fraction of training samples.

Geometry. The renders come from Alloy's capture.py: a pyrender perspective
camera, vertical FOV 40 deg, square image, looking down -z, with depth.png
storing (z - near) / (far - near) for near = 0.025 m and far = 50 m. The
`*_nc_cam.ply` cloud is in that camera frame, in metres, so a point projects as

    u = x / (-z tan 20deg),   v = y / (-z tan 20deg),   col = (u+1)/2 W,  row = (1-v)/2 H.

Checked on 24 test views (6 plants x 4 views): 92-99 % of points land on
foreground pixels, and point depth sits at or behind the depth map's front
surface (5th percentile -7 mm). The dataset centres and unit-scales the cloud,
so projecting FPS centres needs the per-sample (centroid, scale) that
SorghumDataset4M(return_pc_norm=True) returns.

The neighbour is a billboard: its own render (depth, and RGB for test-time
compositing), flipped at random, translated in the image, and shifted in depth
by `depth_offset` metres. That ignores the small change of perspective from
moving it sideways, which is fine at token resolution (16 px patches).
"""

import math
from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F

# Camera of the Sorghum_15K renders (alloy/shorgum_data/capture.py).
RENDER_FOV_DEG = 40.0
RENDER_NEAR = 0.025          # CAMERA_RADIUS * 0.01
RENDER_FAR = 50.0            # CAMERA_RADIUS * 20
_TAN_HALF_FOV = math.tan(math.radians(RENDER_FOV_DEG) / 2)

# depth.png is 0 on background and >= ~0.029 on any plant (closest point
# ~1.47 m). Bilinear downsizing to 224 smears the silhouette edge toward 0;
# 0.02 keeps those half-background edge pixels out of the foreground.
FG_THRESHOLD = 0.02


def _pair(value, name):
    try:
        lo, hi = (float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"structured_mask.{name} must be a [min, max] pair") from exc
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo > hi:
        raise ValueError(f"structured_mask.{name} must be finite with min <= max")
    return lo, hi


@dataclass(frozen=True)
class NeighbourMaskConfig:
    """`model.structured_mask` in the YAML. Absent / null / enabled: false = off.

    prob               : fraction of training samples whose mask is neighbour-
                         structured; the rest keep plain uniform masking.
    neighbours         : 1 = one neighbour on a random side; 2 = one each side.
    offset             : neighbour centre distance from the target centre, as a
                         fraction of (target half-width + neighbour half-width)
                         of their silhouettes. 1 = silhouettes just touch;
                         smaller = more overlap.
    depth_offset       : metres added to the neighbour's depth. Negative puts it
                         in front of the target. A range straddling 0 makes the
                         two plants' leaves interleave.
    vertical_jitter    : bottoms are aligned (same ground), then moved by up to
                         this fraction of the target's silhouette height.
    flip               : mirror the neighbour left-right half the time.
    patch_hidden_frac  : an RGB/depth patch counts as occluded when at least this
                         fraction of its PLANT pixels is hidden.
    min_patch_plant    : patches with less plant than this fraction of their
                         area are background and are never forced.
    apply_in_eval      : also structure the mask in model.eval(). Off, so
                         validation stays on the uniform regime every arm shares.

    The defaults were tuned on 128 train views x 4 draws (prob 1, two
    neighbours). Sorghum is sparse, so overlapping silhouettes rarely overlap
    leaves: offset [0.3, 0.9] with depth [-0.3, 0.3] hid a median 4 % of the
    plant. These defaults hide a median 20 % of plant pixels (10th-90th
    percentile 5-41 %) and 27 % of cloud points, i.e. ~15 % of plant patches.
    """
    prob: float = 0.5
    neighbours: int = 2
    offset: tuple = (0.0, 0.7)
    depth_offset: tuple = (-0.5, 0.1)
    vertical_jitter: float = 0.1
    flip: bool = True
    patch_hidden_frac: float = 0.4
    min_patch_plant: float = 0.05
    apply_in_eval: bool = False

    @classmethod
    def from_dict(cls, cfg):
        """None / {} / {'enabled': False} -> None; anything else is validated."""
        if isinstance(cfg, cls):
            return cfg
        if not cfg:
            return None
        cfg = dict(cfg)
        if not cfg.pop('enabled', True):
            return None
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(cfg) - known)
        if unknown:
            raise ValueError(
                f"unknown structured_mask keys {unknown}; expected {sorted(known)}")
        for key in ('offset', 'depth_offset'):
            if key in cfg:
                cfg[key] = _pair(cfg[key], key)
        out = cls(**cfg)
        if not 0.0 <= float(out.prob) <= 1.0:
            raise ValueError("structured_mask.prob must be in [0, 1]")
        if int(out.neighbours) not in (1, 2):
            raise ValueError("structured_mask.neighbours must be 1 or 2")
        if out.offset[0] < 0:
            raise ValueError("structured_mask.offset must be non-negative")
        for key in ('patch_hidden_frac', 'min_patch_plant', 'vertical_jitter'):
            if not 0.0 <= float(getattr(out, key)) <= 1.0:
                raise ValueError(f"structured_mask.{key} must be in [0, 1]")
        return out

    def to_dict(self):
        return {f.name: (list(v) if isinstance(v := getattr(self, f.name), tuple) else v)
                for f in fields(self)}


# ── Random draws ──────────────────────────────────────────────────────────────

def draw_params(n, cfg, device, generator=None):
    """Per-sample placement randomness, (n,) / (n, K) tensors on `device`.

    Kept separate from the geometry so a test set can draw it once with a fixed
    generator and every model is scored on identical occluders. `generator`
    must live on `device` when given; None uses the global RNG (training).
    """
    K = int(cfg.neighbours)

    def rand(*shape):
        return torch.rand(*shape, generator=generator, device=device)

    lo, hi = cfg.offset
    dlo, dhi = cfg.depth_offset
    use = rand(n) < cfg.prob
    side0 = torch.where(rand(n) < 0.5, -1.0, 1.0)
    # K=1: one neighbour on a random side. K=2: one on each side.
    side = torch.stack([side0 * (-1.0) ** k for k in range(K)], dim=1)
    return {
        'use': use,
        'side': side,
        'offset': lo + (hi - lo) * rand(n, K),
        'depth': dlo + (dhi - dlo) * rand(n, K),
        'jitter': (2 * rand(n, K) - 1) * cfg.vertical_jitter,
        'flip': (rand(n, K) < 0.5) if cfg.flip else torch.zeros(
            n, K, dtype=torch.bool, device=device),
    }


# ── Geometry ──────────────────────────────────────────────────────────────────

def foreground(depth):
    """(B, 1, H, W) depth.png values -> bool plant mask."""
    return depth > FG_THRESHOLD


def metric_depth(depth):
    """depth.png values -> z-distance in metres (meaningful on foreground only)."""
    return RENDER_NEAR + depth * (RENDER_FAR - RENDER_NEAR)


def silhouette_bbox(fg):
    """(B, 1, H, W) bool -> (B, 4) [x0, x1, y0, y1] in grid_sample coordinates.

    Pixel centres at (2i + 1) / W - 1 (align_corners=False). An empty mask gets
    the full frame, so a degenerate sample degrades to "no useful occluder"
    rather than NaN.
    """
    B, _, H, W = fg.shape
    cols = fg.any(dim=2).squeeze(1)            # (B, W)
    rows = fg.any(dim=3).squeeze(1)            # (B, H)

    def span(any_, n):
        idx = torch.arange(n, device=fg.device)
        has = any_.any(dim=1)
        first = torch.where(any_, idx, n).min(dim=1).values
        last = torch.where(any_, idx, -1).max(dim=1).values
        first = torch.where(has, first, torch.zeros_like(first))
        last = torch.where(has, last, torch.full_like(last, n - 1))
        to_norm = lambda i: (2 * i.float() + 1) / n - 1
        return to_norm(first), to_norm(last)

    x0, x1 = span(cols, W)
    y0, y1 = span(rows, H)
    return torch.stack([x0, x1, y0, y1], dim=1)


def placement_theta(tgt_box, nb_box, side, offset, jitter, flip):
    """Affine (B, 2, 3) mapping output (target) coords -> neighbour image coords.

    The neighbour's silhouette centre lands at
        target centre + side * offset * (target half-width + neighbour half-width)
    and its silhouette bottom on the target's (same ground), moved by `jitter`
    target heights. `flip` mirrors the neighbour about the image's vertical axis.
    """
    tx0, tx1, ty0, ty1 = tgt_box.unbind(1)
    nx0, nx1, ny0, ny1 = nb_box.unbind(1)
    sx = torch.where(flip, -1.0, 1.0)
    cx_n = (nx0 + nx1) / 2
    w_n = (nx1 - nx0) / 2
    cx_t = (tx0 + tx1) / 2
    w_t = (tx1 - tx0) / 2
    # Mirrored, the neighbour's centre is at -cx_n; either way the sampling
    # point for output x is sx * x + t with t chosen so the output centre d
    # reads the neighbour's own centre.
    d = cx_t + side * offset * (w_t + w_n)
    t_x = cx_n - sx * d
    t_y = ny1 - (ty1 + jitter * (ty1 - ty0))
    theta = torch.zeros(tgt_box.shape[0], 2, 3, device=tgt_box.device)
    theta[:, 0, 0] = sx
    theta[:, 0, 2] = t_x
    theta[:, 1, 1] = 1.0
    theta[:, 1, 2] = t_y
    return theta


def warp(img, theta):
    """Resample `img` (B, C, H, W) through `theta`; empty where it leaves the frame."""
    grid = F.affine_grid(theta, list(img.shape), align_corners=False)
    return F.grid_sample(img, grid, mode='nearest', padding_mode='zeros',
                         align_corners=False)


def occluders(tgt_depth, nb_depths, params, nb_rgbs=None):
    """Place the neighbours and z-test them against the target.

    tgt_depth : (B, 1, H, W) target depth.png values
    nb_depths : list of K (B, 1, H, W) neighbour depth maps (raw, unplaced)
    params    : draw_params output for these B samples
    nb_rgbs   : optional list of K (B, 3, H, W) neighbour images to place too

    Returns a dict:
      occ_z  (B, 1, H, W)  nearest neighbour z in metres, +inf where none
      hidden (B, 1, H, W)  target plant pixels a neighbour is in front of
      shown  (B, 1, H, W)  pixels where the composite shows a neighbour
      tgt_fg (B, 1, H, W)  the target's own silhouette
      rgb    (B, 3, H, W)  nearest neighbour's colour (only if nb_rgbs given)
    Samples with params['use'] False get no neighbour at all.
    """
    tgt_fg = foreground(tgt_depth)
    tgt_box = silhouette_bbox(tgt_fg)
    inf = torch.tensor(float('inf'), device=tgt_depth.device)
    occ_z = torch.full_like(tgt_depth, float('inf'))
    rgb = None if nb_rgbs is None else torch.zeros_like(nb_rgbs[0])
    for k, nb_depth in enumerate(nb_depths):
        theta = placement_theta(
            tgt_box, silhouette_bbox(foreground(nb_depth)),
            params['side'][:, k], params['offset'][:, k],
            params['jitter'][:, k], params['flip'][:, k])
        placed = warp(nb_depth, theta)
        z = torch.where(foreground(placed),
                        metric_depth(placed) + params['depth'][:, k].view(-1, 1, 1, 1),
                        inf)
        z = torch.where(params['use'].view(-1, 1, 1, 1), z, inf)
        nearer = z < occ_z
        occ_z = torch.where(nearer, z, occ_z)
        if rgb is not None:
            rgb = torch.where(nearer, warp(nb_rgbs[k], theta), rgb)
    tgt_z = metric_depth(tgt_depth)
    hidden = tgt_fg & (occ_z < tgt_z)
    shown = torch.isfinite(occ_z) & (~tgt_fg | (occ_z < tgt_z))
    out = {'occ_z': occ_z, 'hidden': hidden, 'shown': shown, 'tgt_fg': tgt_fg}
    if rgb is not None:
        out['rgb'] = rgb
    return out


def project_points(points, pc_norm, H, W):
    """Unit-normalised cloud points (B, N, 3) -> (row, col, z) in an H x W image.

    pc_norm (B, 4) = (centroid xyz, scale) from the dataset, so
    camera-frame metres = points * scale + centroid. Returns long row/col
    clamped into the frame, the camera z-distance, and an in-frame mask.
    """
    p = points * pc_norm[:, None, 3:4] + pc_norm[:, None, :3]
    z = (-p[..., 2]).clamp(min=1e-3)
    u = p[..., 0] / (z * _TAN_HALF_FOV)
    v = p[..., 1] / (z * _TAN_HALF_FOV)
    col = torch.floor((u + 1) / 2 * W)
    row = torch.floor((1 - v) / 2 * H)
    inside = (col >= 0) & (col < W) & (row >= 0) & (row < H)
    return (row.clamp(0, H - 1).long(), col.clamp(0, W - 1).long(), z, inside)


def hidden_points(points, pc_norm, occ_z):
    """(B, N) bool: which cloud points sit behind a placed neighbour."""
    B, _, H, W = occ_z.shape
    row, col, z, inside = project_points(points, pc_norm, H, W)
    flat = occ_z.view(B, -1)
    oz = torch.gather(flat, 1, row * W + col)
    return inside & (oz < z)


def image_token_flags(occ, patch_size, cfg):
    """(B, L) bool over row-major patches (PatchEmbed order): occluded plant patches."""
    plant = F.avg_pool2d(occ['tgt_fg'].float(), patch_size)
    hid = F.avg_pool2d(occ['hidden'].float(), patch_size)
    frac = hid / plant.clamp(min=1e-6)
    flags = (plant >= cfg.min_patch_plant) & (frac >= cfg.patch_hidden_frac)
    return flags.flatten(1)


def scores_from_flags(flags, generator=None):
    """Ranking scores for random_masking_dirichlet: lowest stays visible.

    Flagged tokens score in [1, 2), the rest in [0, 1), so flagged tokens are
    masked first and the remaining budget is filled uniformly at random.
    """
    noise = torch.rand(flags.shape, generator=generator, device=flags.device)
    return flags.float() + noise


def training_scores(depth, pc_centers, pc_norm, cfg, patch_size, modalities):
    """Masking scores for one training batch, or None when nothing applies.

    depth      : (B, 1, H, W) depth.png values (the encoder input)
    pc_centers : (B, L_pc, 3) FPS centres of THIS forward pass, unit-normalised
    pc_norm    : (B, 4) dataset (centroid, scale) for those clouds
    modalities : active streams; text is never structured.
    """
    B = depth.shape[0]
    if B < 2:
        return None              # nobody in the batch to borrow as a neighbour
    device = depth.device
    params = draw_params(B, cfg, device)
    if not bool(params['use'].any()):
        return None
    K = int(cfg.neighbours)
    shifts = torch.randint(1, B, (K,)).tolist()
    occ = occluders(depth, [depth.roll(s, 0) for s in shifts], params)

    scores = {}
    if 'rgb' in modalities or 'depth' in modalities:
        img = image_token_flags(occ, patch_size, cfg)
        for name in ('rgb', 'depth'):
            if name in modalities:
                scores[name] = scores_from_flags(img)
    if 'pc' in modalities:
        scores['pc'] = scores_from_flags(hidden_points(pc_centers, pc_norm, occ['occ_z']))
    return scores
