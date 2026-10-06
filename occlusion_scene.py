"""
Occluded-scene training for the 4M model: a target plant among its neighbours.

Why (occlusion meeting after 2026-09-29: Adarsh, B, Alloy, Yongyun). Masking
the tokens a neighbour WOULD hide (structured_mask.py) never shows the model a
neighbour -- its input is still one clean plant, and validation was clean too,
so nothing measured whether it can pull one plant out of a crowd. The setup
agreed there:

  input   RGB, depth and point cloud of a SCENE: the target plant in the
          centre, random plants standing around it, their leaves in front of,
          behind and through the target's;
  target  the CLEAN target plant alone, in every modality;
  params  the TARGET's procedural parameters as conditioning, never the
          neighbours' -- without them nothing tells the model which leaves are
          the target's. In the YAML that is text_mask_ratio: 0.0 (every param
          token visible, none reconstructed).

Alloy is rendering such scenes properly (pyrender, real meshes). This module
builds them on the fly from the single-plant renders already on disk, so the
same trainer runs now and can take Alloy's scenes later.

Geometry. capture.py renders each plant with its stem at the world origin
(checked on 60 plants: stem base at x = z = 0, ground y = 0), world y up, from
a camera 2.5 m from the plant's bbox centre with no roll; camera_pose.json is
that camera's cameraToWorld, and *_nc_cam.ply is the complete cloud moved into
the camera frame. So any sample's cloud goes to world exactly, and from world
into any other sample's camera. A neighbour is another plant from the batch:
spun about its own stem by a random yaw, then stood on the target's ground at
`spacing` metres along a row through the target (random row direction, both
sides when neighbours = 2), `row_jitter` metres off the row line. Then:

  * point cloud: the neighbours' points join the target's, the scene is
    cropped to a vertical cylinder of `crop_radius` metres around the target's
    stem -- the bounding box that segments one plant out of a row -- and the
    result is resampled to num_points;
  * depth / RGB: the neighbours' points are splatted into the target's image
    with a z-buffer ((2 splat_px + 1)^2 footprint, which closes the gaps of
    ~8k points at ~1 px spacing) and win every pixel where they are nearer
    than the target's own depth. Each point takes its colour from the
    neighbour's own render, at the pixel it projects to in its own camera.

Both come from the same 3D placement, so a leaf that hides the target in the
image also sits in front of it in the cloud.

Measured footprint (60 plants, world frame): 95 % of a plant's points lie
within 0.52 m of its stem (median; 0.89 m max), its farthest point at
0.65-1.16 m. The default spacing [0.35, 0.8] m therefore always overlaps
leaves, and crop_radius 0.75 m keeps nearly all of the target.

Known differences from a rendered scene, all acceptable for training:
  * neighbours are splatted points, not meshes: leaves ~1 px fatter, no
    re-shading after the yaw. Alloy's rendered scenes are the test that a model
    did not learn "splat = neighbour".
  * the cloud stays in the CLEAN target's normalisation frame (its centroid and
    scale), as in eval/eval_occlusion.py -- prediction and target must share a
    frame. A deployed model would normalise by the crop instead.
  * the cloud is the complete scene surface, like every cloud this repo trains
    on, not what one camera sees.

The only images are the 224 x 224 network inputs; nothing here touches disk.
"""

import math
from dataclasses import dataclass, fields, replace

import torch
import torch.nn.functional as F

from structured_mask import (
    RENDER_FAR, RENDER_FOV_DEG, RENDER_NEAR, _pair, foreground, metric_depth,
)

_TAN_HALF_FOV = math.tan(math.radians(RENDER_FOV_DEG) / 2)
MASK_POLICIES = ('uniform', 'neighbour_first')


@dataclass(frozen=True)
class SceneConfig:
    """`model.occlusion_scene` in the YAML. Absent / null / enabled: false = off.

    prob          : fraction of TRAINING samples given neighbours; the rest stay
                    clean. 0 trains clean but still validates on scenes, which
                    is what a control arm needs.
    neighbours    : 1 = one plant on a random side; 2 = one on each side.
    spacing       : stem-to-stem distance along the row, metres.
    row_jitter    : how far off the row line a neighbour may stand, metres.
    yaw           : spin each neighbour about its stem by a random angle.
    crop_radius   : radius of the vertical cylinder around the target's stem
                    that the scene cloud is cut to, metres; null keeps every
                    point. Applied to the target's own points too.
    splat_px      : splat footprint half-width in pixels (1 -> 3 x 3).
    mask_policy   : 'uniform' -- the usual Dirichlet masking; the encoder sees
                    neighbour leaves and has to tell them apart. Deployable.
                    'neighbour_first' -- tokens showing a neighbour (image
                    patches, PC tokens centred on a neighbour point) are masked
                    first, the rest filled uniformly. An ORACLE: at test time
                    nothing says where the neighbours are.
    patch_frac    : an image patch "shows a neighbour" for neighbour_first when
                    at least this share of its pixels does.
    nf_prob       : neighbour_first only: the share of SCENE samples that get
                    neighbour-first masking; the rest are masked uniformly, so
                    the model still sees visible neighbours (as it will at test)
                    while also practising filling in behind them. 1.0 = every
                    scene sample, i.e. every run before 2026-10-03; to_dict
                    leaves it out at 1.0 so older checkpoints' resume guards
                    still match.
    val_seed      : seed of the fixed validation scenes (same for every arm).

    Validation always uses prob 1 and uniform masking, so every arm is scored on
    the same scenes under the same masking whatever it trained with.
    """
    prob: float = 0.5
    neighbours: int = 2
    spacing: tuple = (0.35, 0.8)
    row_jitter: float = 0.1
    yaw: bool = True
    crop_radius: float = 0.75
    splat_px: int = 1
    mask_policy: str = 'uniform'
    patch_frac: float = 0.3
    val_seed: int = 0
    nf_prob: float = 1.0

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
                f"unknown occlusion_scene keys {unknown}; expected {sorted(known)}")
        if 'spacing' in cfg:
            cfg['spacing'] = _pair(cfg['spacing'], 'spacing')
        if cfg.get('crop_radius') in ('null', 'none', 'None'):
            cfg['crop_radius'] = None
        out = cls(**cfg)
        if not 0.0 <= float(out.prob) <= 1.0:
            raise ValueError("occlusion_scene.prob must be in [0, 1]")
        if int(out.neighbours) not in (1, 2):
            raise ValueError("occlusion_scene.neighbours must be 1 or 2")
        if out.spacing[0] < 0 or out.row_jitter < 0:
            raise ValueError("occlusion_scene.spacing / row_jitter must be non-negative")
        if out.crop_radius is not None and not float(out.crop_radius) > 0:
            raise ValueError("occlusion_scene.crop_radius must be positive or null")
        if int(out.splat_px) < 0:
            raise ValueError("occlusion_scene.splat_px must be >= 0")
        if out.mask_policy not in MASK_POLICIES:
            raise ValueError(f"occlusion_scene.mask_policy must be one of {MASK_POLICIES}")
        if not 0.0 < float(out.patch_frac) <= 1.0:
            raise ValueError("occlusion_scene.patch_frac must be in (0, 1]")
        if not 0.0 <= float(out.nf_prob) <= 1.0:
            raise ValueError("occlusion_scene.nf_prob must be in [0, 1]")
        if float(out.nf_prob) < 1.0 and out.mask_policy != 'neighbour_first':
            raise ValueError("occlusion_scene.nf_prob only applies to mask_policy: neighbour_first")
        return out

    def to_dict(self):
        return {f.name: (list(v) if isinstance(v := getattr(self, f.name), tuple) else v)
                for f in fields(self)
                if not (f.name == 'nf_prob' and float(getattr(self, f.name)) == 1.0)}

    def for_validation(self):
        """Every sample occluded, uniform masking: the regime all arms share."""
        return replace(self, prob=1.0, mask_policy='uniform')


# ── Random draws ──────────────────────────────────────────────────────────────

def draw_params(B, cfg, generator=None):
    """Per-sample placement on the CPU, so a fixed generator fixes every scene.

    nb (B, K) is the batch index of each neighbour: never the sample itself,
    and distinct within a sample whenever the batch has room.
    """
    K = int(cfg.neighbours)

    def rand(*shape):
        return torch.rand(*shape, generator=generator)

    use = rand(B) < cfg.prob
    if B - 1 >= K:
        off = 1 + torch.argsort(rand(B, B - 1), dim=1)[:, :K]
    else:
        off = 1 + torch.randint(B - 1, (B, K), generator=generator)
    side0 = torch.where(rand(B) < 0.5, -1.0, 1.0)
    lo, hi = cfg.spacing
    return {
        'use': use,
        'nb': (torch.arange(B)[:, None] + off) % B,
        'row': 2 * math.pi * rand(B),                          # row direction
        'side': torch.stack([side0 * (-1.0) ** k for k in range(K)], dim=1),
        'spacing': lo + (hi - lo) * rand(B, K),
        'across': (2 * rand(B, K) - 1) * cfg.row_jitter,
        'yaw': (2 * math.pi * rand(B, K)) if cfg.yaw else torch.zeros(B, K),
    }


# ── Geometry ──────────────────────────────────────────────────────────────────

def to_world(pc, pc_norm, cam2world):
    """Unit-normalised cloud (B, N, 3) -> camera metres and world metres."""
    cam = pc * pc_norm[:, None, 3:4] + pc_norm[:, None, :3]
    R, t = cam2world[:, :3, :3], cam2world[:, :3, 3]
    return cam, cam @ R.transpose(1, 2) + t[:, None]


def world_to_camera(p_world, cam2world):
    """World points (B, M, 3) -> that sample's camera frame (R^T (p - t))."""
    R, t = cam2world[:, :3, :3], cam2world[:, :3, 3]
    return (p_world - t[:, None]) @ R


def camera_pixels(p_cam, H, W):
    """Camera-frame metres -> (row, col) float pixel indices and z-distance."""
    z = -p_cam[..., 2]
    zc = z.clamp(min=RENDER_NEAR)
    u = p_cam[..., 0] / (zc * _TAN_HALF_FOV)
    v = p_cam[..., 1] / (zc * _TAN_HALF_FOV)
    return torch.floor((1 - v) / 2 * H), torch.floor((u + 1) / 2 * W), z


def own_colours(cam, rgb, depth):
    """Each point's colour in its own render (B, N, 3), in rgb's (normalised) space.

    A point that lands on background (projection error at a silhouette edge)
    takes the sample's mean plant colour instead of the grey backdrop.
    """
    B, _, H, W = rgb.shape
    row, col, z = camera_pixels(cam, H, W)
    inside = (row >= 0) & (row < H) & (col >= 0) & (col < W) & (z > RENDER_NEAR)
    flat = (row.clamp(0, H - 1) * W + col.clamp(0, W - 1)).long()
    c = rgb.flatten(2).gather(2, flat[:, None].expand(-1, 3, -1)).transpose(1, 2)
    fg = foreground(depth)
    on_fg = fg.flatten(1).gather(1, flat) & inside
    mean_fg = (rgb * fg).sum((2, 3)) / fg.sum((2, 3)).clamp(min=1)
    return torch.where(on_fg[..., None], c, mean_fg[:, None])


def splat(p_cam, colours, H, W, px=1, valid=None):
    """Z-buffered point splat. Returns z (B, 1, H, W), +inf where empty, and rgb.

    Every point covers a (2 px + 1)^2 footprint; the nearest point wins each
    pixel and lends it its colour (ties between points at exactly equal depth
    resolve arbitrarily). Then the footprint is closed (3 x 3 dilate, erode),
    a filled pixel taking the nearest depth around it, and colours are averaged
    over each pixel's occupied 3 x 3 neighbourhood. Without that, ~8k points
    leave 1-2 px holes and per-point colour noise -- a speckle no rendered
    plant has, i.e. a free "this is a neighbour" cue.
    """
    B, M, _ = p_cam.shape
    dev = p_cam.device
    row, col, z = camera_pixels(p_cam, H, W)
    d = torch.arange(-px, px + 1, device=dev, dtype=row.dtype)
    dr, dc = torch.meshgrid(d, d, indexing='ij')
    rows = row[..., None] + dr.flatten()                    # (B, M, F)
    cols = col[..., None] + dc.flatten()
    ok = (z > RENDER_NEAR)[..., None] & (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    if valid is not None:
        ok = ok & valid[..., None]
    flat = (rows.clamp(0, H - 1) * W + cols.clamp(0, W - 1)).long().view(B, -1)
    zz = torch.where(ok, z[..., None], torch.inf).view(B, -1)
    zbuf = torch.full((B, H * W), torch.inf, device=dev).scatter_reduce(
        1, flat, zz, reduce='amin', include_self=True)
    win = ok.view(B, -1) & (zz == zbuf.gather(1, flat))
    gidx = (flat + torch.arange(B, device=dev)[:, None] * (H * W))[win]
    cexp = colours[:, :, None, :].expand(-1, -1, rows.shape[-1], -1).reshape(B, -1, 3)
    img = torch.zeros(B * H * W, 3, device=dev, dtype=colours.dtype)
    img[gidx] = cexp[win]
    zimg = zbuf.view(B, 1, H, W)
    img = img.view(B, H, W, 3).permute(0, 3, 1, 2)

    occ = torch.isfinite(zimg)
    occf = occ.to(img.dtype)
    closed = -F.max_pool2d(-F.max_pool2d(occf, 3, 1, 1), 3, 1, 1) > 0.5
    fill = closed & ~occ
    znear = -F.max_pool2d(-torch.where(occ, zimg, torch.full_like(zimg, 1e6)), 3, 1, 1)
    zimg = torch.where(fill, znear, zimg)
    smooth = F.avg_pool2d(img * occf, 3, 1, 1) / F.avg_pool2d(occf, 3, 1, 1).clamp(min=1e-6)
    img = torch.where(occ | fill, smooth, img)
    return zimg, img


def resample(points, keep, n_out, generator=None):
    """Pick n_out of each sample's kept points at random (B, M, 3) -> (B, n_out, 3).

    Repeats kept points when fewer than n_out survive, like the dataset's own
    padding. Returns the picks' indices into `points` too.
    """
    score = torch.rand(keep.shape, generator=generator, device=points.device)
    order = torch.argsort(torch.where(keep, score, 2.0), dim=1)
    n_kept = keep.sum(1, keepdim=True).clamp(min=1)
    j = torch.arange(n_out, device=points.device)[None]
    idx = order.gather(1, torch.where(j < n_kept, j, j % n_kept))
    return points.gather(1, idx[..., None].expand(-1, -1, 3)), idx


# ── The scene ─────────────────────────────────────────────────────────────────

def compose(rgb, depth, pc, pc_norm, cam2world, cfg, patch_size=16, generator=None,
            near_far=None):
    """Turn a batch of clean single plants into occluded scenes, or None.

    rgb, depth, pc, pc_norm, cam2world : the dataset's tensors for B samples
        (SorghumDataset4M(return_pose=True)), on one device. They stay the
        reconstruction targets; this returns new INPUT tensors.
    near_far : (B, 2) the depth.png encoding's near and far plane per sample, in
        metres, or None for sorghum's renderer (RENDER_NEAR / RENDER_FAR, the
        same for every sample). Maize's renderer sets them per plant and writes
        them to camera_pose.json, so MaizeDataset4M(return_pose=True) returns
        them. A neighbour is written into the target's depth image with the
        TARGET's encoding. Field of view is 40 degrees in both renderers.
    generator : a CPU torch.Generator for a fixed scene set (validation), or
        None for the global RNGs (training).

    Returns a dict:
      rgb, depth, pc   the scene inputs; samples without neighbours are their
                       clean tensors, bit for bit
      nb_point         (B, N) bool, input points that belong to a neighbour
      shown            (B, 1, H, W) bool, pixels where a neighbour shows
      loss_tokens      {'rgb', 'depth'}: (B, L) bool, patches where a neighbour
                       shows -- there the input is not the target, so they are
                       scored even when visible
      mask_flags       neighbour_first only: {'rgb', 'depth'} (B, L) and
                       'pc_points' (B, N) bool, for EmbodiedMAE4M's masking
      stats            floats over the occluded samples: hidden_px (share of the
                       target's plant pixels a neighbour covers), nb_points
                       (share of the input cloud that is neighbour),
                       tgt_points_kept (share of the target the crop keeps),
                       loss_patches (scored-while-visible patches per sample)
    None when nothing in the batch gets neighbours (B < 2, or every draw clean).
    """
    B, N, _ = pc.shape
    if B < 2:
        return None
    p = draw_params(B, cfg, generator)
    if not bool(p['use'].any()):
        return None
    dev = pc.device
    p = {k: v.to(dev) for k, v in p.items()}
    dgen = None
    if generator is not None:
        dgen = torch.Generator(device=dev).manual_seed(
            int(torch.randint(2 ** 62, (1,), generator=generator)))
    use = p['use']
    K = p['nb'].shape[1]
    H, W = depth.shape[-2:]

    cam, world = to_world(pc, pc_norm, cam2world)

    # Neighbours: spun about their own stem (world y through the origin), then
    # stood on the target's ground along the row.
    nbw = world[p['nb']]                                      # (B, K, N, 3)
    x, y, z = nbw.unbind(-1)
    cy, sy = torch.cos(p['yaw'])[..., None], torch.sin(p['yaw'])[..., None]
    cr, sr = torch.cos(p['row'])[:, None, None], torch.sin(p['row'])[:, None, None]
    along = (p['side'] * p['spacing'])[..., None]
    across = p['across'][..., None]
    nbw = torch.stack([cy * x + sy * z + along * cr - across * sr,
                       y,
                       -sy * x + cy * z + along * sr + across * cr], dim=-1)
    nbw = nbw.reshape(B, K * N, 3)
    nb_cam = world_to_camera(nbw, cam2world)
    nb_on = use[:, None].expand(B, K * N)

    # Images: splat the neighbours, nearest surface wins.
    colours = own_colours(cam, rgb, depth)[p['nb']].reshape(B, K * N, 3)
    nb_z, nb_rgb = splat(nb_cam, colours, H, W, int(cfg.splat_px), valid=nb_on)
    tgt_fg = foreground(depth)
    if near_far is None:
        tgt_z = metric_depth(depth)
        near, span = RENDER_NEAR, RENDER_FAR - RENDER_NEAR
    else:
        near = near_far[:, 0].view(B, 1, 1, 1).to(depth.dtype)
        span = (near_far[:, 1] - near_far[:, 0]).view(B, 1, 1, 1).to(depth.dtype)
        tgt_z = near + depth * span
    shown = nb_z < torch.where(tgt_fg, tgt_z, torch.inf)
    rgb_in = torch.where(shown, nb_rgb.to(rgb.dtype), rgb)
    depth_in = torch.where(
        shown, ((nb_z - near) / span).clamp(0, 1), depth)

    # Cloud: target + neighbours in the target's normalised frame, cropped to
    # the cylinder around the target's stem, resampled to N.
    nb_norm = (nb_cam - pc_norm[:, None, :3]) / pc_norm[:, None, 3:4]
    pts = torch.cat([pc, nb_norm], dim=1)
    keep = torch.cat([torch.ones(B, N, dtype=torch.bool, device=dev), nb_on], dim=1)
    if cfg.crop_radius is not None:
        w_all = torch.cat([world, nbw], dim=1)
        inside = torch.hypot(w_all[..., 0], w_all[..., 2]) <= float(cfg.crop_radius)
        keep = keep & (inside | ~use[:, None])
    pc_scene, idx = resample(pts, keep, N, dgen)
    pc_in = torch.where(use[:, None, None], pc_scene, pc)
    nb_point = (idx >= N) & use[:, None]

    shown_frac = F.avg_pool2d(shown.float(), patch_size).flatten(1)
    loss_tok = shown_frac > 0
    out = {'rgb': rgb_in, 'depth': depth_in, 'pc': pc_in, 'nb_point': nb_point,
           'shown': shown, 'loss_tokens': {'rgb': loss_tok, 'depth': loss_tok}}
    if cfg.mask_policy == 'neighbour_first':
        img = shown_frac >= float(cfg.patch_frac)
        nbf = nb_point
        if float(cfg.nf_prob) < 1.0:
            # Only this share of the scene samples is masked neighbour-first; a
            # sample with no flags falls back to uniform noise. Drawn last, so
            # every other draw (and nf_prob 1.0 runs) is unchanged.
            sel = (torch.rand(B, generator=generator) < float(cfg.nf_prob)).to(dev)
            img = img & sel[:, None]
            nbf = nbf & sel[:, None]
        out['mask_flags'] = {'rgb': img, 'depth': img, 'pc_points': nbf}

    u = use
    fg_n = tgt_fg.flatten(1).sum(1).clamp(min=1)
    out['stats'] = {
        'hidden_px': ((tgt_fg & shown).flatten(1).sum(1) / fg_n)[u].mean().item(),
        'nb_points': nb_point.float().mean(1)[u].mean().item(),
        'tgt_points_kept': keep[:, :N].float().mean(1)[u].mean().item(),
        'loss_patches': loss_tok.float().sum(1)[u].mean().item(),
    }
    return out
