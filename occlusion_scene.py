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
    on, not what one camera sees -- unless visible_only (2026-10-15).

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
NORMS = ('target', 'crop', 'metric', 'stem')


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
    norm          : the frame of a scene sample's clouds. 'target' -- the clean
                    target's own normalisation (centroid at 0, farthest point on
                    the unit sphere), every run before 2026-10-10. That frame is
                    a leak: the input cloud's origin and scale are the TARGET's,
                    which a real crop does not know. 'crop' -- re-centre and
                    re-scale the scene cloud by ITS OWN centroid and farthest
                    point, and express the clean target in that same frame (so
                    the model must find the target's position and size itself).
                    Applied to scene samples only; a clean sample's own frame is
                    already its crop frame. to_dict leaves it out at 'target'.
                    'metric' -- centred like 'crop', but scaled by a FIXED length,
                    `unit_m` metres per unit, instead of the crop's own radius: the
                    cloud keeps its real size (a depth camera knows metres), so the
                    model can use absolute size and the card's lengths. Applied to
                    EVERY sample, clean ones included (scaled about their own
                    centroid); batches without a scene go through metric_frame().
                    'stem' -- metres / unit_m like 'metric', but the origin is the
                    TARGET's STEM BASE (the world origin, where capture.py stands
                    every plant) instead of a centroid, for every sample. It says
                    which plant to rebuild ("the one standing here") without its
                    centroid or extent; a real system gets it from a stem detector
                    or a click plus the depth camera's ground plane. Batches without
                    a scene go through no_scene_frame().
    unit_m        : metres per unit for norm 'metric' / 'stem' (maize 0.64 = the median
                    plant radius, so clouds stay near unit size). Omitted from
                    to_dict when None.
    visible_only  : the INPUT cloud holds only what the target's camera sees, as a
                    depth camera would: target points visible in their own render
                    (the dataset's per-point `vis`, tested against the full-resolution
                    depth, return_visibility=True) and not behind a neighbour, and
                    neighbour points not behind the target or another neighbour and
                    inside the image. Clean samples get their visible points too
                    (visible_input()). The TARGET stays the complete clean plant.
                    False = the complete cloud, every run before 2026-10-15 -- which
                    hands the model surfaces the images show hidden.
    visible_tol   : visible_only: metres a point may lie behind the z-buffer of the
                    224 px scene (neighbour occlusion) and still count as seen.
    origin_jitter_m : norm 'stem', TRAINING only: per-axis std (metres) of a random
                    error added to the stem-base origin of every sample, so the model
                    cannot rely on an exact stem position (a real stem estimate is
                    off by centimetres; eval/depth_reliance.py measures the cost).
                    The target moves with the frame, so the model must place the
                    plant from the input itself. for_validation() sets it to 0.

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
    norm: str = 'target'
    unit_m: float = None
    visible_only: bool = False
    visible_tol: float = 0.02
    origin_jitter_m: float = 0.0

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
        if out.norm not in NORMS:
            raise ValueError(f"occlusion_scene.norm must be one of {NORMS}")
        if (out.norm in ('metric', 'stem')) != (out.unit_m is not None):
            raise ValueError("occlusion_scene.unit_m is required by, and only used with, norm: metric / stem")
        if out.unit_m is not None and not float(out.unit_m) > 0:
            raise ValueError("occlusion_scene.unit_m must be positive")
        if not float(out.visible_tol) > 0:
            raise ValueError("occlusion_scene.visible_tol must be positive")
        if float(out.origin_jitter_m) < 0:
            raise ValueError("occlusion_scene.origin_jitter_m must be >= 0")
        if float(out.origin_jitter_m) > 0 and out.norm != 'stem':
            raise ValueError("occlusion_scene.origin_jitter_m only applies to norm: stem")
        return out

    def to_dict(self):
        return {f.name: (list(v) if isinstance(v := getattr(self, f.name), tuple) else v)
                for f in fields(self)
                if not (f.name == 'nf_prob' and float(getattr(self, f.name)) == 1.0)
                and not (f.name == 'norm' and getattr(self, f.name) == 'target')
                and not (f.name == 'unit_m' and getattr(self, f.name) is None)
                and not (f.name == 'visible_only' and not getattr(self, f.name))
                and not (f.name == 'visible_tol' and float(getattr(self, f.name)) == 0.02)
                and not (f.name == 'origin_jitter_m' and float(getattr(self, f.name)) == 0.0)}

    def for_validation(self):
        """Every sample occluded, uniform masking, exact stem origin: the regime all
        arms share."""
        return replace(self, prob=1.0, mask_policy='uniform', origin_jitter_m=0.0)


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
            near_far=None, vis=None):
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
    vis : (B, N) bool, cfg.visible_only only: which of each target's points its
        own camera sees (the dataset's return_visibility). Then EVERY sample's
        input cloud is its visible part (clean samples included, so they are no
        longer bit for bit their clean tensors).

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
      target_pc        norm 'crop' / 'metric' only: (B, N, 3) the clean target in the
                       scene cloud's own frame -- score the prediction against
                       THIS, not `pc`
      crop_norm        norm 'crop' / 'metric' only: (B, 4) centroid and scale of that frame
                       in the target's frame (0, 0, 0, 1 for clean samples):
                       target_frame = crop_frame * scale + centroid, and one
                       crop-frame unit is pc_norm[:, 3] * scale metres
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
    if cfg.visible_only:
        if vis is None:
            raise ValueError("occlusion_scene.visible_only needs the dataset's per-point "
                             "visibility (return_visibility=True) passed as vis=")
        # z-buffers of the 224 px scene. 3 x 3 max so a point at a silhouette edge, or
        # under an antialiased edge pixel of the target's render, counts as seen.
        zt = torch.where(tgt_fg, tgt_z, torch.full_like(tgt_z, torch.inf))
        z_nb = F.max_pool2d(nb_z, 3, 1, 1)
        z_all = F.max_pool2d(torch.minimum(zt, nb_z), 3, 1, 1)
        vis_t = vis.to(dev).bool() & seen_in(cam, z_nb, float(cfg.visible_tol))
        vis_n = seen_in(nb_cam, z_all, float(cfg.visible_tol))
        keep = keep & torch.cat([vis_t, vis_n], dim=1)
    pc_scene, idx = resample(pts, keep, N, dgen)
    # visible_only: a clean sample's resample is its own visible points (no neighbour
    # is on, so vis_t = vis), which is its input too
    pc_in = pc_scene if cfg.visible_only else torch.where(use[:, None, None], pc_scene, pc)
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

    if cfg.norm in ('crop', 'metric', 'stem'):
        # Drawn from nothing random (origin_jitter_m aside, drawn last), so every
        # other draw is unchanged.
        if cfg.norm == 'stem':
            c = stem_origin(pc_norm, cam2world)
            if float(cfg.origin_jitter_m) > 0:
                c = c + origin_error(B, cfg, pc_norm, generator)
        else:
            c = torch.where(use[:, None], out['pc'].mean(1), torch.zeros_like(out['pc'][:, 0]))
        if cfg.norm == 'crop':
            r = (out['pc'] - c[:, None]).norm(dim=-1).max(1).values
            r = torch.where(use, r.clamp(min=1e-6), torch.ones_like(r))
        else:
            # target units per metric unit, for EVERY sample (clean ones too)
            r = float(cfg.unit_m) / pc_norm[:, 3]
        out['pc'] = (out['pc'] - c[:, None]) / r[:, None, None]
        out['target_pc'] = (pc - c[:, None]) / r[:, None, None]
        out['crop_norm'] = torch.cat([c, r[:, None]], dim=1)

    u = use
    fg_n = tgt_fg.flatten(1).sum(1).clamp(min=1)
    out['stats'] = {
        'hidden_px': ((tgt_fg & shown).flatten(1).sum(1) / fg_n)[u].mean().item(),
        'nb_points': nb_point.float().mean(1)[u].mean().item(),
        'tgt_points_kept': keep[:, :N].float().mean(1)[u].mean().item(),
        'loss_patches': loss_tok.float().sum(1)[u].mean().item(),
    }
    if cfg.norm in ('crop', 'metric', 'stem'):
        out['stats']['crop_scale'] = out['crop_norm'][:, 3][u].mean().item()
        out['stats']['crop_shift'] = out['crop_norm'][:, :3].norm(dim=1)[u].mean().item()
    return out


def metric_frame(pc, pc_norm, unit_m):
    """norm 'metric' for a batch WITHOUT a scene (clean validation, card-only steps,
    or a batch compose() returned None for): each cloud keeps its own centroid and
    is put in units of `unit_m` metres. Returns (pc_metric, crop_norm) with
    target_frame = pc_metric * crop_norm[:, 3:] + crop_norm[:, :3], as compose()."""
    r = float(unit_m) / pc_norm[:, 3]
    cn = torch.cat([torch.zeros_like(pc_norm[:, :3]), r[:, None]], dim=1)
    return pc / r[:, None, None], cn


def stem_origin(pc_norm, cam2world):
    """The target's stem base -- the world origin -- in each sample's normalised
    camera frame (the frame the dataset's clouds are in): (B, 3)."""
    zero = torch.zeros(cam2world.shape[0], 1, 3, device=cam2world.device, dtype=cam2world.dtype)
    cam = world_to_camera(zero, cam2world)[:, 0]
    return (cam - pc_norm[:, :3]) / pc_norm[:, 3:4]


def no_scene_frame(pc, pc_norm, cam2world, cfg, generator=None):
    """norm 'metric' / 'stem' for a batch WITHOUT a scene (clean validation, card-only
    steps, a batch compose() returned None for). Returns (pc_framed, crop_norm) with
    target_frame = pc_framed * crop_norm[:, 3:] + crop_norm[:, :3], as compose()."""
    if cfg.norm == 'metric':
        return metric_frame(pc, pc_norm, cfg.unit_m)
    if cfg.norm != 'stem':
        raise ValueError(f"no_scene_frame: norm {cfg.norm!r} keeps the target frame")
    c = stem_origin(pc_norm, cam2world)
    if float(cfg.origin_jitter_m) > 0:
        c = c + origin_error(pc.shape[0], cfg, pc_norm, generator)
    r = float(cfg.unit_m) / pc_norm[:, 3]
    return (pc - c[:, None]) / r[:, None, None], torch.cat([c, r[:, None]], dim=1)


def origin_error(B, cfg, pc_norm, generator=None):
    """norm 'stem' training: a random stem-base error, N(0, origin_jitter_m^2) metres
    per axis (isotropic, so the frame it is drawn in does not matter), in the
    target's normalised units -- (B, 3), to add to stem_origin()."""
    e = torch.randn(B, 3, generator=generator).to(pc_norm.device) * float(cfg.origin_jitter_m)
    return e / pc_norm[:, 3:4]


def seen_in(p_cam, zimg, tol):
    """(B, M) bool: the point is inside the image and at most `tol` metres behind
    the z-buffer `zimg` (B, 1, H, W) at its pixel (+inf where nothing is drawn)."""
    B, _, H, W = zimg.shape
    row, col, z = camera_pixels(p_cam, H, W)
    inside = (row >= 0) & (row < H) & (col >= 0) & (col < W) & (z > RENDER_NEAR)
    flat = (row.clamp(0, H - 1) * W + col.clamp(0, W - 1)).long()
    return inside & (z <= zimg.flatten(1).gather(1, flat) + tol)


def point_visibility(p_cam, z_full, tol=0.01):
    """For the datasets (return_visibility): which points of ONE sample's cloud its
    own camera sees. p_cam (N, 3) camera-frame metres; z_full (H, W) the render's
    z-distance in metres at FULL resolution (1024 px: no antialiasing, ~2 mm per
    pixel), +inf on background. Seen = no more than `tol` behind the farthest
    surface in the point's 3 x 3 pixel neighbourhood. Returns (N,) bool."""
    return seen_in(p_cam[None], F.max_pool2d(z_full[None, None], 3, 1, 1), tol)[0]


def visible_input(pc, vis, generator=None):
    """visible_only, a batch WITHOUT a scene: each sample's INPUT is its visible
    points resampled to N (repeats when fewer survive), in pc's own frame. Score
    the prediction against the complete `pc`. `generator` is a CPU generator (or None),
    turned into one on pc's device the way compose() does."""
    dgen = None
    if generator is not None:
        dgen = torch.Generator(device=pc.device).manual_seed(
            int(torch.randint(2 ** 62, (1,), generator=generator)))
    return resample(pc, vis.to(pc.device).bool(), pc.shape[1], dgen)[0]
