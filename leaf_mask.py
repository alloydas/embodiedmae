"""
Leaf-weighted masking for the 4M model: of the masking budget, the target
plant's LEAVES are masked more often than its stem.

Why. Leaves are the hard part of the reconstruction -- thin, curved, far from
the stem -- and recall falls from ~0.31 near the plant's centre to ~0.02 at its
edge. Uniform masking spends the budget evenly; this spends more of it on
leaves, so the model practises reconstructing them more, while the stem is
still masked often enough to be learned. The budget itself is unchanged: the
Dirichlet draw still fixes how many tokens of each modality stay visible, and
this only changes WHICH tokens.

Unlike occlusion_scene's neighbour_first, the weights come from the target's
own geometry, never from knowing where a neighbour is, and validation stays
uniform for every arm -- so nothing here needs information a deployed model
would not have.

Leaf vs stem. Every plant is rendered with its stem on the world y axis through
the origin (capture.py; checked on 60 sorghum plants, and on maize, median
1 cm), so a point's distance from that axis, hypot(x, z) in world metres, says
how far out on the plant it is. Points at least `stem_radius` from the axis are
leaf; the rest are stem. Neighbour points (occluded scenes) are neither and
keep weight 1, so they are masked at the base rate.

Masking. A token's weight is `weight` for leaf, 1 otherwise; image patches take
the mean weight of the target points that project into them (1 where none
does: background, or a neighbour standing clear of the target). The masked set
is a weighted sample without replacement (Gumbel top-k), so with weight 3 and
~80 % masked, leaf tokens are masked ~90 % of the time and stem tokens ~55-65 %.
"""

from dataclasses import dataclass, fields

import torch

from occlusion_scene import camera_pixels, to_world


@dataclass(frozen=True)
class LeafMaskConfig:
    """YAML `model.leaf_mask`; absent / null = uniform masking (every earlier run).

    weight      : masking weight of a leaf token relative to a stem, background
                  or neighbour token (1). 1.0 reproduces uniform masking in
                  distribution.
    stem_radius : metres from the stem axis below which a point counts as stem.
    """
    weight: float = 3.0
    stem_radius: float = 0.05

    @classmethod
    def from_dict(cls, cfg):
        if cfg is None:
            return None
        names = {f.name for f in fields(cls)}
        unknown = set(cfg) - names
        if unknown:
            raise ValueError(f"unknown leaf_mask keys {sorted(unknown)}; expected {sorted(names)}")
        out = cls(**{k: type(getattr(cls, k))(v) for k, v in cfg.items()})
        if out.weight <= 0:
            raise ValueError("leaf_mask.weight must be positive")
        if out.stem_radius < 0:
            raise ValueError("leaf_mask.stem_radius must be >= 0")
        return out

    def to_dict(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}


def point_weights(pc, pc_norm, cam2world, cfg, nb_point=None):
    """Masking weight per input point (B, N).

    pc        : (B, N, 3) cloud in the TARGET's unit-normalised camera frame --
                the clean cloud, or an occluded scene from occlusion_scene.compose
                (which keeps that frame).
    pc_norm   : (B, 4) the target's (centroid, scale).
    cam2world : (B, 4, 4) the target's camera pose.
    nb_point  : (B, N) bool, points that belong to a neighbour (weight 1), or None.
    """
    _, world = to_world(pc, pc_norm, cam2world)
    r = torch.hypot(world[..., 0], world[..., 2])
    w = torch.where(r >= cfg.stem_radius,
                    torch.full_like(r, float(cfg.weight)), torch.ones_like(r))
    if nb_point is not None:
        w = torch.where(nb_point, torch.ones_like(w), w)
    return w


def patch_weights(pc, pc_norm, cam2world, cfg, H, W, patch_size):
    """Masking weight per image patch (B, L): the mean weight of the target's
    own points that project into the patch, 1 where none does.

    pc must be the CLEAN target cloud (the patches' content may be a scene, but
    the weight is about which part of the target sits behind each patch).
    """
    B = pc.shape[0]
    cam, _ = to_world(pc, pc_norm, cam2world)
    w = point_weights(pc, pc_norm, cam2world, cfg)
    row, col, _ = camera_pixels(cam, H, W)
    ok = (row >= 0) & (row < H) & (col >= 0) & (col < W)
    gw = W // patch_size
    L = (H // patch_size) * gw
    pid = ((row.clamp(0, H - 1) // patch_size) * gw + col.clamp(0, W - 1) // patch_size).long()
    okf = ok.to(w.dtype)
    s = torch.zeros(B, L, device=pc.device, dtype=w.dtype).scatter_add_(1, pid, w * okf)
    n = torch.zeros(B, L, device=pc.device, dtype=w.dtype).scatter_add_(1, pid, okf)
    return torch.where(n > 0, s / n.clamp(min=1), torch.ones_like(s))


def scores_from_weights(w, generator=None):
    """Ranking scores for random_masking_dirichlet (highest masked first) that
    make the masked set a sample without replacement proportional to `w`
    (Gumbel top-k). Constant w gives uniform masking."""
    u = torch.rand(w.shape, generator=generator, device=w.device).clamp_(1e-12, 1 - 1e-12)
    return torch.log(w) - torch.log(-torch.log(u))


def mask_weights(pc_in, pc_clean, pc_norm, cam2world, cfg, H, W, patch_size, nb_point=None):
    """The `mask_flags` dict EmbodiedMAE4M takes, as float weights:
    {'rgb', 'depth': (B, L), 'pc_points': (B, N)}."""
    pw = patch_weights(pc_clean, pc_norm, cam2world, cfg, H, W, patch_size)
    return {'rgb': pw, 'depth': pw,
            'pc_points': point_weights(pc_in, pc_norm, cam2world, cfg, nb_point)}
