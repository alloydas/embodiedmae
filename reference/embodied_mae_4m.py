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
    earth_movers_distance,
)


# ── Numeric parameter layout ──────────────────────────────────────────────────

N_PARAMS = 9   # fixed-size float vector per token

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

def load_spline_params(yml_path, max_leaves: int = 24):
    """Parse a spline YAML into:
      valid        : (1+max_leaves,)            float32 — 1=real token, 0=pad
      param_floats : (1+max_leaves, N_PARAMS)   float32 — encoder input + target
    """
    import yaml
    with open(yml_path) as f:
        data = yaml.safe_load(f)

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
    ):
        super().__init__()

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
        self.target_points      = target_points
        self.points_per_token   = target_points // num_pc_tokens

        # ── Embedders ─────────────────────────────────────────────────────
        self.rgb_embed   = PatchEmbed(img_size, patch_size, in_chans_rgb,   embed_dim)
        self.depth_embed = PatchEmbed(img_size, patch_size, in_chans_depth, embed_dim)
        self.pc_embed    = PointCloudEmbed(num_pc_tokens, pc_group_size, embed_dim)
        self.param_embed = ParamEmbed(N_PARAMS, embed_dim)

        # ── Modality embeddings ───────────────────────────────────────────
        self.modality_embed_rgb   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.modality_embed_depth = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.modality_embed_pc    = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.modality_embed_text  = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # ── Positional embeddings ─────────────────────────────────────────
        self.pos_embed_2d   = nn.Parameter(
            torch.zeros(1, self.num_patches,   embed_dim), requires_grad=False)
        self.pos_embed_pc   = nn.Parameter(
            torch.zeros(1, num_pc_tokens,      embed_dim), requires_grad=True)
        self.pos_embed_text = nn.Parameter(
            torch.zeros(1, self.n_text_tokens, embed_dim), requires_grad=True)

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

        self.decoder_pos_embed_2d   = nn.Parameter(
            torch.zeros(1, self.num_patches,   decoder_embed_dim), requires_grad=False)
        self.decoder_pos_embed_pc   = nn.Parameter(
            torch.zeros(1, num_pc_tokens,      decoder_embed_dim), requires_grad=False)
        self.decoder_pos_embed_text = nn.Parameter(
            torch.zeros(1, self.n_text_tokens, decoder_embed_dim), requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        # ── Prediction heads ──────────────────────────────────────────────
        self.decoder_pred_rgb   = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans_rgb)
        self.decoder_pred_depth = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans_depth)

        # Param head: 2-layer MLP — produces N_PARAMS normalised floats per token
        self.decoder_pred_params = nn.Sequential(
            nn.Linear(decoder_embed_dim, decoder_embed_dim),
            nn.GELU(),
            nn.LayerNorm(decoder_embed_dim),
            nn.Linear(decoder_embed_dim, N_PARAMS),
        )

        # PC folding decoder
        self.decoder_pc_proj = nn.Linear(decoder_embed_dim, 512)
        self.decoder_pc_fold = nn.Sequential(
            nn.Linear(514, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 3),
        )

        self.initialize_weights()

    # ── Weights ───────────────────────────────────────────────────────────────

    def initialize_weights(self):
        pos_2d = get_2d_sincos_pos_embed(self.embed_dim, int(self.num_patches ** .5))
        self.pos_embed_2d.data.copy_(torch.from_numpy(pos_2d).float().unsqueeze(0))

        dec_dim = self.decoder_blocks[0].norm1.normalized_shape[0]
        dec_2d  = get_2d_sincos_pos_embed(dec_dim, int(self.num_patches ** .5))
        self.decoder_pos_embed_2d.data.copy_(torch.from_numpy(dec_2d).float().unsqueeze(0))

        for p in (self.pos_embed_pc, self.pos_embed_text,
                  self.decoder_pos_embed_pc, self.decoder_pos_embed_text):
            nn.init.normal_(p, std=.02)
        for e in (self.modality_embed_rgb, self.modality_embed_depth,
                  self.modality_embed_pc,  self.modality_embed_text):
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

    def random_masking_dirichlet(self, x_rgb, x_depth, x_pc, x_text,
                                  mask_ratio_total=0.75, min_mask_ratio=0.25):
        B = x_rgb.shape[0]
        L = {m: x.shape[1] for m, x in
             zip(['rgb','depth','pc','text'], [x_rgb, x_depth, x_pc, x_text])}
        total    = sum(L.values())
        nv_total = int(total * (1 - mask_ratio_total))
        alpha    = torch.ones(4) * self.dirichlet_alpha
        lambdas  = torch.distributions.Dirichlet(alpha).sample()

        def _nvis(lam, length):
            return max(1, min(int(lam.item() * nv_total),
                               max(1, int(length * (1 - min_mask_ratio)))))

        nv = [_nvis(lambdas[i], L[k])
              for i, k in enumerate(['rgb','depth','pc','text'])]

        def _mask(x, nv_, L_):
            noise    = torch.rand(B, L_, device=x.device)
            ids_shuf = torch.argsort(noise, dim=1)
            ids_rest = torch.argsort(ids_shuf, dim=1)
            ids_keep = ids_shuf[:, :nv_]
            x_vis    = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1,-1,x.shape[2]))
            mask     = torch.ones(B, L_, device=x.device)
            mask[:, :nv_] = 0
            mask     = torch.gather(mask, 1, ids_rest)
            return x_vis, mask, ids_rest

        xr_v, mr, rr = _mask(x_rgb,   nv[0], L['rgb'])
        xd_v, md, rd = _mask(x_depth, nv[1], L['depth'])
        xp_v, mp, rp = _mask(x_pc,    nv[2], L['pc'])
        xt_v, mt, rt = _mask(x_text,  nv[3], L['text'])

        return (xr_v, xd_v, xp_v, xt_v,
                mr, md, mp, mt,
                rr, rd, rp, rt)

    # ── Encoder ───────────────────────────────────────────────────────────────

    def forward_encoder(self, x_rgb, x_depth, x_pc, x_param, mask_ratio=0.75):
        x_rgb   = self.rgb_embed(x_rgb)     + self.pos_embed_2d    + self.modality_embed_rgb
        x_depth = self.depth_embed(x_depth) + self.pos_embed_2d    + self.modality_embed_depth
        x_pc    = self.pc_embed(x_pc)       + self.pos_embed_pc    + self.modality_embed_pc
        x_text  = self.param_embed(x_param) + self.pos_embed_text  + self.modality_embed_text

        (xr_v, xd_v, xp_v, xt_v,
         mr, md, mp, mt,
         rr, rd, rp, rt) = self.random_masking_dirichlet(
            x_rgb, x_depth, x_pc, x_text, mask_ratio)

        x   = torch.cat([xr_v, xd_v, xp_v, xt_v], dim=1)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x   = torch.cat([cls, x], dim=1)
        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.encoder_norm(x)

        return (x,
                mr, md, mp, mt,
                rr, rd, rp, rt,
                xr_v.shape[1], xd_v.shape[1], xp_v.shape[1], xt_v.shape[1])

    # ── Encoder with explicit modality visibility (cross-modal) ─────────────────

    def forward_encoder_select(self, x_rgb, x_depth, x_pc, x_param, visible):
        """Encoder where each modality is EITHER fully visible OR fully masked
        (0 visible tokens), instead of Dirichlet token-level masking.

        `visible` : iterable subset of {'rgb','depth','pc','text'}. Modalities in
        the set keep all their tokens; the rest contribute 0 encoder tokens and
        are entirely reconstructed by the decoder. This is the training-time
        analogue of generate_crossmodal.py's probe.

        Returns the same 13-tuple as forward_encoder so forward_decoder /
        forward_loss can be reused unchanged. mask_* = 1 everywhere for a masked
        modality, 0 everywhere for a visible one.
        """
        visible = set(visible)
        x_rgb   = self.rgb_embed(x_rgb)     + self.pos_embed_2d    + self.modality_embed_rgb
        x_depth = self.depth_embed(x_depth) + self.pos_embed_2d    + self.modality_embed_depth
        x_pc    = self.pc_embed(x_pc)       + self.pos_embed_pc    + self.modality_embed_pc
        x_text  = self.param_embed(x_param) + self.pos_embed_text  + self.modality_embed_text

        B   = x_rgb.shape[0]
        dev = x_rgb.device

        def _sel(name, e):
            L        = e.shape[1]
            ids_rest = torch.arange(L, device=dev).unsqueeze(0).expand(B, L)
            if name in visible:
                return e, torch.zeros(B, L, device=dev), ids_rest
            return e[:, :0], torch.ones(B, L, device=dev), ids_rest

        xr_v, mr, rr = _sel('rgb',   x_rgb)
        xd_v, md, rd = _sel('depth', x_depth)
        xp_v, mp, rp = _sel('pc',    x_pc)
        xt_v, mt, rt = _sel('text',  x_text)

        x   = torch.cat([xr_v, xd_v, xp_v, xt_v], dim=1)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.encoder_norm(x)

        return (x,
                mr, md, mp, mt,
                rr, rd, rp, rt,
                xr_v.shape[1], xd_v.shape[1], xp_v.shape[1], xt_v.shape[1])

    # ── Decoder ───────────────────────────────────────────────────────────────

    def forward_decoder(self, x, rr, rd, rp, rt, lr_, ld_, lp_, lt_,
                        return_features=False):
        x    = self.decoder_embed(x)
        x_nc = x[:, 1:, :]

        xr_v = x_nc[:, :lr_]
        xd_v = x_nc[:, lr_: lr_ + ld_]
        xp_v = x_nc[:, lr_ + ld_: lr_ + ld_ + lp_]
        xt_v = x_nc[:, lr_ + ld_ + lp_:]

        B = x.shape[0]

        def _restore(x_vis, ids_rest, n_tok):
            pad    = self.mask_token.repeat(B, n_tok - x_vis.shape[1], 1)
            x_full = torch.cat([x_vis, pad], dim=1)
            return torch.gather(x_full, 1,
                                ids_rest.unsqueeze(-1).expand(-1,-1,x_full.shape[2]))

        xr = _restore(xr_v, rr, self.num_patches)   + self.decoder_pos_embed_2d
        xd = _restore(xd_v, rd, self.num_patches)   + self.decoder_pos_embed_2d
        xp = _restore(xp_v, rp, self.num_pc_tokens) + self.decoder_pos_embed_pc
        xt = _restore(xt_v, rt, self.n_text_tokens) + self.decoder_pos_embed_text

        x = torch.cat([xr, xd, xp, xt], dim=1)
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # Token-aligned decoder features (B, L_total, decoder_dim). The layout is
        # [rgb (num_patches) | depth (num_patches) | pc (num_pc_tokens) |
        #  text (n_text_tokens)] and is IDENTICAL regardless of which tokens were
        # masked — so these line up position-for-position between any two forward
        # passes (e.g. a full-modal teacher and a single-modal student). Used as
        # the feature-distillation target.
        feats = x

        n1 = self.num_patches
        n2 = 2 * n1
        n3 = n2 + self.num_pc_tokens

        pred_rgb    = self.decoder_pred_rgb(x[:, :n1])
        pred_depth  = self.decoder_pred_depth(x[:, n1:n2])
        pred_params = self.decoder_pred_params(x[:, n3:])      # (B, n_text_tokens, N_PARAMS)

        # PC folding
        x_pc_dec    = x[:, n2:n3]
        B2, N_tok, _ = x_pc_dec.shape
        pc_feat     = self.decoder_pc_proj(x_pc_dec)
        gs  = int(np.ceil(np.sqrt(self.points_per_token)))
        g1  = torch.linspace(-1, 1, gs, device=x.device)
        gy, gx = torch.meshgrid(g1, g1, indexing='ij')
        grid = torch.stack([gx, gy], dim=-1).reshape(-1, 2)[:self.points_per_token]
        grid = grid.unsqueeze(0).unsqueeze(0).expand(B2, N_tok, -1, -1)
        pc_feat_e = pc_feat.unsqueeze(2).expand(-1,-1,self.points_per_token,-1)
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

    # ── Loss ─────────────────────────────────────────────────────────────────

    def forward_loss(self,
                     imgs_rgb, imgs_depth, pc, param_floats, text_valid,
                     pred_rgb, pred_depth, pred_pc, pred_params,
                     mask_rgb, mask_depth, mask_pc, mask_text):
        # RGB
        tgt_rgb = self.patchify(imgs_rgb, self.patch_size, imgs_rgb.shape[1])
        if self.norm_pix_loss:
            m = tgt_rgb.mean(-1, keepdim=True)
            v = tgt_rgb.var(-1,  keepdim=True)
            tgt_rgb = (tgt_rgb - m) / (v + 1e-6) ** .5
        loss_rgb = ((pred_rgb - tgt_rgb) ** 2).mean(-1)
        # clamp(min=1): when RGB is the fully-visible cross-modal source, mask_rgb
        # is all-zero → guard against 0/0 (yields 0 loss, no NaN). No-op for normal
        # Dirichlet masking where some RGB tokens are always masked.
        loss_rgb = (loss_rgb * mask_rgb).sum() / mask_rgb.sum().clamp(min=1)

        # Depth
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
        loss_depth = ((pred_depth - tgt_d) ** 2).mean(-1)
        loss_depth = (loss_depth * mask_depth).sum() / mask_depth.sum().clamp(min=1)

        # PC (Chamfer)
        loss_pc = chamfer_distance(pred_pc, pc)

        # Parametric text — Smooth-L1 (Huber) on normalised float params.
        # All targets and predictions are in [0, 1] → well-conditioned gradients.
        # Loss is only over tokens that are (a) masked and (b) real (text_valid=1).
        B, L, P = pred_params.shape                        # P = N_PARAMS
        # Smooth-L1 per position: β=0.1 → L2 behaviour near zero for precise values
        per_pos   = F.smooth_l1_loss(
            pred_params.reshape(B * L, P),
            param_floats.reshape(B * L, P),
            reduction='none', beta=0.1,
        ).mean(-1).reshape(B, L)                           # (B, L)

        eff_mask  = mask_text * text_valid                 # (B, L)
        denom     = eff_mask.sum().clamp(min=1)
        loss_text = (per_pos * eff_mask).sum() / denom

        total = (loss_rgb
                 + loss_depth
                 + loss_pc    * self.pc_loss_weight
                 + loss_text  * self.spline_loss_weight)

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
        """
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
                mask_ratio: float = 0.75, visible=None, return_features: bool = False):
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
                imgs_rgb, imgs_depth, pc, param_floats, visible)

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
    model  = embodied_mae_4m_base(target_points=8196).to(device)

    B          = 2
    n_text     = 1 + model.max_leaves
    rgb        = torch.randn(B, 3, 224, 224, device=device)
    depth      = torch.randn(B, 1, 224, 224, device=device)
    pc         = torch.randn(B, 8196, 3, device=device)
    params     = torch.rand(B, n_text, N_PARAMS, device=device)
    text_valid = torch.ones(B, n_text, device=device)
    text_valid[:, 22:] = 0

    total, (lr, ld, lp, lt), (_, _, _, pred_params), _ = model(
        rgb, depth, pc, params, text_valid)

    print(f"Total  : {total.item():.4f}")
    print(f"  RGB  : {lr.item():.4f}")
    print(f"  Depth: {ld.item():.4f}")
    print(f"  PC   : {lp.item():.6f}")
    print(f"  Param: {lt.item():.6f}  (Smooth-L1, params in [0,1])")
    print(f"\nDecoded sample text[0][0]: '{model.decode_params_to_text(pred_params)[0][0]}'")
    print(f"Decoded sample text[0][1]: '{model.decode_params_to_text(pred_params)[0][1]}'")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
