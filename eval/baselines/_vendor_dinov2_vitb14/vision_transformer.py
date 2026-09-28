# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in this directory (a verbatim copy of upstream's).
#
# VENDORED from https://github.com/facebookresearch/dinov2
#   commit 7764ea0f912e53c92e82eb78a2a1631e92725fc8 (2026-06-03, "Safely load
#   weights from specified URLs (#598)")
# Upstream files merged here, with their sha256 at that commit (each checked
# against raw.githubusercontent.com on 2026-09-24):
#   dinov2/models/vision_transformer.py  7799a260f2d7d0fe197331d08502fb8c542f9b7424723650f6a39b64fa2639ea
#   dinov2/layers/attention.py           79c7be7a452b3aad96698ec38d5d5150b9f4d8ac084fa93324510dc9f624775d
#   dinov2/layers/block.py               60c0ac7dfa4474be313fabfa5a23d82faf6f0cecd4e720a88be35de9788cb636
#   dinov2/layers/mlp.py                 255825c73b60a916dd00eb1e38aacbcdbf316e40d6a005efb46e245b7edb43aa
#   dinov2/layers/patch_embed.py         40da6add3d811198ea3e17cb99cdd4e5cda59e369efbbe3d18d89308618cf142
#   dinov2/layers/layer_scale.py         dadd5aafe178f1bf72a205a02a6645c7e635cacbad585d4a7369c200c6e89135
#   dinov2/hub/backbones.py              871fca671b12a9ff02e810654baf509e97ccf461bf8196ce5ddeefff2fd87d3e
#   LICENSE                              600cc67cc4cb2f5ea317dcfc687ad1c74dc4bec8782bbe9db0afd83513b935b7
#
# MODIFICATIONS (Apache-2.0 s.4(b)), all confined to paths the frozen
# ViT-B/14 forward never takes, so the kept path is upstream's line for line:
#   * six files merged into one; logging, flops() and typing clutter removed.
#   * xFormers removed. Upstream's MemEffAttention.forward calls
#     Attention.forward (F.scaled_dot_product_attention) whenever xFormers is
#     absent, which is the case in both `det` and `det_cu128`, so Attention is
#     used directly. The three "xFormers is not available" warnings go with it.
#   * Training-only and other-architecture paths removed: NestedTensorBlock's
#     list path, stochastic depth / DropPath (drop_path_rate is 0 for the hub
#     backbone anyway), BlockChunk (block_chunks=0 for the hub backbone),
#     register tokens (0 for vitb14), SwiGLU FFN (vit_giant2 only),
#     channel_adaptive / bag_of_channels (Cell-DINO), forward_features_list,
#     get_intermediate_layers.
#   * `mask_token` is kept although inference never reads it, so the released
#     checkpoint loads key-for-key under strict=True with nothing dropped.
#   * vit_base_14() inlines hub/backbones.py's dinov2_vitb14 construction
#     arguments; it never downloads (the adapter owns fetching and verifying).
#
# Why vendor at all instead of torch.hub.load: the pinned hubconf imports the
# whole hub package (Cell-DINO, XRay-DINO, dino.txt with ftfy/regex), needs
# trust_repo and a GitHub API call on first load, and executes whatever code
# sits in the cache directory. This file is pure torch, so it runs unchanged
# in det (torch 2.5) and det_cu128 (torch 2.11, no timm), offline.
# baselines/dinov2_vitb14.py records the bitwise check against the hub model.

import math
from functools import partial

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.init import trunc_normal_


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


# ── dinov2/layers/patch_embed.py ─────────────────────────────────────────────
class PatchEmbed(nn.Module):
    """2D image to patch embedding: (B,C,H,W) -> (B,N,D)"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 norm_layer=None, flatten_embedding=True):
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x


# ── dinov2/layers/attention.py (Attention; MemEffAttention without xFormers) ─
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, proj_bias=True,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, is_causal: bool = False) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        x = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.attn_drop if self.training else 0, is_causal=is_causal
        )
        x = x.transpose(1, 2).contiguous().view(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


# ── dinov2/layers/mlp.py ─────────────────────────────────────────────────────
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0, bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ── dinov2/layers/layer_scale.py ─────────────────────────────────────────────
class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False, device=None, dtype=None):
        super().__init__()
        self.inplace = inplace
        self.init_values = init_values
        self.gamma = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.gamma, self.init_values)

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


# ── dinov2/layers/block.py (Block; eval path of NestedTensorBlock on a Tensor)
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, proj_bias=True,
                 ffn_bias=True, drop=0.0, attn_drop=0.0, init_values=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_class=Attention,
                 ffn_layer=Mlp):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        def attn_residual_func(x: Tensor) -> Tensor:
            return self.ls1(self.attn(self.norm1(x)))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        # upstream's `else` branch: no stochastic depth (drop_path 0 / eval)
        x = x + attn_residual_func(x)
        x = x + ffn_residual_func(x)
        return x


# ── dinov2/models/vision_transformer.py ─────────────────────────────────────
def named_apply(fn, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class DinoVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4.0, qkv_bias=True, ffn_bias=True, proj_bias=True,
                 init_values=None, embed_layer=PatchEmbed, act_layer=nn.GELU,
                 block_fn=Block, interpolate_antialias=False, interpolate_offset=0.1):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = 0
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))

        self.blocks = nn.ModuleList([
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=Mlp,
                init_values=init_values,
            )
            for _ in range(depth)
        ])

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        # Unused at inference; kept so the released checkpoint loads strict.
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        return x

    def forward_features(self, x, masks=None):
        x = self.prepare_tokens_with_masks(x, masks)

        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def vit_base_14():
    """hub/backbones.py dinov2_vitb14(pretrained=False), weights NOT loaded.

    _make_dinov2_model defaults: img_size 518 (so pos_embed is 1+37*37 and is
    interpolated to the input grid), patch 14, init_values 1.0 (LayerScale
    present), ffn 'mlp', block_chunks 0, 0 register tokens, interpolate_
    antialias False, interpolate_offset 0.1; vit_base: 768 wide, 12 deep,
    12 heads, mlp_ratio 4.
    """
    return DinoVisionTransformer(
        img_size=518,
        patch_size=14,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        init_values=1.0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    )
