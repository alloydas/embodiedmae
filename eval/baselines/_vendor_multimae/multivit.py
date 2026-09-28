# Copyright (c) EPFL VILAB. All rights reserved.
# Licensed CC BY-NC 4.0: see LICENSE in this directory.
# --------------------------------------------------------
# Adapted from https://github.com/EPFL-VILAB/MultiMAE
# at commit 66910f5b5ba236f5e731883db85fe4f24ee01106, which is itself based on
# timm, DeiT, DINO, MoCo-v3, BEiT, MAE-priv and MAE. Source files:
#   multimae/multimae_utils.py  build_2d_sincos_posemb :29-45, Mlp :138-155,
#                               Attention :158-182, Block :217-232
#   multimae/input_adapters.py  PatchedInputAdapter :27-119
#   multimae/multimae.py        MultiMAE.__init__ :61-98 (module and key names),
#                               MultiViT.process_input :439-467 and
#                               forward :469-491 (unmasked forward),
#                               multivit_base :505-520 (hyperparameters)
# --------------------------------------------------------
"""The MultiMAE-B encoder, RGB + depth only, pure torch.

Only the forward pass is vendored. The graph and the state-dict key names are
upstream's, so the release checkpoint's encoder keys load with strict=True and
no remapping. What was changed, and why none of it touches the numbers:
  * einops `rearrange` became flatten/transpose/reshape, and `repeat` became
    expand. Same memory order.
  * DropPath and Dropout are gone. Upstream builds them with rate 0 for this
    model, they hold no parameters, and they are identities in eval().
  * No `utils.registry`, no timm, no einops. See __init__.py for why.
  * `torch.meshgrid(..., indexing='ij')` is spelled out. Upstream relies on the
    default, which is 'ij' and warns under torch>=1.10.
  * There is no semseg input adapter and no output adapters (decoders). An
    input adapter only runs for a modality that is fed, and the adapters share
    no weights, so leaving semseg out changes nothing for an RGB+D input. The
    scout confirmed this: upstream MultiViT built with and without the semseg
    adapter gives max|diff| 0.0 on an RGB+D input.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_2d_sincos_posemb(h, w, embed_dim=768, temperature=10000.):
    """multimae_utils.py:29-45 (MoCo-v3's). Returns (1, embed_dim, h, w).

    The release checkpoint stores these as a frozen Parameter
    (requires_grad=False, input_adapters.py:81-82), so the loaded values
    replace these. build() asserts the two agree, which confirms that the
    checkpoint is the sin-cos variant.
    """
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
    assert embed_dim % 4 == 0, 'embed_dim must be divisible by 4 for 2D sin-cos'
    pos_dim = embed_dim // 4
    omega = 1. / (temperature ** (torch.arange(pos_dim, dtype=torch.float32) / pos_dim))
    out_w = grid_w.flatten()[:, None] * omega[None, :]
    out_h = grid_h.flatten()[:, None] * omega[None, :]
    pos_emb = torch.cat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)
    # 'b (h w) d -> b d h w'
    return pos_emb.reshape(h, w, embed_dim).permute(2, 0, 1).unsqueeze(0).contiguous()


class Mlp(nn.Module):
    """multimae_utils.py:138-155; the two Dropout(0) are identities."""
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()                       # exact erf GELU, as upstream
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    """multimae_utils.py:158-182. The attention matrix is explicit, not SDPA,
    so the result is bit-identical to upstream's."""
    def __init__(self, dim, num_heads, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(B, N, C))


class Block(nn.Module):
    """multimae_utils.py:217-232, pre-norm. LayerNorm eps 1e-6 per multimae.py:73/:518."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, eps=1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class PatchedInputAdapter(nn.Module):
    """input_adapters.py:27-119, with stride_level 1 and sin-cos pos emb (the
    rgb and depth entries of run_pretraining_multimae.py:49-63 DOMAIN_CONF)."""
    def __init__(self, num_channels, dim_tokens=768, patch_size=16, image_size=224):
        super().__init__()
        self.P = patch_size
        n = image_size // patch_size
        self.pos_emb = nn.Parameter(build_2d_sincos_posemb(n, n, dim_tokens),
                                    requires_grad=False)
        self.proj = nn.Conv2d(num_channels, dim_tokens, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.P == 0 and W % self.P == 0, f'{H}x{W} not divisible by patch {self.P}'
        x_patch = self.proj(x).flatten(2).transpose(1, 2)              # b d nh nw -> b (nh nw) d
        # input_adapters.py:113. At 224 px this is a 14x14 -> 14x14 resample,
        # an identity. It is kept so that any other size behaves as upstream's.
        pe = F.interpolate(self.pos_emb, size=(H // self.P, W // self.P),
                           mode='bicubic', align_corners=False)
        return x_patch + pe.flatten(2).transpose(1, 2)


class MultiViTEncoder(nn.Module):
    """MultiViT (multimae.py:419-491) restricted to rgb + depth inputs.

    forward({'rgb': (B,3,H,W), 'depth': (B,1,H,W)}) -> (B, N_rgb + N_depth + 1, 768).
    Tokens are concatenated in the input dict's order, then the ONE global
    token is APPENDED LAST (multimae.py:461-465). It is tokens[:, -1], not
    tokens[:, 0]. Copying linear_probe's `latent[:, 0]` would take the top-left
    RGB patch, which is background. There is no final norm: upstream's encoder
    is `nn.Sequential(Block, ...)` (multimae.py:94-98), and the checkpoint has
    no norm key.
    """
    DOMAINS = {'rgb': 3, 'depth': 1}

    def __init__(self, dim_tokens=768, depth=12, num_heads=12, mlp_ratio=4.,
                 num_global_tokens=1, patch_size=16, image_size=224):
        super().__init__()
        self.input_adapters = nn.ModuleDict({
            k: PatchedInputAdapter(c, dim_tokens, patch_size, image_size)
            for k, c in self.DOMAINS.items()})
        self.num_global_tokens = num_global_tokens
        self.global_tokens = nn.Parameter(torch.zeros(1, num_global_tokens, dim_tokens))
        self.encoder = nn.Sequential(*[
            Block(dim_tokens, num_heads, mlp_ratio, qkv_bias=True) for _ in range(depth)])

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        unknown = set(x) - set(self.input_adapters)
        if unknown:
            # Upstream drops an unknown domain silently (multimae.py:454-458),
            # which would turn an RGB+D row into an RGB-only one. Refuse instead.
            raise KeyError(f'no input adapter for {sorted(unknown)}')
        tokens = [self.input_adapters[k](v) for k, v in x.items()]
        B = tokens[0].shape[0]
        tokens.append(self.global_tokens.expand(B, -1, -1))
        return self.encoder(torch.cat(tokens, dim=1))
