"""Point-MAE, vendored as pure torch: no pointnet2_ops, no knn_cuda, no timm.

UPSTREAM
  repo    https://github.com/Pang-Yatian/Point-MAE  (MIT, see ./LICENSE)
  commit  7445a68068d707da6d959b66773bfef84c4e3e32  (main, 2025-03-31)
  files   models/Point_MAE.py  (last changed 336d42f, 2022-03-09)
          utils/misc.py        fps()
          FPS kernel: erikwijmans/Pointnet2_PyTorch
                      pointnet2_ops_lib/pointnet2_ops/_ext-src/src/sampling_gpu.cu

Every nn.Module below keeps upstream's class structure and attribute names, so
the release checkpoint's keys map 1:1 once the DDP 'module.' prefix is gone.
The layer code is transcribed from models/Point_MAE.py; line numbers are
cited per class. What changed, and why:

  * Dropout / DropPath are gone. Upstream builds them with p=0 at eval
    (attn_drop=0, proj_drop=0, drop=0), and drop_path is identity in eval().
    Neither holds parameters, so the key set is unchanged.
  * misc.fps (pointnet2_ops CUDA) and knn_cuda.KNN are replaced by the torch
    functions below. pointnet2_ops needs a CUDA build against the installed
    torch, and the knn_cuda wheel URL in upstream README.md:62 returned HTTP
    404 when checked on 2026-09-25. So the upstream install path is broken,
    and a torch port is required, not optional.
  * timm.trunc_normal_ -> torch.nn.init.trunc_normal_ (same a=-2, b=2 defaults).
    This only matters for the random-init reference in the self-check.
"""

import contextlib

import torch
import torch.nn as nn


# ─────────────────────────── FPS + kNN (pure torch) ──────────────────────────

@contextlib.contextmanager
def _one_cpu_thread(device):
    """Run the FPS loop single-threaded on CPU, then restore the thread count.

    Each of its ~1000 steps is a handful of (B, N) elementwise ops, far too
    small to amortise OpenMP's fork/join. Measured, B=32 x 8196 -> 1024: 6.5 s
    on 1 thread, 155 s on 4 threads on a loaded node (load average ~70 on 16
    cores). A CPU-only probe job (--gres=NONE, CLAUDE.md) would otherwise
    spend hours here.
    """
    if torch.device(device).type != 'cpu':
        yield
        return
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(n)


def furthest_point_sample(xyz, m):
    """pointnet2_ops.furthest_point_sample, reproducing the CUDA kernel's rules.

    sampling_gpu.cu furthest_point_sampling_kernel:
      * the first index is ALWAYS 0 (`int old = 0; idxs[0] = old;`, :85-86).
        So the result is a deterministic function of point order, and the
        probe's seeded loader fixes that order.
      * points with |p|^2 <= 1e-3 are skipped and can never be selected after
        the first (`if (mag <= 1e-3) continue;`, :100-101). Upstream feeds
        centred, unit-sphere clouds, so this excludes a 0.032-radius ball round
        the centroid: 0.52% of a sorghum cloud. A textbook FPS would pick
        different centres near the stem, so the quirk is kept.
      * temp starts at 1e10 (python wrapper) and d2 = min(d, temp[k]).
      * argmax ties go to the lowest index, matching the kernel's strict '>'
        within a thread. Ties across threads can resolve differently in the
        kernel, and nvcc may fuse the distance into FMAs (a last-ulp
        difference), so a near-exact float tie could pick another centre.
        Neither is a practical case on a real scan.
    Distances are exact coordinate differences in the kernel's order,
    dx*dx + dy*dy + dz*dz, on planar (B, 3, N) storage, not the
    |a|^2+|b|^2-2ab matmul form, which can reorder near-ties.

    xyz (B, N, 3) float -> (B, m) long.
    """
    B, N, _ = xyz.shape
    dev = xyz.device
    with _one_cpu_thread(dev):
        P = xyz.transpose(1, 2).contiguous()                 # (B, 3, N)
        X, Y, Z = P[:, 0], P[:, 1], P[:, 2]
        skip = (X * X + Y * Y + Z * Z) <= 1e-3
        # Skipped points hold -1 for good: min(-1, d>=0) stays -1, so they are
        # never updated and never win the argmax, exactly as `continue` does.
        temp = torch.full((B, N), 1e10, dtype=xyz.dtype, device=dev).masked_fill_(skip, -1.0)
        idx = torch.zeros(B, m, dtype=torch.long, device=dev)
        old = torch.zeros(B, 1, dtype=torch.long, device=dev)
        d = torch.empty_like(temp)
        t = torch.empty_like(temp)
        for j in range(1, m):
            torch.sub(X, X.gather(1, old), out=t)
            torch.mul(t, t, out=d)
            torch.sub(Y, Y.gather(1, old), out=t)
            d.addcmul_(t, t)
            torch.sub(Z, Z.gather(1, old), out=t)
            d.addcmul_(t, t)
            torch.minimum(temp, d, out=temp)
            old = temp.argmax(1, keepdim=True)                # first max = lowest index
            idx[:, j] = old[:, 0]
    return idx


def gather_points(xyz, idx):
    """pointnet2_ops.gather_operation, for (B, N, 3) x (B, ...) -> (B, ..., 3)."""
    B = xyz.shape[0]
    flat = idx.reshape(B, -1)
    out = torch.gather(xyz, 1, flat[..., None].expand(-1, -1, xyz.shape[-1]))
    return out.reshape(*idx.shape, xyz.shape[-1])


def fps(xyz, m):
    """utils/misc.py fps(): FPS indices, then the gathered points. (B, m, 3)."""
    return gather_points(xyz, furthest_point_sample(xyz, m))


def knn(ref, query, k):
    """knn_cuda.KNN(k, transpose_mode=True)(ref, query)[1] -> (B, M, k) indices.

    The ascending k nearest reference points to each query point. The query
    points ARE reference points here (FPS centres), so each centre is its own
    first neighbour, as with knn_cuda. The order inside a group is irrelevant,
    because the mini-PointNet max-pools over it.
    """
    d = ((query[:, :, None, :] - ref[:, None, :, :]) ** 2).sum(-1)   # (B, M, N)
    return d.topk(k, dim=-1, largest=False).indices


# ─────────────────────────────── model ───────────────────────────────────────

class Encoder(nn.Module):   # models/Point_MAE.py:16-47, verbatim apart from comments
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        feature = self.first_conv(point_groups.transpose(2, 1))              # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]         # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)   # BG 512 n
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):     # models/Point_MAE.py:50-79, FPS + kNN swapped for the torch ports
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz):
        """xyz (B, N, 3) -> neighborhood (B, G, M, 3) relative to its centre, center (B, G, 3)."""
        center = fps(xyz, self.num_group)                    # misc.fps(xyz, num_group)
        idx = knn(xyz, center, self.group_size)              # self.knn(xyz, center)
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        neighborhood = gather_points(xyz, idx)               # B G M 3
        neighborhood = neighborhood - center.unsqueeze(2)    # upstream "# normalize"
        return neighborhood, center


class Mlp(nn.Module):       # models/Point_MAE.py:82-98, Dropout(0) removed
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):  # models/Point_MAE.py:101-125, Dropout(0) removed
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Block(nn.Module):      # models/Point_MAE.py:128-146, DropPath (identity at eval) removed
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):   # models/Point_MAE.py:149-165
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale)
            for _ in range(depth)])

    def forward(self, x, pos):
        # The position embedding is re-added before EVERY block (:163-164),
        # not once at the input as in a ViT. A port that adds it once loads
        # strictly and runs, and gives different features.
        for block in self.blocks:
            x = block(x + pos)
        return x


class TransformerDecoder(nn.Module):   # models/Point_MAE.py:168-198
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        for block in self.blocks:
            x = block(x + pos)
        return self.head(self.norm(x[:, -return_token_num:]))


class MaskTransformer(nn.Module):   # models/Point_MAE.py:202-325, config -> kwargs
    """The pretrained encoder. Its state_dict is ckpt['base_model']['module.MAE_encoder.*'].

    Defaults are cfgs/pretrain.yaml: trans_dim = encoder_dims = 384, depth 12,
    6 heads, mask_ratio 0.6, mask_type 'rand'. There is no CLS token:
    PointTransformer's cls_token/cls_pos (:436-437) are fine-tuning parameters
    and are not in the pretrain checkpoint.
    """

    def __init__(self, trans_dim=384, encoder_dims=384, depth=12, num_heads=6, mask_ratio=0.6):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.trans_dim = trans_dim
        self.encoder = Encoder(encoder_channel=encoder_dims)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, trans_dim),
        )
        self.blocks = TransformerEncoder(embed_dim=trans_dim, depth=depth, num_heads=num_heads)
        self.norm = nn.LayerNorm(trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):   # :236-247
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, neighborhood, center, bool_masked_pos=None):
        """Upstream forward (:303-325). bool_masked_pos=None is upstream's
        noaug=True path (:286-287: an all-False mask), i.e. every group is visible.

        Upstream draws the pretraining mask inside forward with np.random.shuffle
        (_mask_center_rand, :278-300). Here the caller passes it, so the
        self-check can apply the SAME mask to the pretrained and random models.
        """
        group_input_tokens = self.encoder(neighborhood)       # B G C
        B, G, C = group_input_tokens.shape
        if bool_masked_pos is None:
            bool_masked_pos = torch.zeros(B, G, dtype=torch.bool, device=center.device)
        x_vis = group_input_tokens[~bool_masked_pos].reshape(B, -1, C)
        masked_center = center[~bool_masked_pos].reshape(B, -1, 3)
        pos = self.pos_embed(masked_center)
        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)
        return x_vis, bool_masked_pos


def rand_mask(B, G, mask_ratio, generator=None, device='cpu'):
    """_mask_center_rand (:278-300): exactly int(ratio*G) masked groups per sample, uniform."""
    num_mask = int(mask_ratio * G)
    keys = torch.rand(B, G, generator=generator)
    order = keys.argsort(dim=1)
    mask = torch.zeros(B, G, dtype=torch.bool)
    mask.scatter_(1, order[:, :num_mask], True)
    return mask.to(device)


class PointMAE(nn.Module):   # models/Point_MAE.py:328-418 (class Point_MAE), for the self-check only
    """Full pretraining model: encoder + decoder + point head.

    The adapter's features never touch the decoder. It exists so the
    self-check can load ALL 209 checkpoint keys strictly and measure
    masked-group reconstruction, which is the only way to show that the
    weights are the pretrained ones AND that the inputs are in the convention
    they were trained on.
    """

    def __init__(self, trans_dim=384, depth=12, num_heads=6, num_group=64, group_size=32,
                 decoder_depth=4, decoder_num_heads=6, mask_ratio=0.6):
        super().__init__()
        self.trans_dim = trans_dim
        self.MAE_encoder = MaskTransformer(trans_dim=trans_dim, encoder_dims=trans_dim,
                                           depth=depth, num_heads=num_heads,
                                           mask_ratio=mask_ratio)
        self.group_size = group_size
        self.num_group = num_group
        self.mask_token = nn.Parameter(torch.zeros(1, 1, trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, trans_dim)
        )
        self.MAE_decoder = TransformerDecoder(embed_dim=trans_dim, depth=decoder_depth,
                                              num_heads=decoder_num_heads)
        self.group_divider = Group(num_group=num_group, group_size=group_size)
        self.increase_dim = nn.Sequential(nn.Conv1d(trans_dim, 3 * group_size, 1))
        nn.init.trunc_normal_(self.mask_token, std=.02)

    def reconstruct(self, neighborhood, center, mask):
        """Upstream forward (:381-418) up to the loss: (rebuilt, target) for the masked groups.

        Both are (B*M, group_size, 3) in centre-relative coordinates, as upstream
        passes them to ChamferDistanceL2.
        """
        x_vis, mask = self.MAE_encoder(neighborhood, center, mask)
        B, _, C = x_vis.shape
        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)
        _, N, _ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, N, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)
        x_rec = self.MAE_decoder(x_full, pos_full, N)
        B, M, C = x_rec.shape
        rebuild = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)
        gt = neighborhood[mask].reshape(B * M, -1, 3)
        return rebuild, gt


def chamfer_l2_per_group(a, b):
    """extensions/chamfer_dist ChamferDistanceL2 (:28-44), per group.

    mean over a of min squared distance to b, plus the reverse. Upstream then
    averages over groups (torch.mean over the batch of dist1 and dist2); the
    per-group values are returned so the caller can average per plant.
    """
    d = ((a[:, :, None, :] - b[:, None, :, :]) ** 2).sum(-1)       # (G, n, m)
    return d.min(2).values.mean(1) + d.min(1).values.mean(1)
