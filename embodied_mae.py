"""
EmbodiedMAE: A Unified 3D Multi-Modal Representation for Robot Manipulation
PyTorch Implementation

Paper: https://arxiv.org/abs/2505.10105
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def chamfer_distance(pred_pc, target_pc):
    """
    Compute Chamfer Distance between predicted and target point clouds
    
    Args:
        pred_pc: (B, N, 3) predicted point cloud
        target_pc: (B, M, 3) target point cloud
        
    Returns:
        chamfer_dist: scalar Chamfer distance
    """
    # pred_pc: (B, N, 3)
    # target_pc: (B, M, 3)
    
    # Expand dimensions for broadcasting
    # pred_pc: (B, N, 1, 3)
    # target_pc: (B, 1, M, 3)
    pred_expanded = pred_pc.unsqueeze(2)
    target_expanded = target_pc.unsqueeze(1)
    
    # Compute pairwise distances: (B, N, M)
    distances = torch.sum((pred_expanded - target_expanded) ** 2, dim=-1)
    
    # Forward direction: for each point in pred, find nearest in target
    min_dist_pred_to_target, _ = torch.min(distances, dim=2)  # (B, N)
    forward_loss = torch.mean(min_dist_pred_to_target)
    
    # Backward direction: for each point in target, find nearest in pred
    min_dist_target_to_pred, _ = torch.min(distances, dim=1)  # (B, M)
    backward_loss = torch.mean(min_dist_target_to_pred)
    
    # Chamfer distance is the sum of both directions
    chamfer_dist = forward_loss + backward_loss
    
    return chamfer_dist


def earth_movers_distance(pred_pc, target_pc, num_samples=1000):
    """
    Compute Earth Mover's Distance (EMD) / Wasserstein Distance between point clouds
    
    Uses approximate EMD for efficiency with large point clouds:
    - Subsamples points for tractable computation
    - Uses Hungarian algorithm for optimal matching
    
    Args:
        pred_pc: (B, N, 3) predicted point cloud
        target_pc: (B, M, 3) target point cloud
        num_samples: Number of points to use for EMD calculation (default: 1000)
        
    Returns:
        emd: scalar Earth Mover's Distance
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        print("⚠️  scipy not available, EMD will return 0. Install with: pip install scipy")
        return torch.tensor(0.0, device=pred_pc.device)
    
    B, N, _ = pred_pc.shape
    _, M, _ = target_pc.shape
    
    # For large point clouds, subsample for efficiency
    if N > num_samples:
        # Randomly sample points from pred
        indices_pred = torch.randperm(N, device=pred_pc.device)[:num_samples]
        pred_sampled = pred_pc[:, indices_pred, :]
    else:
        pred_sampled = pred_pc
        num_samples = N
    
    if M > num_samples:
        # Randomly sample points from target
        indices_target = torch.randperm(M, device=target_pc.device)[:num_samples]
        target_sampled = target_pc[:, indices_target, :]
    else:
        target_sampled = target_pc
    
    # Compute EMD for each batch
    emd_list = []
    for b in range(B):
        pred_b = pred_sampled[b].cpu().detach().numpy()  # (num_samples, 3)
        target_b = target_sampled[b].cpu().detach().numpy()  # (num_samples, 3)
        
        # Compute pairwise distance matrix
        # dist_matrix[i, j] = distance from pred[i] to target[j]
        dist_matrix = np.sqrt(np.sum((pred_b[:, None, :] - target_b[None, :, :]) ** 2, axis=-1))
        
        # Solve optimal assignment problem (Hungarian algorithm)
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        
        # EMD is the mean distance of optimal assignment
        emd_b = dist_matrix[row_ind, col_ind].mean()
        emd_list.append(emd_b)
    
    # Average across batch
    emd = torch.tensor(np.mean(emd_list), device=pred_pc.device, dtype=torch.float32)
    
    return emd


from typing import Tuple, Optional, List
import math


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding for RGB and Depth
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class PointCloudEmbed(nn.Module):
    """
    Point Cloud to Token Embedding
    Uses FPS (Farthest Point Sampling) and grouping
    """
    def __init__(self, num_tokens=196, group_size=32, embed_dim=768):
        super().__init__()
        self.num_tokens = num_tokens
        self.group_size = group_size
        
        # Local feature extraction
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
            nn.Conv1d(512, embed_dim, 1)
        )
        
    def fps(self, xyz, npoint):
        """
        Farthest Point Sampling
        """
        device = xyz.device
        B, N, C = xyz.shape
        centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
        batch_indices = torch.arange(B, dtype=torch.long).to(device)
        
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
            
        return centroids
    
    def knn_grouping(self, xyz, points, centroids_idx):
        """
        K-NN grouping around sampled centers
        """
        B, N, C = xyz.shape
        _, S, _ = centroids_idx.shape if len(centroids_idx.shape) == 3 else (B, centroids_idx.shape[1], 3)
        
        centroids = xyz[torch.arange(B)[:, None], centroids_idx]
        
        # Compute distances
        dist = torch.cdist(centroids, xyz)  # (B, S, N)
        
        # Get k nearest neighbors
        _, idx = torch.topk(dist, self.group_size, dim=2, largest=False)
        
        # Group points
        grouped_xyz = xyz[torch.arange(B)[:, None, None], idx]  # (B, S, k, 3)
        grouped_xyz = grouped_xyz - centroids.unsqueeze(2)  # Normalize
        
        if points is not None:
            grouped_points = points[torch.arange(B)[:, None, None], idx]
            grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
        else:
            grouped_points = grouped_xyz
            
        return grouped_points
        
    def forward(self, xyz):
        """
        Args:
            xyz: (B, N, 3) point cloud
        Returns:
            tokens: (B, num_tokens, embed_dim)
        """
        B, N, _ = xyz.shape
        
        # Farthest point sampling
        fps_idx = self.fps(xyz, self.num_tokens)
        
        # Grouping
        grouped_points = self.knn_grouping(xyz, None, fps_idx)  # (B, num_tokens, group_size, 3)
        
        # Local feature extraction
        grouped_points = grouped_points.permute(0, 1, 3, 2)  # (B, num_tokens, 3, group_size)
        B, S, C, K = grouped_points.shape
        grouped_points = grouped_points.reshape(B * S, C, K)
        
        features = self.first_conv(grouped_points)  # (B*S, 256, K)
        features_global = torch.max(features, dim=2)[0]  # (B*S, 256)
        features = torch.cat([features, features_global.unsqueeze(-1).expand(-1, -1, K)], dim=1)
        
        features = self.second_conv(features)  # (B*S, embed_dim, K)
        features = torch.max(features, dim=2)[0]  # (B*S, embed_dim)
        
        tokens = features.reshape(B, S, -1)  # (B, num_tokens, embed_dim)
        
        return tokens


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    Generate 2D sinusoidal positional embeddings
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    
    # Use half of dimensions for x and y
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class EmbodiedMAE(nn.Module):
    """
    EmbodiedMAE: Multi-modal Masked Autoencoder for RGB, Depth, and Point Cloud
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans_rgb=3,
        in_chans_depth=1,
        num_pc_tokens=196,
        pc_group_size=32,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
        norm_pix_loss=True,
        dirichlet_alpha=1.0,
        pc_loss_weight=50.0,
        target_points=10000,
        normalize_depth_global=True,
        depth_norm_type='minmax'
    ):
        super().__init__()
        
        # Configuration
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.num_pc_tokens = num_pc_tokens
        self.embed_dim = embed_dim
        self.norm_pix_loss = norm_pix_loss
        self.dirichlet_alpha = dirichlet_alpha
        self.pc_loss_weight = pc_loss_weight
        self.normalize_depth_global = normalize_depth_global
        self.depth_norm_type = depth_norm_type  # 'minmax' or 'standard'
        
        # Patch embeddings for each modality
        self.rgb_embed = PatchEmbed(img_size, patch_size, in_chans_rgb, embed_dim)
        self.depth_embed = PatchEmbed(img_size, patch_size, in_chans_depth, embed_dim)
        self.pc_embed = PointCloudEmbed(num_pc_tokens, pc_group_size, embed_dim)
        
        # Modality embeddings
        self.modality_embed_rgb = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.modality_embed_depth = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.modality_embed_pc = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embeddings
        self.pos_embed_2d = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)
        self.pos_embed_pc = nn.Parameter(torch.zeros(1, num_pc_tokens, embed_dim), requires_grad=True)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Encoder
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True)
            for _ in range(depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)
        
        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        self.decoder_pos_embed_2d = nn.Parameter(
            torch.zeros(1, self.num_patches, decoder_embed_dim), requires_grad=False
        )
        self.decoder_pos_embed_pc = nn.Parameter(
            torch.zeros(1, num_pc_tokens, decoder_embed_dim), requires_grad=False
        )
        
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        # Reconstruction heads (modality-specific)
        self.decoder_pred_rgb = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans_rgb, bias=True)
        self.decoder_pred_depth = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans_depth, bias=True)
        
        # Point cloud reconstruction - upsample from tokens to full point cloud
        # Each token will generate multiple points to reach target_points (e.g., 10000)
        self.target_points = target_points  # Target number of output points
        self.points_per_token = self.target_points // num_pc_tokens  # e.g., 10000/196 ≈ 51
        
        # Folding-based upsampling (inspired by FoldingNet and Point-MAE)
        self.decoder_pc_proj = nn.Linear(decoder_embed_dim, 512, bias=True)
        self.decoder_pc_fold = nn.Sequential(
            nn.Linear(512 + 2, 512),  # +2 for 2D grid coordinates
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)  # Output (x, y, z)
        )
        
        self.initialize_weights()
        
    def initialize_weights(self):
        # Initialize positional embeddings
        pos_embed_2d = get_2d_sincos_pos_embed(self.embed_dim, int(self.num_patches**.5))
        self.pos_embed_2d.data.copy_(torch.from_numpy(pos_embed_2d).float().unsqueeze(0))
        
        decoder_pos_embed_2d = get_2d_sincos_pos_embed(
            self.decoder_blocks[0].norm1.normalized_shape[0], int(self.num_patches**.5)
        )
        self.decoder_pos_embed_2d.data.copy_(torch.from_numpy(decoder_pos_embed_2d).float().unsqueeze(0))
        
        # Initialize point cloud positional embeddings randomly
        nn.init.normal_(self.pos_embed_pc, std=.02)
        nn.init.normal_(self.decoder_pos_embed_pc, std=.02)
        
        # Initialize modality embeddings
        nn.init.normal_(self.modality_embed_rgb, std=.02)
        nn.init.normal_(self.modality_embed_depth, std=.02)
        nn.init.normal_(self.modality_embed_pc, std=.02)
        
        # Initialize cls token and mask token
        nn.init.normal_(self.cls_token, std=.02)
        nn.init.normal_(self.mask_token, std=.02)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def random_masking_dirichlet(self, x_rgb, x_depth, x_pc, mask_ratio_total=0.75):
        """
        Random masking using Dirichlet distribution to allocate visible tokens.

        A single Dirichlet sample determines the modality split for the whole batch,
        so all samples have identical visible-token counts per modality. This avoids
        the padding/zero-token corruption that arises from per-sample allocation.
        """
        B = x_rgb.shape[0]
        L_rgb = x_rgb.shape[1]
        L_depth = x_depth.shape[1]
        L_pc = x_pc.shape[1]

        total_tokens = L_rgb + L_depth + L_pc
        num_visible_total = int(total_tokens * (1 - mask_ratio_total))

        # One Dirichlet draw per batch step — varies across iterations, uniform within batch
        alpha = torch.ones(3) * self.dirichlet_alpha
        lambdas = torch.distributions.Dirichlet(alpha).sample()  # (3,)

        # Each modality draws from its own lambda — symmetric allocation.
        # Using the last modality as a remainder biases it toward full visibility.
        num_visible_rgb   = max(1, min(int(lambdas[0].item() * num_visible_total), L_rgb))
        num_visible_depth = max(1, min(int(lambdas[1].item() * num_visible_total), L_depth))
        num_visible_pc    = max(1, min(int(lambdas[2].item() * num_visible_total), L_pc))

        def _mask(x, num_visible, L):
            noise = torch.rand(B, L, device=x.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)

            ids_keep = ids_shuffle[:, :num_visible]
            x_vis = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[2]))

            mask = torch.ones(B, L, device=x.device)
            mask[:, :num_visible] = 0
            mask = torch.gather(mask, 1, ids_restore)

            return x_vis, mask, ids_restore

        x_rgb_vis, mask_rgb, ids_restore_rgb = _mask(x_rgb, num_visible_rgb, L_rgb)
        x_depth_vis, mask_depth, ids_restore_depth = _mask(x_depth, num_visible_depth, L_depth)
        x_pc_vis, mask_pc, ids_restore_pc = _mask(x_pc, num_visible_pc, L_pc)

        return (x_rgb_vis, x_depth_vis, x_pc_vis,
                mask_rgb, mask_depth, mask_pc,
                ids_restore_rgb, ids_restore_depth, ids_restore_pc)
    
    def forward_encoder(self, x_rgb, x_depth, x_pc, mask_ratio=0.75):
        """
        Forward pass through encoder
        """
        # Embed patches
        x_rgb = self.rgb_embed(x_rgb)  # (B, L, D)
        x_depth = self.depth_embed(x_depth)  # (B, L, D)
        x_pc = self.pc_embed(x_pc)  # (B, M, D)
        
        # Add positional embeddings
        x_rgb = x_rgb + self.pos_embed_2d
        x_depth = x_depth + self.pos_embed_2d
        x_pc = x_pc + self.pos_embed_pc
        
        # Add modality embeddings
        x_rgb = x_rgb + self.modality_embed_rgb
        x_depth = x_depth + self.modality_embed_depth
        x_pc = x_pc + self.modality_embed_pc
        
        # Masking using Dirichlet distribution
        (x_rgb_vis, x_depth_vis, x_pc_vis,
         mask_rgb, mask_depth, mask_pc,
         ids_restore_rgb, ids_restore_depth, ids_restore_pc) = self.random_masking_dirichlet(
            x_rgb, x_depth, x_pc, mask_ratio
        )
        
        # Concatenate visible tokens from all modalities
        x = torch.cat([x_rgb_vis, x_depth_vis, x_pc_vis], dim=1)
        
        # Add cls token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Apply encoder blocks
        for block in self.encoder_blocks:
            x = block(x)
        x = self.encoder_norm(x)
        
        return (x, mask_rgb, mask_depth, mask_pc,
                ids_restore_rgb, ids_restore_depth, ids_restore_pc,
                x_rgb_vis.shape[1], x_depth_vis.shape[1], x_pc_vis.shape[1])
    
    def forward_decoder(self, x, ids_restore_rgb, ids_restore_depth, ids_restore_pc,
                       len_rgb, len_depth, len_pc):
        """
        Forward pass through decoder
        """
        # Embed tokens
        x = self.decoder_embed(x)
        
        # Remove cls token
        x_no_cls = x[:, 1:, :]
        
        # Split back into modalities
        x_rgb_vis = x_no_cls[:, :len_rgb]
        x_depth_vis = x_no_cls[:, len_rgb:len_rgb+len_depth]
        x_pc_vis = x_no_cls[:, len_rgb+len_depth:]
        
        # Add mask tokens for each modality
        B = x.shape[0]
        
        # RGB
        mask_tokens_rgb = self.mask_token.repeat(B, self.num_patches - len_rgb, 1)
        x_rgb_full = torch.cat([x_rgb_vis, mask_tokens_rgb], dim=1)
        x_rgb_full = torch.gather(x_rgb_full, dim=1, 
                                  index=ids_restore_rgb.unsqueeze(-1).repeat(1, 1, x_rgb_full.shape[2]))
        
        # Depth
        mask_tokens_depth = self.mask_token.repeat(B, self.num_patches - len_depth, 1)
        x_depth_full = torch.cat([x_depth_vis, mask_tokens_depth], dim=1)
        x_depth_full = torch.gather(x_depth_full, dim=1,
                                    index=ids_restore_depth.unsqueeze(-1).repeat(1, 1, x_depth_full.shape[2]))
        
        # Point cloud
        mask_tokens_pc = self.mask_token.repeat(B, self.num_pc_tokens - len_pc, 1)
        x_pc_full = torch.cat([x_pc_vis, mask_tokens_pc], dim=1)
        x_pc_full = torch.gather(x_pc_full, dim=1,
                                index=ids_restore_pc.unsqueeze(-1).repeat(1, 1, x_pc_full.shape[2]))
        
        # Add positional embeddings
        x_rgb_full = x_rgb_full + self.decoder_pos_embed_2d
        x_depth_full = x_depth_full + self.decoder_pos_embed_2d
        x_pc_full = x_pc_full + self.decoder_pos_embed_pc
        
        # Concatenate all tokens
        x = torch.cat([x_rgb_full, x_depth_full, x_pc_full], dim=1)
        
        # Apply decoder blocks
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)
        
        # Split and predict for each modality
        x_rgb_dec = x[:, :self.num_patches]
        x_depth_dec = x[:, self.num_patches:2*self.num_patches]
        x_pc_dec = x[:, 2*self.num_patches:]
        
        pred_rgb = self.decoder_pred_rgb(x_rgb_dec)
        pred_depth = self.decoder_pred_depth(x_depth_dec)
        
        # Point cloud reconstruction with upsampling
        # x_pc_dec: (B, num_pc_tokens, decoder_embed_dim) - e.g., (B, 196, 512)
        # We need to upsample to (B, 2048, 3)
        
        B, N_tokens, _ = x_pc_dec.shape
        
        # Project to feature space
        pc_features = self.decoder_pc_proj(x_pc_dec)  # (B, N_tokens, 512)
        
        # Generate 2D grid for folding operation
        # Each token will generate points_per_token points
        # Create a grid that can handle non-square numbers
        grid_size = int(np.ceil(np.sqrt(self.points_per_token)))  # Round up to ensure enough points
        grid_1d = torch.linspace(-1, 1, grid_size, device=x_pc_dec.device)
        grid_y, grid_x = torch.meshgrid(grid_1d, grid_1d, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1)  # (grid_size, grid_size, 2)
        grid = grid.reshape(-1, 2)[:self.points_per_token]  # Take exactly points_per_token points, (points_per_token, 2)
        grid = grid.unsqueeze(0).unsqueeze(0)  # (1, 1, points_per_token, 2)
        grid = grid.expand(B, N_tokens, -1, -1)  # (B, N_tokens, points_per_token, 2)
        
        # Expand features to match grid
        pc_features_expanded = pc_features.unsqueeze(2).expand(-1, -1, self.points_per_token, -1)  # (B, N_tokens, points_per_token, 512)
        
        # Concatenate features with grid coordinates
        folding_input = torch.cat([pc_features_expanded, grid], dim=-1)  # (B, N_tokens, points_per_token, 514)
        
        # Apply folding network to generate 3D points
        folding_input_flat = folding_input.reshape(B * N_tokens * self.points_per_token, -1)  # (B*N_tokens*points_per_token, 514)
        pred_pc_flat = self.decoder_pc_fold(folding_input_flat)  # (B*N_tokens*points_per_token, 3)
        pred_pc = pred_pc_flat.reshape(B, N_tokens * self.points_per_token, 3)  # (B, 2048, 3)
        
        # Trim to exact target_points if needed (due to rounding)
        if pred_pc.shape[1] > self.target_points:
            pred_pc = pred_pc[:, :self.target_points, :]
        elif pred_pc.shape[1] < self.target_points:
            padding = self.target_points - pred_pc.shape[1]
            pred_pc = torch.cat([pred_pc, pred_pc[:, :padding, :]], dim=1)
        
        return pred_rgb, pred_depth, pred_pc
    
    def forward_loss(self, imgs_rgb, imgs_depth, pc, pred_rgb, pred_depth, pred_pc,
                    mask_rgb, mask_depth, mask_pc):
        """
        Compute reconstruction loss
        """
        # RGB loss
        target_rgb = self.patchify(imgs_rgb, self.patch_size, imgs_rgb.shape[1])
        if self.norm_pix_loss:
            mean = target_rgb.mean(dim=-1, keepdim=True)
            var = target_rgb.var(dim=-1, keepdim=True)
            target_rgb = (target_rgb - mean) / (var + 1.e-6)**.5
        
        loss_rgb = (pred_rgb - target_rgb) ** 2
        loss_rgb = loss_rgb.mean(dim=-1)
        loss_rgb = (loss_rgb * mask_rgb).sum() / mask_rgb.sum()
        
        # Depth loss — normalize TARGET only; model learns to predict normalized values
        target_depth = self.patchify(imgs_depth, self.patch_size, imgs_depth.shape[1])

        if self.normalize_depth_global:
            B_d = target_depth.shape[0]
            N_d = target_depth.shape[1]
            target_flat = target_depth.reshape(B_d, -1)

            if self.depth_norm_type == 'minmax':
                t_min = target_flat.min(dim=1, keepdim=True).values
                t_max = target_flat.max(dim=1, keepdim=True).values
                denom = (t_max - t_min).clamp(min=1e-6)
                target_flat = (target_flat - t_min) / denom
            elif self.depth_norm_type == 'standard':
                t_mean = target_flat.mean(dim=1, keepdim=True)
                t_std = target_flat.std(dim=1, keepdim=True).clamp(min=1e-6)
                target_flat = (target_flat - t_mean) / t_std

            target_depth = target_flat.reshape(B_d, N_d, -1)

        elif self.norm_pix_loss:
            mean = target_depth.mean(dim=-1, keepdim=True)
            var = target_depth.var(dim=-1, keepdim=True)
            target_depth = (target_depth - mean) / (var + 1.e-6)**.5
        
        loss_depth = (pred_depth - target_depth) ** 2
        loss_depth = loss_depth.mean(dim=-1)
        loss_depth = (loss_depth * mask_depth).sum() / mask_depth.sum()
        
        # Point cloud loss - batched Chamfer Distance
        loss_pc = chamfer_distance(pred_pc, pc)
        
        # Apply weight to point cloud loss to balance with RGB/Depth
        # Chamfer distance is naturally much smaller, so we scale it up
        loss_pc_weighted = loss_pc * self.pc_loss_weight
        
        total_loss = loss_rgb + loss_depth + loss_pc_weighted
        
        return total_loss, loss_rgb, loss_depth, loss_pc
    
    def patchify(self, imgs, patch_size, in_chans):
        """
        imgs: (B, C, H, W)
        return: (B, L, patch_size**2 * C)
        """
        p = patch_size
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], in_chans, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p**2 * in_chans)
        return x
    
    def forward(self, imgs_rgb, imgs_depth, pc, mask_ratio=0.75):
        """
        Forward pass
        
        Args:
            imgs_rgb: (B, 3, H, W) RGB images
            imgs_depth: (B, 1, H, W) Depth images
            pc: (B, N, 3) Point clouds
            mask_ratio: Masking ratio
            
        Returns:
            loss, pred, mask
        """
        # Encoder
        (latent, mask_rgb, mask_depth, mask_pc,
         ids_restore_rgb, ids_restore_depth, ids_restore_pc,
         len_rgb, len_depth, len_pc) = self.forward_encoder(
            imgs_rgb, imgs_depth, pc, mask_ratio
        )
        
        # Decoder
        pred_rgb, pred_depth, pred_pc = self.forward_decoder(
            latent, ids_restore_rgb, ids_restore_depth, ids_restore_pc,
            len_rgb, len_depth, len_pc
        )
        
        # Loss
        loss, loss_rgb, loss_depth, loss_pc = self.forward_loss(
            imgs_rgb, imgs_depth, pc, pred_rgb, pred_depth, pred_pc,
            mask_rgb, mask_depth, mask_pc
        )
        
        return loss, (loss_rgb, loss_depth, loss_pc), (pred_rgb, pred_depth, pred_pc), (mask_rgb, mask_depth, mask_pc)


def embodied_mae_small(**kwargs):
    model = EmbodiedMAE(
        embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=192, decoder_depth=4, decoder_num_heads=3,
        mlp_ratio=4, **kwargs
    )
    return model


def embodied_mae_base(**kwargs):
    model = EmbodiedMAE(
        embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, **kwargs
    )
    return model


def embodied_mae_large(**kwargs):
    model = EmbodiedMAE(
        embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, **kwargs
    )
    return model


def embodied_mae_giant(**kwargs):
    model = EmbodiedMAE(
        embed_dim=1408, depth=40, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, **kwargs
    )
    return model


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = embodied_mae_base()
    model = model.to(device)
    
    # Dummy data
    rgb = torch.randn(2, 3, 224, 224).to(device)
    depth = torch.randn(2, 1, 224, 224).to(device)
    pc = torch.randn(2, 2048, 3).to(device)
    
    # Forward pass
    loss, (loss_rgb, loss_depth, loss_pc), preds, masks = model(rgb, depth, pc)
    
    print(f"Total Loss: {loss.item():.4f}")
    print(f"RGB Loss: {loss_rgb.item():.4f}")
    print(f"Depth Loss: {loss_depth.item():.4f}")
    print(f"PC Loss: {loss_pc.item():.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")