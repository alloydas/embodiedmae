"""
Utilities for EmbodiedMAE: Visualization and Model Distillation
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
from embodied_mae import embodied_mae_base, embodied_mae_giant


def unpatchify(x, patch_size, channels):
    """
    Reconstruct image from patches
    
    Args:
        x: (B, L, patch_size**2 * C)
        patch_size: patch size
        channels: number of channels
    
    Returns:
        imgs: (B, C, H, W)
    """
    p = patch_size
    h = w = int(x.shape[1]**.5)
    assert h * w == x.shape[1]
    
    x = x.reshape(x.shape[0], h, w, p, p, channels)
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(x.shape[0], channels, h * p, w * p)
    return imgs


@torch.no_grad()
def visualize_reconstruction(model, rgb, depth, pc, device, save_path=None):
    """
    Visualize MAE reconstruction results
    
    Args:
        model: EmbodiedMAE model
        rgb: (B, 3, H, W) RGB images
        depth: (B, 1, H, W) Depth images
        pc: (B, N, 3) Point clouds
        device: torch device
        save_path: path to save visualization
    """
    model.eval()
    
    rgb = rgb.to(device)
    depth = depth.to(device)
    pc = pc.to(device)
    
    # Forward pass
    _, _, (pred_rgb, pred_depth, pred_pc), (mask_rgb, mask_depth, mask_pc) = model(rgb, depth, pc)
    
    # Unpatchify predictions
    pred_rgb_img = unpatchify(pred_rgb, model.patch_size, 3)
    pred_depth_img = unpatchify(pred_depth, model.patch_size, 1)
    
    # Denormalize RGB if needed
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    rgb_denorm = rgb * std + mean
    
    # Create visualization for first sample in batch
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # RGB
    axes[0, 0].imshow(rgb_denorm[0].cpu().permute(1, 2, 0).clamp(0, 1))
    axes[0, 0].set_title('Original RGB')
    axes[0, 0].axis('off')
    
    # Create masked RGB
    mask_rgb_img = mask_rgb[0].reshape(int(mask_rgb.shape[1]**.5), int(mask_rgb.shape[1]**.5))
    mask_rgb_img = mask_rgb_img.unsqueeze(0).unsqueeze(0).float()
    mask_rgb_img = torch.nn.functional.interpolate(
        mask_rgb_img, size=(rgb.shape[2], rgb.shape[3]), mode='nearest'
    )
    rgb_masked = rgb_denorm[0] * (1 - mask_rgb_img[0])
    axes[0, 1].imshow(rgb_masked.cpu().permute(1, 2, 0).clamp(0, 1))
    axes[0, 1].set_title('Masked RGB')
    axes[0, 1].axis('off')
    
    pred_rgb_denorm = pred_rgb_img[0] * std[0] + mean[0]
    axes[0, 2].imshow(pred_rgb_denorm.cpu().permute(1, 2, 0).clamp(0, 1))
    axes[0, 2].set_title('Reconstructed RGB')
    axes[0, 2].axis('off')
    
    # Depth
    axes[1, 0].imshow(depth[0, 0].cpu(), cmap='viridis')
    axes[1, 0].set_title('Original Depth')
    axes[1, 0].axis('off')
    
    mask_depth_img = mask_depth[0].reshape(int(mask_depth.shape[1]**.5), int(mask_depth.shape[1]**.5))
    mask_depth_img = mask_depth_img.unsqueeze(0).unsqueeze(0).float()
    mask_depth_img = torch.nn.functional.interpolate(
        mask_depth_img, size=(depth.shape[2], depth.shape[3]), mode='nearest'
    )
    depth_masked = depth[0] * (1 - mask_depth_img[0])
    axes[1, 1].imshow(depth_masked[0].cpu(), cmap='viridis')
    axes[1, 1].set_title('Masked Depth')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(pred_depth_img[0, 0].cpu(), cmap='viridis')
    axes[1, 2].set_title('Reconstructed Depth')
    axes[1, 2].axis('off')
    
    # Point Cloud (3D scatter plot)
    from mpl_toolkits.mplot3d import Axes3D
    
    # Original PC
    ax1 = fig.add_subplot(3, 3, 7, projection='3d')
    pc_np = pc[0].cpu().numpy()
    ax1.scatter(pc_np[:, 0], pc_np[:, 1], pc_np[:, 2], c=pc_np[:, 2], cmap='viridis', s=1)
    ax1.set_title('Original Point Cloud')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Masked PC
    ax2 = fig.add_subplot(3, 3, 8, projection='3d')
    mask_pc_bool = (1 - mask_pc[0]).bool().cpu().numpy()
    pc_masked_np = pc_np[mask_pc_bool[:len(pc_np)]]
    ax2.scatter(pc_masked_np[:, 0], pc_masked_np[:, 1], pc_masked_np[:, 2], 
               c=pc_masked_np[:, 2], cmap='viridis', s=1)
    ax2.set_title('Masked Point Cloud')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # Reconstructed PC
    ax3 = fig.add_subplot(3, 3, 9, projection='3d')
    pred_pc_np = pred_pc[0].detach().cpu().numpy()
    ax3.scatter(pred_pc_np[:, 0], pred_pc_np[:, 1], pred_pc_np[:, 2],
               c=pred_pc_np[:, 2], cmap='viridis', s=1)
    ax3.set_title('Reconstructed Point Cloud')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


class ModelDistiller:
    """
    Knowledge distillation from teacher (Giant) to student models (Small/Base/Large)
    """
    def __init__(self, teacher_model, student_model, device, 
                 distill_layers=[0.25, 0.5, 0.75, 1.0], temperature=1.0):
        """
        Args:
            teacher_model: Teacher model (e.g., EmbodiedMAE-Giant)
            student_model: Student model (e.g., EmbodiedMAE-Base)
            device: torch device
            distill_layers: Relative positions of layers to distill (0-1)
            temperature: Temperature for distillation
        """
        self.teacher = teacher_model.to(device).eval()
        self.student = student_model.to(device)
        self.device = device
        self.distill_layers = distill_layers
        self.temperature = temperature
        
        # Calculate actual layer indices
        self.teacher_layer_indices = [
            int(pos * (len(self.teacher.encoder_blocks) - 1))
            for pos in distill_layers
        ]
        self.student_layer_indices = [
            int(pos * (len(self.student.encoder_blocks) - 1))
            for pos in distill_layers
        ]
        
        # Feature alignment layers (if dimensions don't match)
        self.alignment_layers = nn.ModuleList()
        for i in range(len(self.teacher_layer_indices)):
            if self.teacher.embed_dim != self.student.embed_dim:
                self.alignment_layers.append(
                    nn.Linear(self.student.embed_dim, self.teacher.embed_dim).to(device)
                )
            else:
                self.alignment_layers.append(nn.Identity().to(device))
    
    def get_intermediate_features(self, model, x, layer_indices):
        """
        Extract intermediate features from specified layers
        """
        features = []
        
        for idx, block in enumerate(model.encoder_blocks):
            x = block(x)
            if idx in layer_indices:
                features.append(x)
        
        return features
    
    @torch.no_grad()
    def get_teacher_features(self, rgb, depth, pc, mask_ratio=0.75):
        """
        Get teacher features
        """
        # Embed patches
        x_rgb = self.teacher.rgb_embed(rgb)
        x_depth = self.teacher.depth_embed(depth)
        x_pc = self.teacher.pc_embed(pc)
        
        # Add positional and modality embeddings
        x_rgb = x_rgb + self.teacher.pos_embed_2d + self.teacher.modality_embed_rgb
        x_depth = x_depth + self.teacher.pos_embed_2d + self.teacher.modality_embed_depth
        x_pc = x_pc + self.teacher.pos_embed_pc + self.teacher.modality_embed_pc
        
        # Masking
        (x_rgb_vis, x_depth_vis, x_pc_vis, _, _, _, _, _, _) = \
            self.teacher.random_masking_dirichlet(x_rgb, x_depth, x_pc, mask_ratio)
        
        # Concatenate visible tokens
        x = torch.cat([x_rgb_vis, x_depth_vis, x_pc_vis], dim=1)
        
        # Add cls token
        cls_token = self.teacher.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Get intermediate features
        features = self.get_intermediate_features(
            self.teacher, x, self.teacher_layer_indices
        )
        
        return features
    
    def compute_distillation_loss(self, student_features, teacher_features):
        """
        Compute feature-level distillation loss
        """
        loss = 0
        for i, (student_feat, teacher_feat) in enumerate(zip(student_features, teacher_features)):
            # Align student features to teacher dimension if needed
            student_feat_aligned = self.alignment_layers[i](student_feat)
            
            # MSE loss between features
            loss += F.mse_loss(student_feat_aligned, teacher_feat)
        
        return loss / len(student_features)
    
    def distill_step(self, rgb, depth, pc, mask_ratio=0.75, alpha=0.5):
        """
        One distillation step
        
        Args:
            rgb, depth, pc: Input data
            mask_ratio: Masking ratio
            alpha: Weight for distillation loss (1-alpha for reconstruction loss)
        
        Returns:
            total_loss, recon_loss, distill_loss
        """
        # Get teacher features (no gradient)
        teacher_features = self.get_teacher_features(rgb, depth, pc, mask_ratio)
        
        # Student forward pass
        # Embed patches
        x_rgb = self.student.rgb_embed(rgb)
        x_depth = self.student.depth_embed(depth)
        x_pc = self.student.pc_embed(pc)
        
        # Add positional and modality embeddings
        x_rgb = x_rgb + self.student.pos_embed_2d + self.student.modality_embed_rgb
        x_depth = x_depth + self.student.pos_embed_2d + self.student.modality_embed_depth
        x_pc = x_pc + self.student.pos_embed_pc + self.student.modality_embed_pc
        
        # Masking (use same random seed as teacher for consistency)
        (x_rgb_vis, x_depth_vis, x_pc_vis,
         mask_rgb, mask_depth, mask_pc,
         ids_restore_rgb, ids_restore_depth, ids_restore_pc) = \
            self.student.random_masking_dirichlet(x_rgb, x_depth, x_pc, mask_ratio)
        
        # Concatenate visible tokens
        x = torch.cat([x_rgb_vis, x_depth_vis, x_pc_vis], dim=1)
        
        # Add cls token
        cls_token = self.student.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Get student intermediate features
        student_features = self.get_intermediate_features(
            self.student, x, self.student_layer_indices
        )
        
        # Apply encoder norm
        x = self.student.encoder_norm(x)
        
        # Decoder
        pred_rgb, pred_depth, pred_pc = self.student.forward_decoder(
            x, ids_restore_rgb, ids_restore_depth, ids_restore_pc,
            x_rgb_vis.shape[1], x_depth_vis.shape[1], x_pc_vis.shape[1]
        )
        
        # Reconstruction loss
        recon_loss, _, _, _ = self.student.forward_loss(
            rgb, depth, pc, pred_rgb, pred_depth, pred_pc,
            mask_rgb, mask_depth, mask_pc
        )
        
        # Distillation loss
        distill_loss = self.compute_distillation_loss(student_features, teacher_features)
        
        # Total loss
        total_loss = (1 - alpha) * recon_loss + alpha * distill_loss
        
        return total_loss, recon_loss, distill_loss


def extract_features(model, rgb, depth, pc, device):
    """
    Extract features from EmbodiedMAE for downstream tasks
    
    Args:
        model: EmbodiedMAE model
        rgb, depth, pc: Input data
        device: torch device
    
    Returns:
        features: Extracted features
    """
    model.eval()
    
    rgb = rgb.to(device)
    depth = depth.to(device)
    pc = pc.to(device)
    
    with torch.no_grad():
        # Embed patches
        x_rgb = model.rgb_embed(rgb)
        x_depth = model.depth_embed(depth)
        x_pc = model.pc_embed(pc)
        
        # Add positional and modality embeddings
        x_rgb = x_rgb + model.pos_embed_2d + model.modality_embed_rgb
        x_depth = x_depth + model.pos_embed_2d + model.modality_embed_depth
        x_pc = x_pc + model.pos_embed_pc + model.modality_embed_pc
        
        # Concatenate all tokens (no masking)
        x = torch.cat([x_rgb, x_depth, x_pc], dim=1)
        
        # Add cls token
        cls_token = model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Apply encoder
        for block in model.encoder_blocks:
            x = block(x)
        x = model.encoder_norm(x)
        
        # Return cls token as global feature
        cls_feature = x[:, 0]
        
        # Or return all tokens
        all_features = x[:, 1:]
        
    return cls_feature, all_features


if __name__ == "__main__":
    # Test visualization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = embodied_mae_base()
    model = model.to(device)
    
    # Dummy data
    rgb = torch.randn(2, 3, 224, 224).to(device)
    depth = torch.randn(2, 1, 224, 224).to(device)
    pc = torch.randn(2, 2048, 3).to(device)
    
    print("Testing visualization...")
    visualize_reconstruction(model, rgb, depth, pc, device, save_path='visualization.png')
    
    print("\nTesting feature extraction...")
    cls_feat, all_feat = extract_features(model, rgb, depth, pc, device)
    print(f"CLS feature shape: {cls_feat.shape}")
    print(f"All features shape: {all_feat.shape}")
    
    print("\nTesting model distillation...")
    teacher = embodied_mae_giant()
    student = embodied_mae_base()
    distiller = ModelDistiller(teacher, student, device)
    
    total_loss, recon_loss, distill_loss = distiller.distill_step(rgb, depth, pc)
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print(f"Distillation loss: {distill_loss.item():.4f}")
