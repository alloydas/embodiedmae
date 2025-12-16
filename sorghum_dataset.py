"""
Custom Dataset Loader for Sorghum Data Structure
Each sample is in a separate folder with *_nc.ply, rgb.png, and depth.png
Automatically handles train/val split
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import open3d as o3d


class SorghumDataset(Dataset):
    """
    Dataset for Sorghum data structure
    
    Expected structure:
    data_root/
        Sorghum_1_1/
            Sorghum_1_nc.ply  (or any file ending with _nc.ply)
            rgb.png
            depth.png
        Sorghum_9_99/
            Sorghum_9_nc.ply
            rgb.png
            depth.png
        ...
    """
    def __init__(self, data_root, img_size=224, num_points=2048, train=True, train_split=0.9, seed=42):
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.num_points = num_points
        
        # Image transformations
        self.rgb_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.depth_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        
        # Get all sample folders and verify they have required files
        all_samples = []
        for folder in sorted(self.data_root.iterdir()):
            if not folder.is_dir():
                continue
            
            # Check for required files
            rgb_path = folder / 'rgb.png'
            depth_path = folder / 'depth_no_bg.png'
            
            # Find point cloud file ending with _nc.ply
            pc_files = list(folder.glob('*_nc.ply'))
            
            if rgb_path.exists() and depth_path.exists() and len(pc_files) > 0:
                all_samples.append(folder)
            else:
                print(f"⚠️  Skipping {folder.name}: missing files (RGB={rgb_path.exists()}, Depth={depth_path.exists()}, PC={len(pc_files)>0})")
        
        if len(all_samples) == 0:
            raise ValueError(f"No valid samples found in {data_root}!\n"
                           f"Expected structure: folder/*_nc.ply, folder/rgb.png, folder/depth_no_bg.png")
        
        # Shuffle with fixed seed for reproducible splits
        np.random.seed(seed)
        indices = np.random.permutation(len(all_samples))
        all_samples = [all_samples[i] for i in indices]
        
        # Split into train/val
        split_idx = int(len(all_samples) * train_split)
        if train:
            self.samples = all_samples[:split_idx]
            split_name = 'train'
        else:
            self.samples = all_samples[split_idx:]
            split_name = 'val'
        
        print(f"✅ Loaded {len(self.samples)} samples for {split_name} split (total: {len(all_samples)})")
    
    def __len__(self):
        return len(self.samples)
    
    def find_pointcloud_file(self, folder):
        """Find the point cloud file ending with _nc.ply"""
        pc_files = list(folder.glob('*_nc.ply'))
        if len(pc_files) == 0:
            raise FileNotFoundError(f"No *_nc.ply file found in {folder}")
        return pc_files[0]  # Return the first one if multiple exist
    
    def load_pointcloud(self, ply_path):
        """Load point cloud from PLY file"""
        try:
            pcd = o3d.io.read_point_cloud(str(ply_path))
            points = np.asarray(pcd.points)
            
            if len(points) == 0:
                raise ValueError(f"Empty point cloud in {ply_path}")
            
            # Sample or pad to fixed size
            num_points = points.shape[0]
            if num_points >= self.num_points:
                indices = np.random.choice(num_points, self.num_points, replace=False)
                points = points[indices]
            else:
                # Pad with duplicated points
                indices = np.random.choice(num_points, self.num_points - num_points, replace=True)
                padding = points[indices]
                points = np.vstack([points, padding])
            
            # Normalize point cloud
            centroid = np.mean(points, axis=0)
            points = points - centroid
            max_dist = np.max(np.linalg.norm(points, axis=1))
            if max_dist > 0:
                points = points / max_dist
            
            return points.astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Error loading point cloud from {ply_path}: {e}")
    
    def __getitem__(self, idx):
        sample_dir = self.samples[idx]
        
        try:
            # Load RGB
            rgb_path = sample_dir / 'rgb.png'
            rgb = Image.open(rgb_path).convert('RGB')
            rgb = self.rgb_transform(rgb)
            
            # Load Depth
            depth_path = sample_dir / 'depth_no_bg.png'
            depth = Image.open(depth_path).convert('L')
            depth = self.depth_transform(depth)
            
            # Find and load Point Cloud
            pc_path = self.find_pointcloud_file(sample_dir)
            pc = self.load_pointcloud(pc_path)
            pc = torch.from_numpy(pc)
            
            return rgb, depth, pc, str(sample_dir.name)
        
        except Exception as e:
            print(f"❌ Error loading sample {sample_dir.name}: {e}")
            # Return a dummy sample or raise
            raise


if __name__ == '__main__':
    # Test the dataset
    dataset = SorghumDataset('./data', train=True)
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        rgb, depth, pc, name = dataset[0]
        print(f"\nSample: {name}")
        print(f"RGB shape: {rgb.shape}")
        print(f"Depth shape: {depth.shape}")
        print(f"Point cloud shape: {pc.shape}")
