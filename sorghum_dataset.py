"""
Custom Dataset Loader for Sorghum Data Structure
Loads data from separate train and val folders
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
    Dataset for Sorghum data structure with separate train/val folders
    
    Expected structure:
    Option 1:
    data_root/
        train/
            Sorghum_1_1/
                Sorghum_1_nc.ply
                rgb.png
                depth_no_bg.png
            Sorghum_2_2/
                ...
        val/
            Sorghum_9_1/
                Sorghum_9_nc.ply
                rgb.png
                depth_no_bg.png
            ...
    
    Option 2:
    train/
        Sorghum_1_1/
            ...
    val/
        Sorghum_9_1/
            ...
    
    Usage:
        # For training
        train_dataset = SorghumDataset(data_root='./data/train')
        # OR if using Option 1
        train_dataset = SorghumDataset(data_root='./data', split='train')
        
        # For validation
        val_dataset = SorghumDataset(data_root='./data/val')
        # OR if using Option 1
        val_dataset = SorghumDataset(data_root='./data', split='val')
    """
    def __init__(self, data_root, img_size=224, num_points=10000, split=None):
        """
        Args:
            data_root: Path to data directory
            img_size: Image size for resizing
            num_points: Number of points in point cloud
            split: Optional split name ('train' or 'val'). If provided, will look for data_root/split/
        """
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.num_points = num_points
        
        # Determine the folder to load from
        if split is not None:
            # Option 1: data_root/train/ or data_root/val/
            self.load_dir = self.data_root / split
            if not self.load_dir.exists():
                raise ValueError(f"Split directory not found: {self.load_dir}")
        else:
            # Option 2: data_root is already train/ or val/
            self.load_dir = self.data_root
        
        print(f"Loading data from: {self.load_dir}")
        
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
        self.samples = []
        for folder in sorted(self.load_dir.iterdir()):
            if not folder.is_dir():
                continue
            
            # Check for required files
            rgb_path = folder / 'rgb.png'
            depth_path = folder / 'depth.png'
            
            # Find point cloud file ending with _nc.ply
            pc_files = list(folder.glob('*_nc_cam.ply'))
            #print(pc_files)
            
            if rgb_path.exists() and depth_path.exists() and len(pc_files) > 0:
                self.samples.append(folder)
            else:
                print(f"⚠️  Skipping {folder.name}: missing files (RGB={rgb_path.exists()}, Depth={depth_path.exists()}, PC={len(pc_files)>0})")
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {self.load_dir}!\n"
                           f"Expected structure: folder/*_nc.ply, folder/rgb.png, folder/depth.png")
        
        print(f"✅ Loaded {len(self.samples)} samples from {self.load_dir.name}")
    
    def __len__(self):
        return len(self.samples)
    
    def find_pointcloud_file(self, folder):
        """Find the point cloud file ending with _nc.ply"""
        pc_files = list(folder.glob('*_nc_cam.ply'))
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
            depth_path = sample_dir / 'depth.png'
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root', help='Path containing train/ and val/ subdirectories')
    args = parser.parse_args()

    for split in ('train', 'val'):
        ds = SorghumDataset(args.data_root, split=split)
        print(f"{split}: {len(ds)} samples")
        if len(ds) > 0:
            rgb, depth, pc, name = ds[0]
            print(f"  sample={name}  rgb={tuple(rgb.shape)}  depth={tuple(depth.shape)}  pc={tuple(pc.shape)}")
