"""
Custom Dataset Loader for Sorghum Data Structure
Loads data from separate train and val folders
"""

import hashlib
import json
import os

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
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
            split: Optional split name (for example 'train', 'val', or 'test').
                If provided, load from data_root/split/.
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

        # Training keeps random point sampling as augmentation.  Evaluation
        # splits use a per-file stable seed so repeated validation/test passes
        # see the same target cloud.  Infer the split from the directory name
        # as well, preserving the documented ``data_root=/path/to/val`` usage.
        sampling_split = str(split if split is not None else self.load_dir.name).lower()
        self._deterministic_point_sampling = sampling_split in {
            'val', 'validation', 'test', 'testing'
        }
        
        print(f"Loading data from: {self.load_dir}", flush=True)
        
        # Image transformations
        self.rgb_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Get all sample folders and verify they have required files
        self.samples = self.cached_index('3m', self._scan_samples)

        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {self.load_dir}!\n"
                           f"Expected structure: folder/*_nc.ply, folder/rgb.png, folder/depth.png")

        print(f"✅ Loaded {len(self.samples)} samples from {self.load_dir.name}", flush=True)

    def cached_index(self, tag, build):
        """Folder list for this split, cached on disk.

        The scan globs three patterns in every sample directory; over 105 000
        directories on a busy shared filesystem that has taken hours, and it is
        repeated by every run and every evaluation. The dataset is static, so
        the resulting names are cached next to the code. Set
        SORGHUM_INDEX_REFRESH=1 to force a rescan.
        """
        key = hashlib.sha256(f'{tag}|{self.load_dir}'.encode()).hexdigest()[:16]
        cache = Path(__file__).resolve().parent / '.index_cache' / f'{key}.json'
        if os.environ.get('SORGHUM_INDEX_REFRESH') != '1' and cache.exists():
            try:
                names = json.loads(cache.read_text())['names']
                print(f"📇 index cache hit: {len(names)} samples ({cache.name})", flush=True)
                return [self.load_dir / name for name in names]
            except Exception as exc:
                print(f"⚠️  index cache unreadable ({exc}); rescanning", flush=True)
        folders = build()
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            # Write-then-rename: two array tasks may build the same index at
            # once, and neither must ever read the other's half-written file.
            tmp = cache.with_suffix(f'.{os.getpid()}.tmp')
            tmp.write_text(json.dumps({'root': str(self.load_dir),
                                       'names': [f.name for f in folders]}))
            os.replace(tmp, cache)
            print(f"📇 index cached: {len(folders)} samples -> {cache.name}", flush=True)
        except OSError as exc:
            print(f"⚠️  could not write index cache: {exc}", flush=True)
        return folders

    def _scan_samples(self):
        found = []
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
                found.append(folder)
            else:
                print(f"⚠️  Skipping {folder.name}: missing files (RGB={rgb_path.exists()}, Depth={depth_path.exists()}, PC={len(pc_files)>0})")
        return found

    
    def __len__(self):
        return len(self.samples)
    
    def find_pointcloud_file(self, folder):
        """Find the point cloud file ending with _nc.ply"""
        pc_files = list(folder.glob('*_nc_cam.ply'))
        if len(pc_files) == 0:
            raise FileNotFoundError(f"No *_nc.ply file found in {folder}")
        return pc_files[0]  # Return the first one if multiple exist

    def _pointcloud_rng(self, ply_path):
        """Return the sampling RNG for a point-cloud file.

        Python's built-in ``hash`` is process-randomised, so evaluation uses a
        SHA-256-derived seed.  Including both the sample directory and file
        name distinguishes camera-view folders whose PLY basenames are equal.
        A fresh generator makes every fetch of a val/test file deterministic.
        """
        # Keep subclasses that predate this attribute working; evaluation
        # subclasses should still opt in explicitly to deterministic sampling.
        if not getattr(self, '_deterministic_point_sampling', False):
            return np.random

        path = Path(ply_path)
        sample_key = f'{path.parent.name}/{path.name}'.encode('utf-8')
        seed = int.from_bytes(hashlib.sha256(sample_key).digest()[:8], 'little')
        return np.random.default_rng(seed)
    
    def load_pointcloud(self, ply_path):
        """Load point cloud from PLY file"""
        return self.load_pointcloud_with_radius(ply_path)[0]

    def load_pointcloud_with_radius(self, ply_path):
        """Like load_pointcloud, plus the radius it divided by.

        The normalisation removes absolute size from the target; the 4M
        geometry-conditioning token hands this radius back to the model so
        size-type params (stem_length, leaf length) stay interpretable.
        """
        try:
            pcd = o3d.io.read_point_cloud(str(ply_path))
            points = np.asarray(pcd.points)
            
            if len(points) == 0:
                raise ValueError(f"Empty point cloud in {ply_path}")

            # Establish a canonical coordinate frame from the complete cloud.
            # Computing these statistics after a random crop makes the target's
            # translation and scale change on every fetch.
            centroid = np.mean(points, axis=0)
            points = points - centroid
            max_dist = float(np.max(np.linalg.norm(points, axis=1)))
            if max_dist > 0:
                points = points / max_dist

            # Sample or pad to fixed size
            num_points = points.shape[0]
            rng = self._pointcloud_rng(ply_path)
            if num_points >= self.num_points:
                indices = rng.choice(num_points, self.num_points, replace=False)
                points = points[indices]
            else:
                # Pad with duplicated points
                indices = rng.choice(
                    num_points, self.num_points - num_points, replace=True
                )
                padding = points[indices]
                points = np.vstack([points, padding])

            return points.astype(np.float32), max_dist
        except Exception as e:
            raise RuntimeError(f"Error loading point cloud from {ply_path}: {e}")

    def load_depth(self, depth_path):
        """Load depth without collapsing packed RGBA values to luminance.

        The active dataset stores one scalar depth value across four bytes in
        big-endian RGBA order.  Grayscale 8/16-bit depth PNGs are also accepted.
        In either case the returned tensor is float32 in [0, 1], shaped
        (1, img_size, img_size).  Zero remains the background value.
        """
        with Image.open(depth_path) as image:
            depth_array = np.asarray(image)

        if depth_array.ndim == 3:
            if depth_array.shape[2] != 4 or depth_array.dtype != np.uint8:
                raise ValueError(
                    f"Unsupported multi-channel depth image: shape={depth_array.shape}, "
                    f"dtype={depth_array.dtype}"
                )

            # depth.png uses big-endian packed RGBA:
            # scalar = R*256^3 + G*256^2 + B*256 + A.
            rgba = depth_array.astype(np.float32)
            depth_array = (
                rgba[..., 0] * (256.0 ** 3)
                + rgba[..., 1] * (256.0 ** 2)
                + rgba[..., 2] * 256.0
                + rgba[..., 3]
            ) / float((256 ** 4) - 1)
        elif depth_array.ndim == 2:
            if np.issubdtype(depth_array.dtype, np.integer):
                depth_array = depth_array.astype(np.float32) / np.iinfo(depth_array.dtype).max
            else:
                depth_array = depth_array.astype(np.float32)
                if not np.isfinite(depth_array).all():
                    raise ValueError(f"Depth image contains non-finite values: {depth_path}")
        else:
            raise ValueError(f"Unsupported depth image shape: {depth_array.shape}")

        depth = torch.from_numpy(np.ascontiguousarray(depth_array)).unsqueeze(0)
        depth = TF.resize(
            depth,
            [self.img_size, self.img_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        return depth
    
    def __getitem__(self, idx):
        item = self.load_item(idx)
        return item['rgb'], item['depth'], item['pc'], item['name']

    def load_item(self, idx):
        """All per-sample data as a dict; subclasses add fields to it."""
        sample_dir = self.samples[idx]
        
        try:
            # Load RGB
            rgb_path = sample_dir / 'rgb.png'
            rgb = Image.open(rgb_path).convert('RGB')
            rgb = self.rgb_transform(rgb)
            
            # Load Depth
            depth_path = sample_dir / 'depth.png'
            depth = self.load_depth(depth_path)
            
            # Find and load Point Cloud
            pc_path = self.find_pointcloud_file(sample_dir)
            pc, pc_radius = self.load_pointcloud_with_radius(pc_path)
            pc = torch.from_numpy(pc)
            
            return {'rgb': rgb, 'depth': depth, 'pc': pc,
                    'name': str(sample_dir.name), 'pc_radius': pc_radius,
                    'dir': sample_dir}
        
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
