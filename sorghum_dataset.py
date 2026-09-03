"""
Custom Dataset Loader for Sorghum Data Structure
Loads data from separate train and val folders
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from pathlib import Path
import open3d as o3d

import hashlib
import json
import os
import tempfile


# ── Dataset index cache ───────────────────────────────────────────────────────
# Scanning the sample folders is expensive on the shared filesystem: every folder
# costs an iterdir + a glob, which is ~311 s for the 22.5k-folder val split and
# ~24 min for the 105k train split. Every DDP rank pays it at every job start and
# every preemption-requeue. We cache the resolved file names instead, which also
# removes the per-__getitem__ glob in find_pointcloud_file / the 4M spline lookup.
#
# Validity is keyed on the split directory's mtime, which changes when sample
# folders are added or removed. It does NOT change when files inside an existing
# sample folder change — set SORGHUM_INDEX_REBUILD=1 to force a rescan in that case.
INDEX_CACHE_VERSION = 1


def _index_cache_dir():
    return Path(os.environ.get(
        'SORGHUM_INDEX_CACHE',
        Path(__file__).resolve().parent / '.dataset_index_cache'))


def _index_cache_path(load_dir, tag):
    key = f"{Path(load_dir).resolve()}|{tag}|v{INDEX_CACHE_VERSION}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return _index_cache_dir() / f"{Path(load_dir).name}_{tag}_{digest}.json"


def _read_index_cache(load_dir, tag):
    if os.environ.get('SORGHUM_INDEX_REBUILD'):
        return None
    path = _index_cache_path(load_dir, tag)
    try:
        with open(path) as fh:
            blob = json.load(fh)
    except (OSError, ValueError):
        return None
    try:
        if blob.get('version') != INDEX_CACHE_VERSION:
            return None
        if blob.get('dir_mtime_ns') != Path(load_dir).stat().st_mtime_ns:
            return None
    except OSError:
        return None
    return blob.get('entries')


def _write_index_cache(load_dir, tag, entries):
    """Atomic write so concurrent DDP ranks cannot observe a partial file."""
    path = _index_cache_path(load_dir, tag)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = {'version': INDEX_CACHE_VERSION,
                'dir_mtime_ns': Path(load_dir).stat().st_mtime_ns,
                'load_dir': str(Path(load_dir).resolve()),
                'entries': entries}
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix='.tmp')
        try:
            with os.fdopen(fd, 'w') as fh:
                json.dump(blob, fh)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except OSError:
        pass          # a read-only cache dir must never be fatal


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
        
        # Get all sample folders and verify they have required files.
        # `entries` is [[folder_name, pc_file_name], ...] and is cached to disk —
        # see the index-cache helpers at the top of this module.
        entries = _read_index_cache(self.load_dir, 'base')
        if entries is None:
            entries = []
            for folder in sorted(self.load_dir.iterdir()):
                if not folder.is_dir():
                    continue

                # Check for required files
                rgb_path = folder / 'rgb.png'
                depth_path = folder / 'depth.png'

                # Find point cloud file ending with _nc.ply
                pc_files = list(folder.glob('*_nc_cam.ply'))

                if rgb_path.exists() and depth_path.exists() and len(pc_files) > 0:
                    entries.append([folder.name, pc_files[0].name])
                else:
                    print(f"⚠️  Skipping {folder.name}: missing files (RGB={rgb_path.exists()}, Depth={depth_path.exists()}, PC={len(pc_files)>0})")
            _write_index_cache(self.load_dir, 'base', entries)
        else:
            print(f"⚡ index cache hit ({len(entries)} samples) — skipped the folder scan")

        self.samples = [self.load_dir / name for name, _ in entries]
        # Resolved point-cloud names, so find_pointcloud_file never globs per item.
        self._pc_names = {name: pc for name, pc in entries}
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {self.load_dir}!\n"
                           f"Expected structure: folder/*_nc.ply, folder/rgb.png, folder/depth.png")
        
        print(f"✅ Loaded {len(self.samples)} samples from {self.load_dir.name}")
    
    def __len__(self):
        return len(self.samples)
    
    def find_pointcloud_file(self, folder):
        """Resolve the *_nc_cam.ply for a sample folder.

        Uses the cached name when the index supplied one (the common path, and
        the reason __getitem__ no longer does a directory listing per sample);
        falls back to globbing for callers holding a folder we did not index.
        """
        cached = getattr(self, '_pc_names', {}).get(Path(folder).name)
        if cached is not None:
            return Path(folder) / cached
        pc_files = list(Path(folder).glob('*_nc_cam.ply'))
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
