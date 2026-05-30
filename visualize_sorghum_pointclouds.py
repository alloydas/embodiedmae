"""
Sorghum Point Cloud Visualizer  –  cam view only
=================================================
Saves one PNG per sub-plant showing only the simulated-scan (cam) point cloud.

Output naming:
    save_dir/
      Sorghum_1/
        Sorghum_1_SubPlant_01.png
        Sorghum_1_SubPlant_02.png
        ...
      Sorghum_2/
        Sorghum_2_SubPlant_01.png
        ...

Usage:
    python visualize_sorghum_pointclouds.py --data_root /path/to/dataset --save_dir ./viz_out
    python visualize_sorghum_pointclouds.py --data_root /path/to/dataset --save_dir ./viz_out --plant Sorghum_3
    python visualize_sorghum_pointclouds.py --data_root /path/to/dataset --save_dir ./viz_out --subsample 20000
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# PLY loader
# ─────────────────────────────────────────────────────────────────────────────

def load_ply(path: Path, max_points: int | None = None) -> np.ndarray:
    """Load XYZ from ASCII or binary-little-endian PLY. Returns (N,3) float32."""
    with open(path, 'rb') as f:
        raw = f.read()

    header_end = raw.find(b'end_header')
    if header_end == -1:
        raise ValueError(f"Invalid PLY (no end_header): {path}")
    header     = raw[:header_end].decode('ascii', errors='replace')
    body_start = header_end + len('end_header\n')

    n_vertices = 0
    is_binary  = False
    dt_fields  = []

    for line in header.splitlines():
        line = line.strip()
        if line.startswith('element vertex'):
            n_vertices = int(line.split()[-1])
        elif 'binary_little_endian' in line:
            is_binary = True
        elif line.startswith('property float '):
            dt_fields.append((line.split()[-1], np.float32))
        elif line.startswith('property double '):
            dt_fields.append((line.split()[-1], np.float64))
        elif line.startswith('property uchar ') or line.startswith('property uint8 '):
            dt_fields.append((line.split()[-1], np.uint8))
        elif line.startswith('property int ') or line.startswith('property int32 '):
            dt_fields.append((line.split()[-1], np.int32))
        elif line.startswith('property uint ') or line.startswith('property uint32 '):
            dt_fields.append((line.split()[-1], np.uint32))

    if n_vertices == 0:
        return np.zeros((0, 3), dtype=np.float32)

    if is_binary:
        data = np.frombuffer(raw[body_start:], dtype=np.dtype(dt_fields), count=n_vertices)
        xyz  = np.stack([data['x'].astype(np.float32),
                         data['y'].astype(np.float32),
                         data['z'].astype(np.float32)], axis=1)
    else:
        lines = raw[body_start:].decode('ascii', errors='replace').splitlines()
        rows  = []
        for line in lines[:n_vertices]:
            vals = line.strip().split()
            if len(vals) >= 3:
                rows.append([float(vals[0]), float(vals[1]), float(vals[2])])
        xyz = np.array(rows, dtype=np.float32) if rows else np.zeros((0, 3), dtype=np.float32)

    if max_points and len(xyz) > max_points:
        idx = np.random.choice(len(xyz), max_points, replace=False)
        xyz = xyz[idx]

    return xyz


# ─────────────────────────────────────────────────────────────────────────────
# Dataset discovery
# ─────────────────────────────────────────────────────────────────────────────

PLY_RE = re.compile(r'Sorghum_.+_(nc_cam|nc|cam)\.ply$', re.IGNORECASE)

def variant_key(filename: str) -> str:
    m = PLY_RE.match(filename)
    return m.group(1).lower() if m else ''

def discover_dataset(data_root: Path) -> dict:
    """
    Returns { plant_folder_name : { sub_folder_name : { variant : Path } } }
    Handles hierarchical (Plant/Sub/*.ply), flat-per-plant (Plant/*.ply),
    and fully-flat (*.ply) layouts.
    """
    dataset = {}

    for plant_dir in sorted(data_root.iterdir()):
        if not plant_dir.is_dir():
            continue
        sub_dirs = sorted(d for d in plant_dir.iterdir() if d.is_dir())

        if sub_dirs:
            for sub_dir in sub_dirs:
                plys = {variant_key(f.name): f
                        for f in sorted(sub_dir.glob('*.ply'))
                        if PLY_RE.match(f.name)}
                if plys:
                    dataset.setdefault(plant_dir.name, {})[sub_dir.name] = plys
        else:
            plys = {variant_key(f.name): f
                    for f in sorted(plant_dir.glob('*.ply'))
                    if PLY_RE.match(f.name)}
            if plys:
                dataset.setdefault(plant_dir.name, {})['root'] = plys

    root_plys = {variant_key(f.name): f
                 for f in sorted(data_root.glob('*.ply'))
                 if PLY_RE.match(f.name)}
    if root_plys:
        dataset.setdefault('root', {})['root'] = root_plys

    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# Render  –  cam view only, single panel
# ─────────────────────────────────────────────────────────────────────────────

def render_cam(
    plant_name: str,
    sub_name:   str,
    paths:      dict,
    subsample:  int,
    out_path:   Path,
) -> None:
    """Save a single 3-D scatter of the cam PLY coloured by Z height."""

    fig = plt.figure(figsize=(7, 7), facecolor='white')
    ax  = fig.add_subplot(111, projection='3d')

    pc = np.zeros((0, 3), dtype=np.float32)
    if 'cam' in paths:
        try:
            pc = load_ply(paths['cam'], max_points=subsample)
        except Exception as exc:
            print(f'    WARNING – {paths["cam"].name}: {exc}')

    if len(pc) > 0:
        z_norm = (pc[:, 2] - pc[:, 2].min()) / (pc[:, 2].max() - pc[:, 2].min() + 1e-8)
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2],
                   c=z_norm, cmap='viridis',
                   s=1.2, alpha=0.7, depthshade=True)
        n_str = f'{len(pc):,} points'
    else:
        n_str = 'cam PLY not found'

    ax.set_title(f'Simulated Scan (cam)\n{n_str}', fontsize=10)
    ax.set_xlabel('X', fontsize=8, labelpad=2)
    ax.set_ylabel('Y', fontsize=8, labelpad=2)
    ax.set_zlabel('Z', fontsize=8, labelpad=2)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=25, azim=45)
    ax.set_facecolor('#f5f5f5')

    fig.suptitle(f'{plant_name}  /  {sub_name}', fontsize=12, fontweight='bold')
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Save cam-view point-cloud PNGs for every sub-plant folder.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--data_root', required=True,
                        help='Root: data_root/PlantFolder/SubFolder/*.ply')
    parser.add_argument('--save_dir', default='./viz_output',
                        help='Output root  (save_dir/PlantFolder/PlantFolder_SubFolder.png)')
    parser.add_argument('--plant', default=None,
                        help='Process only this plant folder name (e.g. Sorghum_3)')
    parser.add_argument('--subsample', type=int, default=8192,
                        help='Max points rendered per cloud')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    save_dir  = Path(args.save_dir)

    if not data_root.exists():
        print(f'ERROR: data_root not found: {data_root}')
        sys.exit(1)

    print(f'Scanning {data_root} ...')
    dataset = discover_dataset(data_root)

    if not dataset:
        print('No matching PLY files found.')
        sys.exit(1)

    if args.plant:
        if args.plant not in dataset:
            print(f'Plant "{args.plant}" not found. Available: {sorted(dataset.keys())}')
            sys.exit(1)
        dataset = {args.plant: dataset[args.plant]}

    total = sum(len(v) for v in dataset.values())
    print(f'Found {len(dataset)} plant(s), {total} sub-folder(s)')
    print(f'Saving to: {save_dir}\n')

    saved = 0
    for plant_name, sub_dict in sorted(dataset.items()):
        print(f'  [{plant_name}]  {len(sub_dict)} sub-folder(s)')
        for sub_name, paths in sorted(sub_dict.items()):
            # filename = PlantFolder_SubFolder.png  inside  save_dir/PlantFolder/
            filename = f'{plant_name}_{sub_name}.png'
            out_path =   filename
            render_cam(plant_name, sub_name, paths, args.subsample, out_path)
            print(f'    saved  {plant_name}/{filename}')
            saved += 1

    print(f'\nDone — {saved} figure(s) saved under {save_dir}')


if __name__ == '__main__':
    main()