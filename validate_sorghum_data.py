"""
Validate your Sorghum dataset before training
Quick check to ensure data loads correctly
"""
import numpy as np
import torch
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sorghum_dataset import SorghumDataset



def visualize_sample(rgb, depth, pc, name, save_path='test_sample.png'):
    """
    Visualize a single sample
    """
    fig = plt.figure(figsize=(15, 5))
    
    # Denormalize RGB
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    rgb_denorm = rgb * std + mean
    
    # RGB
    ax1 = fig.add_subplot(131)
    ax1.imshow(rgb_denorm.permute(1, 2, 0).clamp(0, 1))
    ax1.set_title(f'RGB Image\n{name}')
    ax1.axis('off')
    
    # Depth
    ax2 = fig.add_subplot(132)
    # depth_data = depth[0].numpy()
    # depth_masked = np.ma.masked_where(depth_data > 0.4, depth_data)
    ax2.imshow(depth[0], cmap='viridis_r')
    ax2.set_title('Depth Image')
    ax2.axis('off')
    
    # Point Cloud
    ax3 = fig.add_subplot(133, projection='3d')
    pc_np = pc.numpy()
    
    # Subsample for visualization
    if pc_np.shape[0] > 2000:
        # import numpy as np
        indices = np.random.choice(pc_np.shape[0], 2000, replace=False)
        pc_vis = pc_np[indices]
    else:
        pc_vis = pc_np
    
    ax3.scatter(pc_vis[:, 0], pc_vis[:, 1], pc_vis[:, 2], 
               c=pc_vis[:, 2], cmap='viridis', s=1)
    ax3.set_title('Point Cloud')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualization saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Validate Sorghum Dataset')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to data directory')
    parser.add_argument('--num_samples', type=int, default=3,
                       help='Number of samples to visualize')
    args = parser.parse_args()
    
    print("=" * 80)
    print("Sorghum Dataset Validation")
    print("=" * 80)
    
    # Check if data directory exists
    data_path = Path(args.data_root)
    if not data_path.exists():
        print(f"❌ Error: Data directory not found: {args.data_root}")
        return
    
    print(f"\n📁 Data directory: {args.data_root}")
    
    # Get all sample folders
    sample_folders = sorted([d for d in data_path.iterdir() if d.is_dir()])
    print(f"📊 Found {len(sample_folders)} sample folders")
    
    if len(sample_folders) == 0:
        print("❌ No sample folders found!")
        print("\nExpected structure:")
        print("  data_root/")
        print("    Sorghum_1_1/")
        print("      pointcloud.ply")
        print("      rgb.png")
        print("      depth.png")
        return
    
    # Check first few folders for correct structure
    print("\n🔍 Checking folder structure...")
    valid_samples = 0
    for folder in sample_folders[:5]:
        has_rgb = (folder / 'rgb.png').exists()
        has_depth = (folder / 'depth_no_bg.png').exists()
        
        # Check for files ending with _nc.ply
        pc_files = list(folder.glob('*_nc.ply'))
        has_pc = len(pc_files) > 0
        pc_filename = pc_files[0].name if has_pc else "NOT FOUND"
        
        status = "✅" if (has_rgb and has_depth and has_pc) else "❌"
        print(f"  {status} {folder.name}: RGB={has_rgb}, Depth={has_depth}, PC={pc_filename}")
        
        if has_rgb and has_depth and has_pc:
            valid_samples += 1
    
    if valid_samples == 0:
        print("\n❌ No valid samples found!")
        print("Each folder should contain: *_nc.ply, rgb.png, depth_no_bg.png")
        print("Example: Sorghum_9_nc.ply, rgb.png, depth_no_bg.png")
        return
    
    print(f"\n✅ Dataset structure looks good!")
    
    # Try loading dataset
    print("\n📦 Loading dataset...")
    try:
        dataset = SorghumDataset(args.data_root, train=True)
        print(f"✅ Successfully loaded {len(dataset)} training samples")
        
        val_dataset = SorghumDataset(args.data_root, train=False)
        print(f"✅ Successfully loaded {len(val_dataset)} validation samples")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Load and visualize samples
    print(f"\n🎨 Visualizing {args.num_samples} samples...")
    num_to_viz = min(args.num_samples, len(dataset))
    
    for i in range(num_to_viz):
        try:
            rgb, depth, pc, name = dataset[i]
            
            print(f"\n  Sample {i+1}: {name}")
            print(f"    RGB shape: {rgb.shape}")
            print(f"    Depth shape: {depth.shape}")
            print(f"    Point cloud shape: {pc.shape}")
            print(f"    RGB range: [{rgb.min():.3f}, {rgb.max():.3f}]")
            print(f"    Depth range: [{depth.min():.3f}, {depth.max():.3f}]")
            print(f"    PC range: X=[{pc[:, 0].min():.3f}, {pc[:, 0].max():.3f}], "
                  f"Y=[{pc[:, 1].min():.3f}, {pc[:, 1].max():.3f}], "
                  f"Z=[{pc[:, 2].min():.3f}, {pc[:, 2].max():.3f}]")
            
            # Visualize
            save_path = f'validation_sample_{i+1}_{name}.png'
            visualize_sample(rgb, depth, pc, name, save_path)
            
        except Exception as e:
            print(f"    ❌ Error loading sample: {e}")
    
    # Test DataLoader
    print("\n🔄 Testing DataLoader...")
    try:
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
        rgb_batch, depth_batch, pc_batch, names = next(iter(dataloader))
        
        print(f"✅ DataLoader working!")
        print(f"    Batch RGB shape: {rgb_batch.shape}")
        print(f"    Batch Depth shape: {depth_batch.shape}")
        print(f"    Batch PC shape: {pc_batch.shape}")
        print(f"    Sample names: {names}")
    except Exception as e:
        print(f"❌ Error with DataLoader: {e}")
        return
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ VALIDATION COMPLETE - Dataset is ready for training!")
    print("=" * 80)
    print("\n📝 Next steps:")
    print("  1. Check the generated visualization images")
    print("  2. Start training with:")
    print(f"     python train_sorghum.py --data_root {args.data_root} --batch_size 8 --epochs 100")
    print("\n💡 Tips:")
    print("  - Use --batch_size 4 if you have limited GPU memory")
    print("  - Use --model_size small for faster training")
    print("  - Visualizations will be saved every 10 epochs automatically")


if __name__ == '__main__':
    main()
