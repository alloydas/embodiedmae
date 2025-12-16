"""
Test Depth Background Masking
Verify that depth maps are displayed without background
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
from pathlib import Path


def visualize_depth_with_without_bg(depth_path, save_path='depth_comparison.png'):
    """
    Compare depth visualization with and without background masking
    """
    # Load depth image
    depth_img = Image.open(depth_path)
    print(depth_img)
    depth_array = np.array(depth_img) / 255.0  # Normalize to 0-1
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original depth (with background)
    ax1 = axes[0]
    im1 = ax1.imshow(depth_img)
    ax1.set_title('Original Depth\n(WITH background)')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    # Masked depth (background transparent)
    ax2 = axes[1]
    depth_masked = np.ma.masked_where(depth_array < 0.95, depth_array)
    im2 = ax2.imshow(depth_masked, cmap='viridis_r')
    ax2.set_title('Masked Depth\n(NO background - threshold < 0.01)')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    # Statistics
    ax3 = axes[2]
    ax3.axis('off')
    
    bg_pixels = np.sum(depth_array < 0.01)
    fg_pixels = np.sum(depth_array >= 0.01)
    total_pixels = depth_array.size
    
    stats_text = f"""
Depth Map Statistics:

Total pixels: {total_pixels:,}
Background pixels: {bg_pixels:,} ({bg_pixels/total_pixels*100:.1f}%)
Foreground pixels: {fg_pixels:,} ({fg_pixels/total_pixels*100:.1f}%)

Value Range:
  Min: {depth_array.min():.4f}
  Max: {depth_array.max():.4f}
  Mean (fg only): {depth_array[depth_array >= 0.01].mean():.4f}

Background Detection:
  Threshold: < 0.01
  Color: Masked (transparent)
    """
    
    ax3.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center')
    
    plt.suptitle(f'Depth Background Masking Test\n{Path(depth_path).name}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved comparison to: {save_path}")
    plt.close()
    
    return bg_pixels, fg_pixels


def test_sample_folder(folder_path):
    """
    Test depth masking for a sample folder
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"❌ Folder not found: {folder_path}")
        return
    
    depth_path = folder / 'depth_no_bg.png'
    
    if not depth_path.exists():
        print(f"❌ depth_no_bg.png not found in {folder_path}")
        return
    
    print(f"Testing depth masking for: {folder.name}")
    print("=" * 60)
    
    bg_pixels, fg_pixels = visualize_depth_with_without_bg(
        depth_path, 
        save_path=f'depth_test_{folder.name}.png'
    )
    
    print(f"\n📊 Results:")
    print(f"   Background pixels: {bg_pixels:,}")
    print(f"   Foreground pixels: {fg_pixels:,}")
    print(f"   Background ratio: {bg_pixels/(bg_pixels+fg_pixels)*100:.1f}%")
    print("\n✅ Background will be hidden in training visualizations!")


def main():
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        print("Usage: python test_depth_masking.py /path/to/Sorghum_X_X")
        print("\nOr provide path when prompted:")
        folder_path = input("Enter sample folder path: ").strip()
    
    test_sample_folder(folder_path)


if __name__ == '__main__':
    main()
