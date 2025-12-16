"""
Quick Data Structure Checker
Run this to see what files are in each folder
"""

from pathlib import Path
import sys

def check_data_structure(data_root):
    """Check and display the data structure"""
    data_path = Path(data_root)
    
    if not data_path.exists():
        print(f"❌ Directory not found: {data_root}")
        return
    
    print("=" * 80)
    print(f"Data Structure in: {data_root}")
    print("=" * 80)
    
    folders = sorted([d for d in data_path.iterdir() if d.is_dir()])
    
    if len(folders) == 0:
        print("❌ No folders found!")
        return
    
    print(f"\n📁 Found {len(folders)} folders\n")
    
    # Check first 10 folders
    for i, folder in enumerate(folders[:10]):
        print(f"\n{i+1}. {folder.name}/")
        
        # List all files
        files = sorted(folder.glob('*'))
        for file in files:
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                
                # Add emoji based on file type
                if file.suffix == '.png':
                    if file.name == 'rgb.png':
                        emoji = "🟢"
                    elif file.name == 'depth_no_bg.png':
                        emoji = "🔵"
                    else:
                        emoji = "🖼️ "
                elif file.suffix == '.ply':
                    if file.name.endswith('_nc.ply'):
                        emoji = "✨"
                    else:
                        emoji = "📊"
                else:
                    emoji = "📄"
                
                print(f"   {emoji} {file.name} ({size_mb:.2f} MB)")
    
    if len(folders) > 10:
        print(f"\n... and {len(folders) - 10} more folders")
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ Required files per folder:")
    print("   🟢 rgb.png")
    print("   🔵 depth_no_bg.png")
    print("   ✨ *_nc.ply (e.g., Sorghum_9_nc.ply)")
    print("=" * 80)
    
    # Check completeness
    print("\n🔍 Checking completeness...")
    complete = 0
    incomplete = []
    
    for folder in folders:
        has_rgb = (folder / 'rgb.png').exists()
        has_depth = (folder / 'depth_no_bg.png').exists()
        has_pc = len(list(folder.glob('*_nc.ply'))) > 0
        
        if has_rgb and has_depth and has_pc:
            complete += 1
        else:
            incomplete.append(folder.name)
    
    print(f"✅ Complete samples: {complete}/{len(folders)}")
    
    if incomplete:
        print(f"⚠️  Incomplete samples: {len(incomplete)}")
        if len(incomplete) <= 5:
            for name in incomplete:
                print(f"   - {name}")
        else:
            print(f"   (showing first 5)")
            for name in incomplete[:5]:
                print(f"   - {name}")
    
    print("\n" + "=" * 80)
    
    if complete > 0:
        print("✅ Ready for training!")
        print(f"\nNext step:")
        print(f"  python validate_sorghum_data.py --data_root {data_root}")
    else:
        print("❌ No complete samples found. Please check your data structure.")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        data_root = sys.argv[1]
    else:
        print("Usage: python check_structure.py /path/to/data")
        print("\nOr just provide the path when prompted:")
        data_root = input("Enter data directory path: ").strip()
    
    check_data_structure(data_root)
