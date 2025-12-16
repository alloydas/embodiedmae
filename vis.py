import random
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

# Base path containing all the folders
base_path = Path("/work/mech-ai-scratch/alloy/embodiedmae/Dataset/SorghumData")

# Output directory for plots
output_path = Path("visualizations")
output_path.mkdir(exist_ok=True)

# Find all folders that contain rgb.png
folders = [f.parent for f in base_path.rglob("rgb.png")]

# Randomly select 10 folders
selected_folders = random.sample(folders, min(10, len(folders)))

# Visualize each folder separately
for folder in selected_folders:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    rgb_img = Image.open(folder / "rgb.png")
    depth_img = Image.open(folder / "depth_no_bg.png")
    depth_vis_img = Image.open(folder / "depth_no_bg_vis.png")
    
    axes[0].imshow(rgb_img)
    axes[0].set_title("RGB")
    axes[0].axis("off")
    
    axes[1].imshow(depth_img, cmap="gray")
    axes[1].set_title("Depth")
    axes[1].axis("off")
    
    axes[2].imshow(depth_vis_img)
    axes[2].set_title("Depth Vis")
    axes[2].axis("off")
    
    plt.suptitle(folder.name)
    plt.tight_layout()
    plt.savefig(output_path / f"{folder.name}.png", dpi=600)
    plt.close()  # Close to free memory

print(f"Saved {len(selected_folders)} plots to {output_path}")
