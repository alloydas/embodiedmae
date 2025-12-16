from PIL import Image
import numpy as np
import os

# Set the parent folder containing all subfolders
parent_folder = './Dataset/SorghumData'

# Walk through all subdirectories
for root, dirs, files in os.walk(parent_folder):
    if 'rgb.png' in files and 'depth.png' in files:
        rgb_path = os.path.join(root, 'rgb.png')
        depth_path = os.path.join(root, 'depth_vis.png')
        output_path = os.path.join(root, 'depth_no_bg_vis.png')
        
        # Load both images
        rgb_img = Image.open(rgb_path).convert('RGBA')
        depth_img = Image.open(depth_path).convert('RGB')
        
        rgb_data = np.array(rgb_img)
        depth_data = np.array(depth_img)
        
        # Detect background color from RGB (top-left corner)
        bg_color = rgb_data[0, 0, :3]
        
        # Create mask for background pixels
        tolerance = 5
        bg_mask = np.all(np.abs(rgb_data[:, :, :3].astype(int) - bg_color.astype(int)) <= tolerance, axis=2)
        
        # Alpha: 255 for foreground, 0 for background
        alpha_mask = np.where(bg_mask, 0, 255).astype(np.uint8)
        
        # Apply mask to depth image
        depth_rgba = np.zeros((*depth_data.shape[:2], 4), dtype=np.uint8)
        depth_rgba[:, :, 0] = depth_data[:,:,0]
        depth_rgba[:, :, 1] = depth_data[:,:,1]
        depth_rgba[:, :, 2] = depth_data[:,:,2]
        depth_rgba[:, :, 3] = alpha_mask
        
        # Save
        output_img = Image.fromarray(depth_rgba)
        output_img.save(output_path)
        
        print(f'Processed: {root}')

print('Done!')
