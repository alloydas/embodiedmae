from PIL import Image
import numpy as np
import os

# Set the parent folder containing all subfolders
parent_folder = './Dataset/SorghumData'

# Walk through all subdirectories
for root, dirs, files in os.walk(parent_folder):
    if 'rgb.png' in files:
        input_path = os.path.join(root, 'rgb.png')
        output_path = os.path.join(root, 'mask.png')
        
        # Load the image
        img = Image.open(input_path).convert('RGBA')
        data = np.array(img)
        
        # Detect the background color from the top-left corner
        bg_color = data[0, 0, :3]
        
        # Create a mask for pixels that match the background color
        tolerance = 5
        mask = np.all(np.abs(data[:, :, :3].astype(int) - bg_color.astype(int)) <= tolerance, axis=2)
        
        # Set alpha to 0 for background pixels
        result = data.copy()
        result[mask, 3] = 0
        
        # Save the result
        output_img = Image.fromarray(result)
        output_img.save(output_path)
        
        print(f'Processed: {input_path} -> {output_path}')

print('Done!')
