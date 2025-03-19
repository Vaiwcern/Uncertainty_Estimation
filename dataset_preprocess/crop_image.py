import os
import cv2
import argparse

# Define argument parser
parser = argparse.ArgumentParser(description="Split images into 16 crops of 1024x1024 while preserving folder structure")
parser.add_argument("-i", "--input", type=str, required=True, help="Path to the original dataset folder")
parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output cropped folder")
args = parser.parse_args()

# Define input and output directories
dataset_dir = args.input
output_dir = args.output

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

def split_image(image_path, output_folder):
    """Splits an image into 16 crops and saves them while preserving folder structure."""
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return
    
    # Get the original filename without extension
    filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Define crop size and coordinates
    crop_size = 1024
    coords = [(x, y) for x in range(4) for y in range(4)]
    
    for x, y in coords:
        # Calculate the crop coordinates
        x_start, y_start = x * crop_size, y * crop_size
        crop = img[y_start:y_start + crop_size, x_start:x_start + crop_size]
        
        # Define new filename
        new_filename = f"{filename}_coord_{x}_{y}.png"
        
        # Create corresponding subdirectory in output folder
        relative_path = os.path.relpath(os.path.dirname(image_path), dataset_dir)
        output_subfolder = os.path.join(output_folder, relative_path)
        os.makedirs(output_subfolder, exist_ok=True)
        
        # Save path with original structure
        save_path = os.path.join(output_subfolder, new_filename)
        
        # Save the cropped image
        cv2.imwrite(save_path, crop)

# Traverse dataset directory
for root, _, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith(".png"):
            image_path = os.path.join(root, file)
            split_image(image_path, output_dir)

print("Processing complete! Cropped images are saved in the specified output folder while preserving folder structure.")
