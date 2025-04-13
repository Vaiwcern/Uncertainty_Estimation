import os
import cv2
import argparse

# Define argument parser
parser = argparse.ArgumentParser(description="Split square images into NxN crops while preserving folder structure")
parser.add_argument("-i", "--input", type=str, required=True, help="Path to the original dataset folder")
parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output cropped folder")
parser.add_argument("--split", type=int, default=4, help="Number of splits per side (e.g., 8 means 8x8 = 64 crops)")
args = parser.parse_args()

# Define input and output directories
dataset_dir = args.input
output_dir = args.output
split = args.split

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

def split_image(image_path, output_folder):
    """Splits a square image into NxN crops and saves them while preserving folder structure."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error loading image: {image_path}")
        return

    height, width, _ = img.shape
    if height != width:
        print(f"⚠️ Skipping non-square image: {image_path}")
        return

    crop_size = width // split
    if width % split != 0:
        print(f"⚠️ Image not evenly divisible: {image_path}")
        return

    filename = os.path.splitext(os.path.basename(image_path))[0]

    for x in range(split):
        for y in range(split):
            x_start = x * crop_size
            y_start = y * crop_size
            crop = img[y_start:y_start + crop_size, x_start:x_start + crop_size]

            new_filename = f"{filename}_coord_{x}_{y}.png"
            relative_path = os.path.relpath(os.path.dirname(image_path), dataset_dir)
            output_subfolder = os.path.join(output_folder, relative_path)
            os.makedirs(output_subfolder, exist_ok=True)

            save_path = os.path.join(output_subfolder, new_filename)
            cv2.imwrite(save_path, crop)

# Traverse dataset directory
for root, _, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith(".png"):
            image_path = os.path.join(root, file)
            split_image(image_path, output_dir)

print("✅ Done! All square images are split and saved.")
