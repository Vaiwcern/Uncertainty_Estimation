import os
import cv2
import argparse

# Argument parser
parser = argparse.ArgumentParser(description="Split images into crops by rows and columns (preserving folder structure)")
parser.add_argument("-i", "--input", type=str, required=True, help="Path to the original dataset folder")
parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output cropped folder")
parser.add_argument("--split_height", type=int, required=True, help="Number of vertical splits (rows)")
parser.add_argument("--split_width", type=int, required=True, help="Number of horizontal splits (columns)")
args = parser.parse_args()

# Parameters
dataset_dir = args.input
output_dir = args.output
split_h = args.split_height
split_w = args.split_width

os.makedirs(output_dir, exist_ok=True)
    
def split_image(image_path, output_folder):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error loading image: {image_path}")
        return

    height, width, _ = img.shape

    if height % split_h != 0 or width % split_w != 0:
        print(f"⚠️ Skipping non-evenly divisible image: {image_path}")
        return

    crop_h = height // split_h
    crop_w = width // split_w
    filename = os.path.splitext(os.path.basename(image_path))[0]

    for i in range(split_h):
        for j in range(split_w):
            y_start = i * crop_h
            x_start = j * crop_w
            crop = img[y_start:y_start + crop_h, x_start:x_start + crop_w]

            new_filename = f"{filename}_coord_{j}_{i}.png"
            relative_path = os.path.relpath(os.path.dirname(image_path), dataset_dir)
            output_subfolder = os.path.join(output_folder, relative_path)
            os.makedirs(output_subfolder, exist_ok=True)

            save_path = os.path.join(output_subfolder, new_filename)
            cv2.imwrite(save_path, crop)

# Traverse and split
for root, _, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith(".png"):
            image_path = os.path.join(root, file)
            split_image(image_path, output_dir)

print("✅ Done! All images have been split and saved.")
