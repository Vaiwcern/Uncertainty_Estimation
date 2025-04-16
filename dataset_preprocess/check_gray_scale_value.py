import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def plot_and_save_gray_histogram(image_path, num_bins):
    # Đọc ảnh grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Không đọc được ảnh.")

    # Tính histogram
    hist, _ = np.histogram(image.flatten(), bins=num_bins, range=[0, 256])

    # Vẽ nhưng không hiển thị
    plt.figure(figsize=(10, 5))
    plt.bar(range(num_bins), hist, width=1.0, edgecolor='black')
    plt.title("Histogram of Grayscale Image")
    plt.xlabel("Pixel Intensity (0-255)")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    # Tạo tên file output trong current working directory
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(os.getcwd(), f"{base_name}_hist.png")

    # Lưu
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save histogram of a grayscale image to current folder.")
    parser.add_argument("--image_path", type=str, help="Path to grayscale image.")
    parser.add_argument("--num_bins", type=int, default=128, help="Number of bins (default: 128)")
    
    args = parser.parse_args()
    plot_and_save_gray_histogram(args.image_path, args.num_bins)
