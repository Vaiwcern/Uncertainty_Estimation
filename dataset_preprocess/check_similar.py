import imageio.v3 as iio
import numpy as np
import sys

def compare_images(path1, path2):
    img1 = iio.imread(path1)
    img2 = iio.imread(path2)

    if img1.shape != img2.shape:
        print("❌ Ảnh có shape khác nhau:", img1.shape, "vs", img2.shape)
        return False

    total_pixels = np.prod(img1.shape)
    diff_mask = img1 != img2
    num_diff = np.count_nonzero(diff_mask)
    num_same = total_pixels - num_diff
    similarity = (num_same / total_pixels) * 100

    if num_diff == 0:
        print("✅ Hai ảnh giống nhau hoàn toàn.")
    else:
        print("❌ Hai ảnh KHÁC nhau!")
        print(f"🔹 Pixel giống nhau   : {num_same:,}")
        print(f"🔸 Pixel khác nhau    : {num_diff:,}")
        print(f"📊 Tỷ lệ giống nhau  : {similarity:.4f}%")

    return num_diff == 0

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Cách dùng: python compare_images.py <image1.png> <image2.png>")
        sys.exit(1)

    compare_images(sys.argv[1], sys.argv[2])
