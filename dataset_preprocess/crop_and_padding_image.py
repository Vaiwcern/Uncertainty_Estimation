import os
import pandas as pd
from pathlib import Path
import imageio.v2 as imageio  # ✅ tránh warning từ imageio v3
import numpy as np
from tqdm import tqdm

# ==== Cấu hình ====
original_dataset_dir = Path("/home/ltnghia02/MEDICAL_ITERATIVE/Dataset/Massachusetts")
new_dataset_dir = Path("/home/ltnghia02/MEDICAL_ITERATIVE/Dataset/Massachusetts_Crop")
crop_size = 500
final_size = 512
padding = (final_size - crop_size) // 2

# ==== Tạo thư mục mới giữ nguyên cấu trúc ====
subdirs = [
    "tiff/train", "tiff/train_labels",  
    "tiff/test", "tiff/test_labels",
    "tiff/val", "tiff/val_labels"
]
for sub in subdirs:
    os.makedirs(new_dataset_dir / sub, exist_ok=True)

# ==== Đọc metadata gốc ====
metadata_path = original_dataset_dir / "metadata.csv"
df = pd.read_csv(metadata_path, sep=',')

# ==== Danh sách metadata mới ====
new_metadata = []

# ==== Hàm crop và pad ảnh ====
def crop_and_pad_image(img, x, y):
    cropped = img[y*crop_size:(y+1)*crop_size, x*crop_size:(x+1)*crop_size]

    if cropped.ndim == 2:  # grayscale (mask)
        padded = np.pad(cropped, ((padding, padding), (padding, padding)), mode='constant')
    else:  # RGB ảnh
        padded = np.pad(cropped, ((padding, padding), (padding, padding), (0, 0)), mode='constant')

    return padded

# ==== Tiền xử lý từng ảnh ====
for idx, row in tqdm(df.iterrows(), total=len(df)):
    img_path = original_dataset_dir / row['tiff_image_path']
    lbl_path = original_dataset_dir / row['tif_label_path']

    image = imageio.imread(img_path)
    label = imageio.imread(lbl_path)

    for x in range(3):
        for y in range(3):
            # Crop và pad
            image_crop = crop_and_pad_image(image, x, y)
            label_crop = crop_and_pad_image(label, x, y)

            # Tạo ID và đường dẫn mới
            base_id = row['image_id']
            new_id = f"{base_id}_coord_{x}_{y}"
            image_out_path = f"tiff/{row['split']}/{new_id}.tif"
            label_out_path = f"tiff/{row['split']}_labels/{new_id}.tif"

            # Lưu ảnh
            imageio.imwrite(new_dataset_dir / image_out_path, image_crop)
            imageio.imwrite(new_dataset_dir / label_out_path, label_crop)

            # Ghi lại metadata
            new_metadata.append({
                "image_id": new_id,
                "split": row['split'],
                "tiff_image_path": image_out_path,
                "tif_label_path": label_out_path
            })

# ==== Lưu file metadata mới ====
new_df = pd.DataFrame(new_metadata)
new_df.to_csv(new_dataset_dir / "metadata.csv", sep=',', index=False)

print("✅ Done. Đã lưu ảnh và metadata crop vào:", new_dataset_dir)
