import numpy as np
import cv2
import os


image_paths = ["/home/ltnghia02/MEDICAL_ITERATIVE/trained_model/RTdata_iter_dropout_model/predict_epoch_55/amsterdam_-1_-1_sat_coord_0_0_output_0.png", "/home/ltnghia02/MEDICAL_ITERATIVE/trained_model/RTdata_iter_dropout_model/predict_epoch_55/amsterdam_-1_-1_sat_coord_0_0_output_4.png", "/home/ltnghia02/MEDICAL_ITERATIVE/trained_model/RTdata_iter_dropout_model/predict_epoch_55/amsterdam_-1_-1_sat_coord_0_0_output_3.png"]


output_image_path = "variance.png"
output_npy_path = "variance.npy"

# === Đọc ảnh grayscale và chuẩn hóa về [0, 1] ===
images = []
for path in image_paths:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Không thể đọc ảnh: {path}")
    images.append(img.astype(np.float32) / 255.0)   

print("Ảnh 0 vs 1:")
print("All close:", np.allclose(images[0], images[1]))
print("Max diff:", np.max(np.abs(images[0] - images[1])))

print("\nẢnh 0 vs 2:")
print("All close:", np.allclose(images[0], images[2]))
print("Max diff:", np.max(np.abs(images[0] - images[2])))

print("\nẢnh 1 vs 2:")
print("All close:", np.allclose(images[1], images[2]))
print("Max diff:", np.max(np.abs(images[1] - images[2])))

# === Ghép và tính phương sai theo pixel ===
# arr = np.stack(images, axis=0)  # (3, H, W)
# var_matrix = np.var(arr, axis=0)  # (H, W)

images = [img.astype(np.float32) for img in images]
arr = np.stack(images, axis=0)
var_matrix = np.var(arr, axis=0)

print("Max var:", np.max(var_matrix))
print("Min var:", np.min(var_matrix))
print("Variance > 1e-8:", np.sum(var_matrix > 1e-8))


# === Lưu ma trận phương sai thành ảnh grayscale ===
# Scale về [0, 255] để lưu ảnh
var_norm = cv2.normalize(var_matrix, None, 0, 255, cv2.NORM_MINMAX)
var_uint8 = var_norm.astype(np.uint8)
cv2.imwrite(output_image_path, var_uint8)

# === (Optional) Lưu file .npy để phân tích sau ===
np.save(output_npy_path, var_matrix)

print(f"✅ Saved variance image to: {output_image_path}")
print(f"📁 Saved raw variance matrix to: {output_npy_path}")
