import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from CustomDataset.CustomDataset import MassachusettsDatasetTF

# Tạo dataset test
test_dataset = MassachusettsDatasetTF(
    dataset_dir="/home/ltnghia02/MEDICAL_ITERATIVE/Dataset/Massachusetts_Crop",
    batch_size=4,       # bạn có thể thay đổi tùy ý
    split='test',
    channel=4,          # hoặc 3 nếu không muốn thêm channel 0
    normalize=True
)

print("✅ Số lượng ảnh:", len(test_dataset.image_files))
print("✅ Step per epoch:", test_dataset.steps_per_epoch)

# Lấy batch đầu tiên
for images, masks in test_dataset.dataset.take(1):
    # Tạo thư mục nếu chưa có
    # os.makedirs("preview", exist_ok=True)
    
    for i in range(images.shape[0]):
        img = images[i].numpy()
        mask = masks[i].numpy().squeeze()

        # Nếu có 4 channels, bỏ kênh cuối để hiển thị RGB
        if img.shape[-1] == 4:
            img = img[..., :3]

        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(img)
        axs[0].set_title("Image")
        axs[0].axis('off')
        axs[1].imshow(mask, cmap='gray')
        axs[1].set_title("Mask")
        axs[1].axis('off')

        plt.tight_layout()
        plt.savefig(f"/home/ltnghia02/MEDICAL_ITERATIVE/Uncertainty_Estimation/dataset_preprocess/preview_2/sample_{i}.png")
        plt.close()

    break  # chỉ cần 1 batch
