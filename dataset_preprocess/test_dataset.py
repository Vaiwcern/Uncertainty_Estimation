import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from custom_dataset.DatasetController import DatasetController

def save_image_with_mask(image, mask, filename, save_dir):
    """
    image: (H, W, 3) or (H, W, 4)
    mask: (H, W, 1)
    """
    if image.shape[-1] == 4:
        image = image[..., :3]

    mask_vis = np.repeat(mask, 3, axis=-1)  # convert mask to 3 channels

    combined = np.concatenate([image, mask_vis], axis=1)  # (H, W*2, 3)
    
    filename = os.path.splitext(filename)[0] + ".png"

    plt.imsave(os.path.join(save_dir, filename), combined)

def main():
    save_dir = "debug_output"
    os.makedirs(save_dir, exist_ok=True)

    train_ds = DatasetController.get_massachusetts_train_wrapper(
        dataset_path="/home/ltnghia02/MEDICAL_ITERATIVE/dataset/Massachusetts_Crop_512",
        batch_size=32,
        add_channel=True,
        buffer_size = 10
    ).dataset

    test_ds = DatasetController.get_massachusetts_test_wrapper(
        dataset_path="/home/ltnghia02/MEDICAL_ITERATIVE/dataset/Massachusetts_Crop_512",
        batch_size=4,
        add_channel=True,
    ).dataset

    # Lấy một batch train
    train_iter = iter(train_ds)
    train_images, train_masks = next(train_iter)

    # Lưu từng ảnh trong batch
    for i in range(train_images.shape[0]):
        img = train_images[i].numpy()
        msk = train_masks[i].numpy()
        save_image_with_mask(img, msk, f"train_{i}.png", save_dir)

    # Lấy một batch test
    test_iter = iter(test_ds)
    test_images, test_masks, test_filenames = next(test_iter)

    for i in range(test_images.shape[0]):
        img = test_images[i].numpy()
        msk = test_masks[i].numpy()
        save_image_with_mask(img, msk, f"test_{i}_{test_filenames[i].numpy().decode()}", save_dir)

    print(f"✅ Saved {train_images.shape[0]} train and {test_images.shape[0]} test samples to '{save_dir}'.")

if __name__ == "__main__":
    main()
