import tensorflow as tf
import numpy as np
from pathlib import Path
from skimage.io import imread
from sklearn.utils import shuffle
import cv2
import imageio.v2 as imageio

def my_resize(path, size=128, mask=False):
    image = cv2.imread(path)
    image = cv2.resize(image, (size, size))
    if mask == True:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # shape: (size,size,3) -> (size,size,1)
    return image

class MyDS(tf.keras.utils.Sequence):
    def __init__(self, image_dir, mask_dir, batch_size, shuffle=True, normalize=True, augment=False):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.normalize = normalize
        self.augment = augment

        self.image_files = sorted(self.image_dir.glob("*.png"))
        self.mask_files = [self.mask_dir / f"{file.stem}.png" for file in self.image_files]

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_files) / self.batch_size))

    def __getitem__(self, index):
        batch_image_files = self.image_files[index * self.batch_size:(index + 1) * self.batch_size]
        batch_mask_files = self.mask_files[index * self.batch_size:(index + 1) * self.batch_size]

        images = np.array([self.load_image(file) for file in batch_image_files])
        masks = np.array([self.load_mask(file) for file in batch_mask_files])

        if self.augment:
            images, masks = self.apply_augmentation(images, masks)

        return images, masks

    def load_image(self, filepath):
        image = my_resize(filepath)
        if self.normalize:
            image = image / 255.0
        return image

    def load_mask(self, filepath):
        mask = my_resize(filepath, mask=True)
        mask = np.expand_dims(mask, axis=-1)
        if self.normalize:
            mask = mask / 255.0
        return mask

    def apply_augmentation(self, images, masks):
        for i in range(len(images)):
            image, mask = images[i], masks[i]

            # Chuyển đổi TensorFlow Tensor thành NumPy array nếu cần
            if isinstance(image, tf.Tensor):
                image = image.numpy()
            if isinstance(mask, tf.Tensor):
                mask = mask.numpy()

            # Đảm bảo dữ liệu có kiểu float32 để tránh lỗi OpenCV
            image = image.astype(np.float32)
            mask = mask.astype(np.float32)

            if np.random.rand() < 0.5:
                image = np.fliplr(image)
                mask = np.fliplr(mask)

            if np.random.rand() < 0.5:
                image = np.flipud(image)
                mask = np.flipud(mask)

            if np.random.rand() < 0.5:
                angle = np.random.uniform(-30, 30)
                h, w = image.shape[:2]
                center = (w // 2, h // 2)

                # Tạo ma trận xoay
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

                # Xoay ảnh với nội suy tuyến tính
                image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)

                # Xoay mask với nội suy gần nhất (cần squeeze trước và expand_dims sau)
                mask = cv2.warpAffine(mask.squeeze(), rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)
                mask = np.expand_dims(mask, axis=-1)  # Đảm bảo shape (608,576,1)

            images[i], masks[i] = image, mask

        return np.array(images), np.array(masks)

    def on_epoch_end(self):
        if self.shuffle:
            self.image_files, self.mask_files = shuffle(self.image_files, self.mask_files)
