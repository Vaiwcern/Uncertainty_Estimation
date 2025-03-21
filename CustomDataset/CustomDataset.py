import numpy as np
import tensorflow as tf
from pathlib import Path
import imageio.v2 as imageio
import cv2
from sklearn.utils import shuffle
from skimage.io import imread

class RTDataset(tf.keras.utils.Sequence):
    def __init__(self, dataset_dir, batch_size, normalize=True, train=True, thin_label=False):
        self.image_dir = Path(dataset_dir) / ("imagery" if train else "imagery_test")
        self.mask_dir = Path(dataset_dir) / ("masks" if thin_label else "masks_thick")
        
        self.batch_size = batch_size
        self.normalize = normalize

        if train: 
            self.augment = True
            self.shuffle = True
        else: 
            self.augment = False
            self.shuffle = False

        self.image_files = sorted(self.image_dir.glob("*.png"))
        self.mask_files = [
            self.mask_dir / f"{'_'.join(file.stem.split('_')[:-4])}_osm_{'_'.join(file.stem.split('_')[4:])}.png"
            for file in self.image_files
        ]

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_files) / self.batch_size))

    def __getitem__(self, index):
        batch_image_files = self.image_files[index * self.batch_size:(index + 1) * self.batch_size]
        batch_mask_files = self.mask_files[index * self.batch_size:(index + 1) * self.batch_size]

        images = np.array([self.load_image(file) for file in batch_image_files])
        masks = np.array([self.load_mask(file) for file in batch_mask_files])

        if self.augment:
            images, masks = self.apply_augmentation(images, masks)

        return images, masks

    def load_image(self, filepath):
        image = imageio.imread(filepath)
        if self.normalize:
            image = image / 255.0
        return image

    def load_mask(self, filepath):
        mask = imageio.imread(filepath)
        mask = mask[:,:,0]
        mask = (mask >= 128).astype(np.float32) 
        mask = np.expand_dims(mask, axis=-1)
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
                angle = np.random.uniform(-180, 180)
                h, w = image.shape[:2]
                center = (w // 2, h // 2)

                # Tạo ma trận xoay
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

                # Xoay ảnh với nội suy tuyến tính
                image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)

                # Xoay mask với nội suy gần nhất (cần squeeze trước và expand_dims sau)
                mask = cv2.warpAffine(mask.squeeze(), rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)
                mask = np.expand_dims(mask, axis=-1)
            
            images[i], masks[i] = image, mask

        return np.array(images), np.array(masks)

    def on_epoch_end(self):
        if self.shuffle:
            self.image_files, self.mask_files = shuffle(self.image_files, self.mask_files)

class DRIVEDataset(tf.keras.utils.Sequence):
    def __init__(self, image_dir, mask_dir, batch_size, shuffle=True, normalize=True, augment=True):
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
        image = imageio.imread(filepath)
        if self.normalize:
            image = image / 255.0
        return image

    def load_mask(self, filepath):
        mask = imageio.imread(filepath)
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


class CHASEDB1Dataset(tf.keras.utils.Sequence):
    def __init__(self, image_dir, mask_dir, batch_size, shuffle=True, normalize=True, augment=True):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.normalize = normalize
        self.augment = augment

        self.image_files = sorted(self.image_dir.glob("*.jpg"))
        self.mask_files = [self.mask_dir / f"{file.stem}_1stHO.png" for file in self.image_files]

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
        image = imageio.imread(filepath)
        if self.normalize:
            image = image / 255.0
        return image

    def load_mask(self, filepath):
        mask = imageio.imread(filepath)
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


class STAREDataset(tf.keras.utils.Sequence):
    def __init__(self, image_dir, mask_dir, batch_size, shuffle=True, normalize=True, augment=True):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.normalize = normalize
        self.augment = augment

        self.image_files = sorted(self.image_dir.glob("*.png"))
        self.mask_files = [self.mask_dir / f"{file.stem}.ah.png" for file in self.image_files]

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
        image = imageio.imread(filepath)
        if self.normalize:
            image = image / 255.0
        return image

    def load_mask(self, filepath):
        mask = imageio.imread(filepath)
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
