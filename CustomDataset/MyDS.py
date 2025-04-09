import tensorflow as tf
import numpy as np
from pathlib import Path
from skimage.io import imread
from sklearn.utils import shuffle
import cv2
import imageio.v2 as imageio


class MyDSTF:
    def __init__(self, dataset_dir, channel = 4, batch_size=8, normalize=True, train=True, thin_label=False):
        self.image_dir = Path(dataset_dir) / ("train/image" if train else "test/image")
        self.mask_dir = Path(dataset_dir) / ("train/mask" if train else "test/mask")
        self.batch_size = batch_size
        self.normalize = normalize
        self.augment = train
        self.shuffle_data = train
        self.channel = channel

        self.image_files = sorted(self.image_dir.glob("*.png"))
        self.mask_files = [self.mask_dir / f"{p.stem}.png" for p in self.image_files]

        if self.shuffle_data:
            self.image_files, self.mask_files = shuffle(self.image_files, self.mask_files)

        # Tính steps_per_epoch và dataset
        self.steps_per_epoch = len(self.image_files) // self.batch_size
        self.dataset = self.build_dataset()

    def load_pair(self, image_path, mask_path):
        image = imageio.imread(image_path.decode("utf-8"))
        mask = imageio.imread(mask_path.decode("utf-8"))
        mask = (mask >= 128).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)

        if self.normalize:
            image = image / 255.0

        return image.astype(np.float32), mask.astype(np.float32)

    def augment_pair_np(self, image, mask):
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
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR)
            mask = cv2.warpAffine(mask.squeeze(), matrix, (w, h), flags=cv2.INTER_NEAREST)
            mask = np.expand_dims(mask, axis=-1)
        return image.astype(np.float32), mask.astype(np.float32)

    def tf_load_pair(self, image_path, mask_path):
        image, mask = tf.numpy_function(self.load_pair, [image_path, mask_path], [tf.float32, tf.float32])
        image.set_shape([None, None, 3])
        mask.set_shape([None, None, 1])
        return image, mask

    def tf_augment_pair(self, image, mask):
        image, mask = tf.numpy_function(self.augment_pair_np, [image, mask], [tf.float32, tf.float32])
        image.set_shape([None, None, 3])
        mask.set_shape([None, None, 1])

        if (self.channel == 4):
            # ✅ Thêm channel thứ 4 toàn số 0
            zero_channel = tf.zeros_like(image[..., :1])
            image = tf.concat([image, zero_channel], axis=-1)  # (H, W, 4)
            image.set_shape([None, None, 4])  # Cập nhật shape sau concat

        return image, mask

    def build_dataset(self):
        img_paths = [str(p) for p in self.image_files]
        mask_paths = [str(p) for p in self.mask_files]

        dataset = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
        dataset = dataset.map(self.tf_load_pair, num_parallel_calls=tf.data.AUTOTUNE)

        if self.augment:
            dataset = dataset.map(self.tf_augment_pair, num_parallel_calls=tf.data.AUTOTUNE)

        if self.shuffle_data:
            dataset = dataset.shuffle(buffer_size=3840, reshuffle_each_iteration=True)

        # ✅ repeat để tránh OutOfRange
        dataset = dataset.batch(self.batch_size).repeat().prefetch(tf.data.AUTOTUNE)
        return dataset


# class MyDSTF:
#     def __init__(self, dataset_dir, batch_size=8, normalize=True, train=True, thin_label=False):
#         self.image_dir = Path(dataset_dir) / ("train/image" if train else "test/image")
#         self.mask_dir = Path(dataset_dir) / ("train/mask" if train else "test/mask")
#         self.batch_size = batch_size
#         self.normalize = normalize
#         # self.augment = train
#         self.augment = False
#         self.shuffle_data = train
#
#         self.image_files = sorted(self.image_dir.glob("*.png"))
#         self.mask_files = [self.mask_dir / f"{p.stem}.png" for p in self.image_files]
#
#         if self.shuffle_data:
#             self.image_files, self.mask_files = shuffle(self.image_files, self.mask_files)
#
#         # Tính steps_per_epoch và dataset
#         self.steps_per_epoch = len(self.image_files) // self.batch_size
#         self.dataset = self.build_dataset()
#
#     def load_pair(self, image_path, mask_path):
#         image = imageio.imread(image_path.decode("utf-8"))
#         mask = imageio.imread(mask_path.decode("utf-8"))
#         # if mask.ndim == 3:
#         #     mask = mask[:, :, 0]
#         # mask = (mask >= 128).astype(np.float32)
#         mask = np.expand_dims(mask, axis=-1)
#
#         if self.normalize:
#             image = image / 255.0
#
#         return image.astype(np.float32), mask.astype(np.float32)
#
#     def augment_pair_np(self, image, mask):
#         if np.random.rand() < 0.5:
#             image = np.fliplr(image)
#             mask = np.fliplr(mask)
#         if np.random.rand() < 0.5:
#             image = np.flipud(image)
#             mask = np.flipud(mask)
#         if np.random.rand() < 0.5:
#             angle = np.random.uniform(-180, 180)
#             h, w = image.shape[:2]
#             center = (w // 2, h // 2)
#             matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
#             image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR)
#             mask = cv2.warpAffine(mask.squeeze(), matrix, (w, h), flags=cv2.INTER_NEAREST)
#             mask = np.expand_dims(mask, axis=-1)
#         return image.astype(np.float32), mask.astype(np.float32)
#
#     def tf_load_pair(self, image_path, mask_path):
#         image, mask = tf.numpy_function(self.load_pair, [image_path, mask_path], [tf.float32, tf.float32])
#         image.set_shape([None, None, 3])
#         mask.set_shape([None, None, 1])
#         return image, mask
#
#     def tf_augment_pair(self, image, mask):
#         image, mask = tf.numpy_function(self.augment_pair_np, [image, mask], [tf.float32, tf.float32])
#         image.set_shape([None, None, 3])
#         mask.set_shape([None, None, 1])
#
#         # ✅ Thêm channel thứ 4 toàn số 0
#         zero_channel = tf.zeros_like(image[..., :1])
#         image = tf.concat([image, zero_channel], axis=-1)  # (H, W, 4)
#         image.set_shape([None, None, 4])  # Cập nhật shape sau concat
#
#         return image, mask
#
#     def build_dataset(self):
#         img_paths = [str(p) for p in self.image_files]
#         mask_paths = [str(p) for p in self.mask_files]
#
#         dataset = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
#         dataset = dataset.map(self.tf_load_pair, num_parallel_calls=tf.data.AUTOTUNE)
#
#         if self.augment:
#             dataset = dataset.map(self.tf_augment_pair, num_parallel_calls=tf.data.AUTOTUNE)
#
#         if self.shuffle_data:
#             dataset = dataset.shuffle(buffer_size=3840, reshuffle_each_iteration=True)
#
#         # ✅ repeat để tránh OutOfRange
#         dataset = dataset.batch(self.batch_size).repeat().prefetch(tf.data.AUTOTUNE)
#         return dataset

class MyDSTFold:
    def __init__(self, image_dir, mask_dir, batch_size=8, shuffle_data=True, normalize=True, augment=False):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.batch_size = batch_size
        self.shuffle_data = shuffle_data
        self.normalize = normalize
        self.augment = augment

        self.image_paths = sorted(self.image_dir.glob("*.png"))
        self.mask_paths = [self.mask_dir / f"{p.stem}.png" for p in self.image_paths]

        if self.shuffle_data:
            self.image_paths, self.mask_paths = shuffle(self.image_paths, self.mask_paths)

        self.dataset = self.build_dataset()

    def load_pair(self, image_path, mask_path):
        image = imageio.imread(image_path.decode('utf-8'))
        mask = imageio.imread(mask_path.decode('utf-8'))

        if self.normalize:
            image = image / 255.0
            mask = mask / 255.0

        mask = np.expand_dims(mask, axis=-1)

        return image.astype(np.float32), mask.astype(np.float32)

    def augment_pair_np(self, image, mask):
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
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LINEAR)
            mask = cv2.warpAffine(mask.squeeze(), rot_mat, (w, h), flags=cv2.INTER_NEAREST)
            mask = np.expand_dims(mask, axis=-1)

        return image.astype(np.float32), mask.astype(np.float32)

    def tf_load_pair(self, image_path, mask_path):
        image, mask = tf.numpy_function(self.load_pair, [image_path, mask_path], [tf.float32, tf.float32])
        image.set_shape([None, None, 3])  # hoặc 4 nếu ảnh có alpha
        mask.set_shape([None, None, 1])
        return image, mask

    def tf_augment_pair(self, image, mask):
        image, mask = tf.numpy_function(self.augment_pair_np, [image, mask], [tf.float32, tf.float32])
        image.set_shape([None, None, 3])
        mask.set_shape([None, None, 1])
        return image, mask

    def build_dataset(self):
        img_paths = [str(p) for p in self.image_paths]
        mask_paths = [str(p) for p in self.mask_paths]

        dataset = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
        dataset = dataset.map(self.tf_load_pair, num_parallel_calls=tf.data.AUTOTUNE)

        if self.augment:
            dataset = dataset.map(self.tf_augment_pair, num_parallel_calls=tf.data.AUTOTUNE)

        if self.shuffle_data:
            dataset = dataset.shuffle(buffer_size=100, reshuffle_each_iteration=True)

        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset


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


def my_resize(path, size=128, mask=False):
    image = cv2.imread(path)
    image = cv2.resize(image, (size, size))
    if mask == True:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # shape: (size,size,3) -> (size,size,1)
    return image

class MyDSOld(tf.keras.utils.Sequence):
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
