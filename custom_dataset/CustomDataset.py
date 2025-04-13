import numpy as np
import tensorflow as tf
from pathlib import Path
import imageio.v2 as imageio
import cv2
from sklearn.utils import shuffle
from skimage.io import imread
import pandas as pd

class RTDatasetTF:
    def __init__(self, dataset_dir, add_channel = True, batch_size=8, normalize=True, train=True, thin_label=False, buffer_size = None):
        self.image_dir = Path(dataset_dir) / ("imagery" if train else "imagery_test")
        self.mask_dir = Path(dataset_dir) / ("masks" if thin_label else "masks_thick")
        self.batch_size = batch_size
        self.normalize = normalize
        self.augment = train
        self.shuffle_data = train
        self.add_channel = add_channel

        print("📂 Looking for images in:", self.image_dir.resolve())
        print("🔎 Pattern: *.png")
        self.image_files = sorted(self.image_dir.glob("*.png"))
        print("📸 Found:", len(self.image_files), "images")

        if buffer_size:
            self.buffer_size = buffer_size
        else: 
            self.buffer_size = len(self.image_files)
            
        self.mask_files = [
            self.mask_dir / f"{'_'.join(f.stem.split('_')[:-4])}_osm_{'_'.join(f.stem.split('_')[4:])}.png"
            for f in self.image_files
        ]

        if self.shuffle_data:
            self.image_files, self.mask_files = shuffle(self.image_files, self.mask_files)

        # Tính steps_per_epoch và dataset
        self.steps_per_epoch = len(self.image_files) // self.batch_size
        self.dataset = self.build_dataset()

    def load_pair(self, image_path, mask_path):
        image = imageio.imread(image_path.decode("utf-8"))
        mask = imageio.imread(mask_path.decode("utf-8"))[:, :, 0]
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

    def tf_load_pair(self, image_path, mask_path, filename):
        image, mask = tf.numpy_function(self.load_pair, [image_path, mask_path], [tf.float32, tf.float32])
        image.set_shape([None, None, 3])
        mask.set_shape([None, None, 1])
        return image, mask, filename

    def tf_augment_pair(self, image, mask):
        image, mask = tf.numpy_function(self.augment_pair_np, [image, mask], [tf.float32, tf.float32])
        image.set_shape([None, None, 3])
        mask.set_shape([None, None, 1])

        if self.add_channel: 
            # add channel 4th full 0s
            zero_channel = tf.zeros_like(image[..., :1])
            image = tf.concat([image, zero_channel], axis=-1)  # (H, W, 4)
            image.set_shape([None, None, 4])  # Cập nhật shape sau concat

        return image, mask

    def build_dataset(self):
        img_paths = [str(p) for p in self.image_files]
        mask_paths = [str(p) for p in self.mask_files]
        filenames = [p.name for p in self.image_files]

        dataset = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths, filenames))

        if not self.augment:  # train=False
            dataset = dataset.map(self.tf_load_pair, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        else:
            dataset = dataset.map(lambda x, y, z: self.tf_load_pair(x, y, z)[0:2], num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.map(self.tf_augment_pair, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.shuffle(buffer_size=self.buffer_size, reshuffle_each_iteration=True)
            dataset = dataset.batch(self.batch_size).repeat().prefetch(tf.data.AUTOTUNE)

        return dataset


class MassachusettsDatasetTF:
    def __init__(self, dataset_dir, batch_size=8, normalize=True, split='train', add_channel=True, buffer_size = None):
        self.dataset_dir = Path(dataset_dir)
        self.batch_size = batch_size
        self.normalize = normalize
        self.augment = (split == 'train')
        self.shuffle_data = (split == 'train')
        self.add_channel = add_channel

        df = pd.read_csv(self.dataset_dir / "metadata.csv", sep=',', header=0)
        print(df.columns.tolist())
        df = df[df['split'] == split]

        print("📂 Looking for images in:", self.image_dir.resolve())
        print("🔎 Pattern: *.png")
        self.image_files = [self.dataset_dir / p for p in df['tiff_image_path'].tolist()]
        print("📸 Found:", len(self.image_files), "images")

        if buffer_size:
            self.buffer_size = buffer_size
        else: 
            self.buffer_size = len(self.image_files)
            
        self.mask_files = [self.dataset_dir / p for p in df['tif_label_path'].tolist()]

        if self.shuffle_data:
            self.image_files, self.mask_files = shuffle(self.image_files, self.mask_files)

        self.steps_per_epoch = len(self.image_files) // self.batch_size
        self.dataset = self.build_dataset()

    def load_pair(self, image_path, mask_path):
        image = imageio.imread(image_path.decode('utf-8'))
        mask = imageio.imread(mask_path.decode('utf-8'))
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

        if self.add_channel:
            zero_channel = tf.zeros_like(image[..., :1])  # (H, W, 1)
            image = tf.concat([image, zero_channel], axis=-1)  # (H, W, 4)
            image.set_shape([None, None, 4])

        return image, mask

    def build_dataset(self):
        img_paths = [str(p) for p in self.image_files]
        mask_paths = [str(p) for p in self.mask_files]

        dataset = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
        dataset = dataset.map(self.tf_load_pair, num_parallel_calls=tf.data.AUTOTUNE)

        if self.augment:
            dataset = dataset.map(self.tf_augment_pair, num_parallel_calls=tf.data.AUTOTUNE)

        if self.shuffle_data:
            dataset = dataset.shuffle(buffer_size=self.buffer_size, reshuffle_each_iteration=True)

        dataset = dataset.batch(self.batch_size).repeat().prefetch(tf.data.AUTOTUNE)
        return dataset

