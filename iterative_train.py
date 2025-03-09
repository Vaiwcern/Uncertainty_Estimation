import warnings
import numpy as np
import os
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from imageio import imread
from sklearn.utils import shuffle
from PIL import Image
import cv2
from tensorflow.keras.optimizers import Adam
from types import SimpleNamespace
import imageio.v2 as imageio
import keras
from datetime import datetime
import math
from tensorflow.keras.metrics import MeanIoU, Precision, Recall
from tensorflow.keras.models import load_model

import segmentation_models as sm
from seggradcam.dataloaders import Cityscapes, DRIVE
from seggradcam.unet import csbd_unet, manual_unet, TrainUnet
from seggradcam.training_write import TrainingParameters, TrainingResults
from seggradcam.training_plots import plot_predict_and_gt, plot_loss, plot_metric
from seggradcam.seggradcam import SegGradCAM, SuperRoI, ClassRoI, PixelRoI, BiasRoI
from seggradcam.visualize_sgc import SegGradCAMplot

from DRIVEDataset import DRIVEDataset
from model import unet

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import matplotlib.image as mpimg
import numpy as np

def inference_train(model, train_dataset, epochs, batch_size, learning_rate, loss_fn, metrics, save_path, version, step_per_epoch=None):
    """
    Hàm huấn luyện mô hình với các tham số đầu vào.
    """

    if not step_per_epoch:
        print("Error: Step per epoch")
        return

    # Khởi tạo optimizer
    optimizer = Adam(learning_rate=learning_rate)

    # Biên dịch mô hình với các tham số
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    # Bắt đầu huấn luyện
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        # Huấn luyện trên train_dataset
        for step, (images, labels) in enumerate(train_dataset):
            if step >= step_per_epoch:
                break

            the_images = tf.unstack(images, axis=0)

            # Thêm kênh toàn 0 vào images (kênh thứ 4)
            zeros_channel = np.zeros_like(images[:, :, :, :1])  # Tạo một kênh mới có giá trị 0
            images_with_new_channel = np.concatenate([images, zeros_channel], axis=-1)  # Thêm kênh mới vào cuối cùng

            outputs = []  # Mảng để lưu output của 3 lần forward
            predictions = images_with_new_channel  # Bắt đầu với input images có kênh 0

            zeros_channel = np.zeros_like(the_images[0][:, :, :1])
            for i in range(batch_size):
                the_images[i] = np.concatenate([the_images[i], zeros_channel], axis=-1)

            # Tiến hành forward 3 lần
            with tf.GradientTape() as tape:
                for i in range(3):
                    # Lần forward
                    # print("Input shape: ", predictions.shape)
                    # print(type(predictions))

                    prediction = model(predictions, training=True)
                    outputs.append(prediction)

                    mymaps = []
                    for j in range(batch_size):
                        prop_from_layer = model.layers[-1].name
                        prop_to_layer = 'center_block'
                        cls = 0

                        clsroi = ClassRoI(model=model, image=the_images[j], cls=cls)
                        newsgc = SegGradCAM(model, the_images[j], cls, prop_to_layer,  prop_from_layer, roi=clsroi,
                                            normalize=True, abs_w=False, posit_w=False)
                        mymap = (newsgc.SGC())  # Heatmap với shape (H, W)
                        mymaps.append(mymap)

                    # print("Gradcam map: ", mymaps.shape)

                    for j in range(batch_size):
                        mymap_3d = np.expand_dims(mymaps[j], axis=-1)
                        the_images[j] = np.concatenate([the_images[j][:, :, :-1], mymap_3d], axis=-1)

                    # Thêm heatmap vào kênh thứ 4 của ảnh
                    mymaps_tensor = np.stack(mymaps, axis=0)
                    mymap_tensor = np.expand_dims(mymaps_tensor, axis=-1)
                    predictions = np.concatenate([images, mymap_tensor], axis=-1)

                # Tính loss cho cả 3 output và lấy trung bình
                total_loss = 0
                for i in range(3):
                    total_loss += loss_fn(labels, outputs[i])  # Tính loss cho từng output

                total_loss /= 3  # Lấy trung bình của 3 lần forward

            # Tính gradient và cập nhật trọng số mô hình
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # In giá trị loss sau mỗi step
            print(f"Step {step}: Loss = {total_loss.numpy()}")

        if (epoch + 1) % 20 == 0:
            model.save_weights(save_path + 'epoch_' + str(epoch + 1) + '_ver' + str(version)+ '.weights.h5')
            print("Checkpoint saved!")

if __name__ == "__main__":
    # Config
    BATCH_SIZE = 5
    LR = 1e-3
    EPOCHS = 150
    n_train = 20
    n_val = 20

    trainparam = SimpleNamespace(
        dataset_name="DRIVE",
        learning_rate=LR,
        n_classes=1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        n_train=n_train,
        n_val=n_val,
        steps_per_epoch=(n_train + BATCH_SIZE - 1) // BATCH_SIZE,
        validation_steps=(n_val + BATCH_SIZE - 1) // BATCH_SIZE,
        input_shape=(608, 576, 4),
        save_path="/home/ltnghia02/MEDICAL_ITERATIVE/iterative_model/"
    )

    # Log file
    log_file = os.path.join(trainparam.save_path, "training_log.txt")
    sys.stdout = open(log_file, "w")
    sys.stderr = sys.stdout  # Ghi cả lỗi vào file log

    # Đường dẫn tới dữ liệu train
    train_image_dir = "/content/DRIVE/training/images"
    train_mask_dir = "/content/DRIVE/training/1st_manual"
    train_dataset = DRIVEDataset(image_dir=train_image_dir, mask_dir=train_mask_dir, batch_size=trainparam.batch_size)

    # optimizer, loss, metric
    optim = keras.optimizers.Adam(learning_rate = LR)
    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.BinaryFocalLoss()
    total_loss = focal_loss
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    model = unet()
    mdir = trainparam.save_path
    # model.load_weights(mdir + 'epoch_40_ver6.weights.h5')

    # train
    inference_train(model,
        train_dataset,
        epochs=trainparam.epochs,
        batch_size=trainparam.batch_size,
        learning_rate=trainparam.learning_rate,
        loss_fn=total_loss, metrics=metrics,
        save_path = trainparam.save_path,
        version=1,
        step_per_epoch=trainparam.steps_per_epoch)