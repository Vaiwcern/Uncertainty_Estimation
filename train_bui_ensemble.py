import warnings
import numpy as np
import os
from pathlib import Path
import sys
# import matplotlib.pyplot as plt
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
import sys

import segmentation_models as sm
from seggradcam.dataloaders import Cityscapes, DRIVE
from seggradcam.unet import csbd_unet, manual_unet, TrainUnet
from seggradcam.training_write import TrainingParameters, TrainingResults
from seggradcam.training_plots import plot_predict_and_gt, plot_loss, plot_metric
from seggradcam.seggradcam import SegGradCAM, SuperRoI, ClassRoI, PixelRoI, BiasRoI
from seggradcam.visualize_sgc import SegGradCAMplot

from MyDS import MyDS
from model_drop import unet

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import matplotlib.image as mpimg
import numpy as np

def train(model, train_dataset, val_dataset, epochs, batch_size, learning_rate, loss_fn, metrics, save_path, step_per_epoch=None):
    """
    Hàm huấn luyện mô hình với các tham số đầu vào.

    :param model: Mô hình cần huấn luyện (Keras model)
    :param train_dataset: Dataset huấn luyện (thường là một generator hoặc Sequence)
    :param val_dataset: Dataset kiểm tra (validation dataset)
    :param epochs: Số epoch để huấn luyện
    :param batch_size: Kích thước batch
    :param learning_rate: Tốc độ học
    :param loss_fn: Hàm loss
    :param metrics: Các chỉ số (metrics) cần theo dõi trong quá trình huấn luyện
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
        print()
        print(f"Epoch {epoch+1}/{epochs}")

        # Huấn luyện trên train_dataset
        for step, (images, labels) in enumerate(train_dataset):
            # print(images.shape)
            # print(labels.shape)
            if step >= step_per_epoch:
                break
            with tf.GradientTape() as tape:
                # Tiền xử lý và dự đoán
                predictions = model(images, training=True)
                loss_value = loss_fn(labels, predictions)  # Tính toán loss

            # Tính gradient và cập nhật trọng số mô hình
            gradients = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # In giá trị loss sau mỗi step
            print(f"Step {step}: Loss = {loss_value.numpy()}")

        # Kiểm tra mô hình trên validation dataset sau mỗi epoch
        result = model.evaluate(val_dataset, verbose=1)

        val_loss = result[0]  # Loss là phần tử đầu tiên
        val_metrics = result[1]  # Metric là phần tử thứ hai

        print(f"Validation Loss: {val_loss}")

        # Nếu chỉ có một metric, val_metrics sẽ là float, không cần lặp qua
        if isinstance(val_metrics, list):  # Nếu có nhiều metrics
            for metric_name, val_metric in zip(model.metrics_names[1:], val_metrics):
                print(f"Validation {metric_name}: {val_metric}")
        else:  # Nếu chỉ có một metric
            print(f"Validation IoU: {val_metrics}")

        if (epoch + 1) % 50 == 0:  # Lưu model mỗi 5 epoch
            model.save_weights(save_path + 'epoch_' + str(epoch+1) + '_ver4.weights.h5')
            print("Model saved!")

if __name__ == "__main__":
    BASE_ROOT = "/home/ltnghia02/MEDICAL_ITERATIVE"
    BASE_DS = "/home/ltnghia02/MEDICAL_ITERATIVE/Dataset/my_breast_ultrasound_image"

    # Config
    BATCH_SIZE = 32
    LR = 1e-3
    EPOCHS = 50
    n_train = 582
    n_val = 65

    mdir = f"{BASE_ROOT}/model_BUI_ensemble/"

    if not os.path.exists(mdir):
        os.makedirs(mdir)

    NUM_MODELS = 5
    for i in range(NUM_MODELS):
        model_folder = f"{mdir}model_{i}_"
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        # Log file
        log_file = os.path.join(model_folder, "training_log.txt")
        sys.stdout = open(log_file, "w")
        sys.stderr = sys.stdout  # Ghi cả lỗi vào file log


        trainparam = SimpleNamespace(
            dataset_name="DRIVE",
            learning_rate=LR,
            n_classes=1,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            n_train=n_train,
            n_val=n_val,
            # steps_per_epoch=(n_train + BATCH_SIZE - 1) // BATCH_SIZE,
            # validation_steps=(n_val + BATCH_SIZE - 1) // BATCH_SIZE,
            steps_per_epoch=582 // 65,
            validation_steps=65 // 65,
            input_shape=(None, None, 3),
            save_path=model_folder
        )

        # Đường dẫn tới dữ liệu train
        train_image_dir = f"{BASE_DS}/train/image"
        train_mask_dir = f"{BASE_DS}/train/mask"
        train_dataset = MyDS(image_dir=train_image_dir, mask_dir=train_mask_dir, batch_size=trainparam.batch_size)
        test_image_dir = f"{BASE_DS}/test/image"
        test_mask_dir = f"{BASE_DS}/test/mask"
        test_dataset = MyDS(image_dir=test_image_dir, mask_dir=test_mask_dir, batch_size=trainparam.batch_size)

        # optimizer, loss, metric
        optim = keras.optimizers.Adam(learning_rate = LR)
        dice_loss = sm.losses.DiceLoss()
        focal_loss = sm.losses.BinaryFocalLoss()
        total_loss = focal_loss
        metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]


        # model.load_weights(mdir + 'epoch_40_ver6.weights.h5')

        print(f"\n\n===== TRAINING MODEL {i + 1}/{NUM_MODELS} (Dropout=0) =====\n")

        model_i = unet(input_shape=(128, 128, 3), dropout_rate=0.0)


        # train
        train(model_i,
            train_dataset,
            test_dataset,
            epochs=trainparam.epochs,
            batch_size=trainparam.batch_size,
            learning_rate=trainparam.learning_rate,
            loss_fn=total_loss, metrics=metrics,
            save_path=trainparam.save_path,
            # version=8,
            step_per_epoch=trainparam.steps_per_epoch)