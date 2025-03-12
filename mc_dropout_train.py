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


import segmentation_models as sm
from seggradcam.dataloaders import Cityscapes, DRIVE
from seggradcam.unet import csbd_unet, manual_unet, TrainUnet
from seggradcam.training_write import TrainingParameters, TrainingResults
from seggradcam.training_plots import plot_predict_and_gt, plot_loss, plot_metric
from seggradcam.seggradcam import SegGradCAM, SuperRoI, ClassRoI, PixelRoI, BiasRoI
from seggradcam.visualize_sgc import SegGradCAMplot

from model import dropout_unet
from DRIVEDataset import DRIVEDataset

def train(model, train_dataset, val_dataset, epochs, learning_rate, loss_fn, metrics, save_path, version=1, step_per_epoch=None):
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

        if (epoch + 1) % 20 == 0:  # Lưu model mỗi 5 epoch
            model.save_weights(os.path.join(save_path, 'epoch_' + str(epoch+1) + '_ver' + str(version) + '.weights.h5'))
            print("Model saved!")

if __name__ == "__main__":
    BATCH_SIZE = 4
    LR = 1e-3
    EPOCHS = 600
    n_train = 20
    n_val = 20

    # Ensure steps_per_epoch and validation_steps are integers
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
        input_shape=(None, None, 3),
        save_path="/home/ltnghia02/MEDICAL_ITERATIVE/mc_dropout_model"
    )
    
    os.makedirs(trainparam.save_path, exist_ok=True)
    # Log file
    log_file = os.path.join(trainparam.save_path, "training_log.txt")
    sys.stdout = open(log_file, "w")
    sys.stderr = sys.stdout  # Ghi cả lỗi vào file log


    # Đường dẫn tới dữ liệu train
    train_image_dir = "/home/ltnghia02/MEDICAL_ITERATIVE/Dataset/DRIVE/training/images"
    train_mask_dir = "/home/ltnghia02/MEDICAL_ITERATIVE/Dataset/DRIVE/training/1st_manual"
    train_dataset = DRIVEDataset(image_dir=train_image_dir, mask_dir=train_mask_dir, batch_size=trainparam.batch_size)

    # Đường dẫn tới dữ liệu test
    test_image_dir = "/home/ltnghia02/MEDICAL_ITERATIVE/Dataset/DRIVE/test/images"
    test_mask_dir = "/home/ltnghia02/MEDICAL_ITERATIVE/Dataset/DRIVE/test/1st_manual"

    test_dataset = DRIVEDataset(image_dir=test_image_dir, mask_dir=test_mask_dir, batch_size=trainparam.batch_size, augment=False)

    # optimizer, loss, metric
    optim = keras.optimizers.Adam(learning_rate = LR)
    # dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.BinaryFocalLoss()
    total_loss = focal_loss
    # binary_crossentropy_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    
    model = dropout_unet()

    train(model,
        train_dataset,
        test_dataset,
        epochs=trainparam.epochs,
        learning_rate=trainparam.learning_rate,
        loss_fn=total_loss, metrics=metrics,
        save_path = trainparam.save_path,
        version=1,
        step_per_epoch=trainparam.steps_per_epoch)