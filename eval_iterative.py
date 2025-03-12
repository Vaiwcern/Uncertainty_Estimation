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
import matplotlib.image as mpimg
import pandas as pd
import csv
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, jaccard_score, f1_score, roc_auc_score


import segmentation_models as sm
from seggradcam.dataloaders import Cityscapes, DRIVE
from seggradcam.unet import csbd_unet, manual_unet, TrainUnet
from seggradcam.training_write import TrainingParameters, TrainingResults
from seggradcam.training_plots import plot_predict_and_gt, plot_loss, plot_metric
from seggradcam.seggradcam import SegGradCAM, SuperRoI, ClassRoI, PixelRoI, BiasRoI
from seggradcam.visualize_sgc import SegGradCAMplot


from CustomDataset import DRIVEDataset, CHASEDB1Dataset, STAREDataset
from model import unet

def get_predictions(test_dataset, dataset_name, batch_size): 
    # Lấy một batch ảnh từ test_dataset
    for images, labels in test_dataset:
        break

    # Dự đoán phân đoạn (segmentation) cho batch ảnh
    the_images = tf.unstack(images, axis=0)

    # Thêm kênh toàn 0 vào images (kênh thứ 4)
    zeros_channel = tf.zeros_like(images[:, :, :, :1])  # Tạo kênh thứ 4 toàn 0
    images_with_new_channel = tf.concat([images, zeros_channel], axis=-1)  # Thêm kênh vào cuối cùng

    outputs = []  # Mảng lưu output của 3 lần forward
    gradcams = []
    predictions = images_with_new_channel  # Bắt đầu với input có kênh 0

    # Thêm GradCAM heatmap vào ảnh (the_images)
    for i in range(batch_size):
        # Thêm kênh toàn 0 cho ảnh đầu tiên
        zeros_channel = np.zeros_like(the_images[0][:, :, :1])
        the_images[i] = np.concatenate([the_images[i], zeros_channel], axis=-1)

    # Tiến hành forward 3 lần (Không cần GradientTape trong evaluate)
    for i in range(3):
        prediction = model(predictions, training=False)  # Forward pass (tính toán dự đoán)
        outputs.append(prediction)

        mymaps = []
        for j in range(batch_size):
            # Tính GradCAM heatmap cho ảnh
            prop_from_layer = model.layers[-1].name
            prop_to_layer = 'center_block'
            cls = 0

            clsroi = ClassRoI(model=model, image=the_images[j], cls=cls)
            newsgc = SegGradCAM(model, the_images[j], cls, prop_to_layer, prop_from_layer, roi=clsroi,
                                normalize=True, abs_w=False, posit_w=False)
            mymap = newsgc.SGC()  # Heatmap với shape (H, W)
            mymaps.append(mymap)

        for j in range(batch_size):
            mymap_3d = np.expand_dims(mymaps[j], axis=-1)  # Thêm chiều kênh cho mymap
            the_images[j] = np.concatenate([the_images[j][:, :, :-1], mymap_3d], axis=-1)

        # Chuyển GradCAM heatmap thành tensor và thêm vào ảnh
        mymaps_tensor = np.stack(mymaps, axis=0)  # Gộp mymaps thành tensor có shape (5, H, W)
        mymaps_tensor = np.expand_dims(mymaps_tensor, axis=-1)  # Thêm chiều kênh cho mymaps_tensor
        predictions = np.concatenate([images, mymaps_tensor], axis=-1)  # Nối heatmap vào ảnh
        gradcams.append(mymaps_tensor)

    # Average các output của 3 lần forward
    stacked_outputs = tf.stack(outputs, axis=0)
    # averaged_output = tf.reduce_mean(stacked_outputs, axis=0) # (batch_size, H, W, 1)

    sigmoid_output = stacked_outputs[2]

    # Threshold 0.5 để chuyển thành nhị phân
    binary_output = tf.cast(sigmoid_output > 0.5, tf.float32)

    predictions = tf.cast(binary_output, tf.float32)

    # Hiển thị ảnh gốc, mask, kết quả phân đoạn và các ảnh GradCAM cho mỗi ảnh trong batch
    num_images = len(images)  # Số lượng ảnh trong batch

    # Tạo figure với số lượng subplot tương ứng với số ảnh trong batch
    plt.figure(figsize=(18, 12))  # Điều chỉnh figsize để đủ chỗ hiển thị 6 cột
    for i in range(num_images):
        # Hiển thị ảnh gốc
        plt.subplot(num_images, 6, i * 6 + 1)
        plt.imshow(images[i])
        plt.title("Original Image")
        plt.axis('off')

        # Hiển thị nhãn (mask)
        plt.subplot(num_images, 6, i * 6 + 2)
        plt.imshow(labels[i].squeeze(), cmap='gray')  # Squeeze để bỏ chiều kênh đơn
        plt.title("True Mask")
        plt.axis('off')

        # Hiển thị phân đoạn (output)
        plt.subplot(num_images, 6, i * 6 + 3)
        plt.imshow(tf.squeeze(predictions[i]).numpy(), cmap='gray')
        plt.title("Predicted Segmentation")
        plt.axis('off')

        # Hiển thị GradCAM 1
        plt.subplot(num_images, 6, i * 6 + 4)
        plt.imshow(gradcams[0][i].squeeze(), cmap='jet')
        plt.title("GradCAM 1")
        plt.axis('off')

        # Hiển thị GradCAM 2
        plt.subplot(num_images, 6, i * 6 + 5)
        plt.imshow(gradcams[1][i].squeeze(), cmap='jet')
        plt.title("GradCAM 2")
        plt.axis('off')

        # Hiển thị GradCAM 3
        plt.subplot(num_images, 6, i * 6 + 6)
        plt.imshow(gradcams[2][i].squeeze(), cmap='jet')
        plt.title("GradCAM 3")
        plt.axis('off')


    prediction_path = os.path.join(result_dir, "predictions")
    os.makedirs(prediction_path, exist_ok=True)
    save_path = os.path.join(prediction_path, f"{dataset_name}_prediction.jpg")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)

    # Hiển thị tất cả các ảnh
    # plt.show()
    plt.close()

def evaluate_custom(model, val_dataset, batch_size, step_per_epoch):
    """
    Hàm evaluate mô hình với dữ liệu validation, tính toán accuracy, IoU, F1 score và AUC.

    :param model: Mô hình đã huấn luyện
    :param val_dataset: Dữ liệu validation
    :param batch_size: Kích thước batch cho dữ liệu
    :return: Accuracy, IoU, F1 score và AUC
    """

    total_accuracy = 0
    total_iou = 0
    total_f1 = 0
    total_auc = 0
    num_samples = 0

    # Lặp qua tất cả các batch trong validation dataset
    for step, (images, labels) in enumerate(val_dataset):

        if (step >= step_per_epoch):
            break

        the_images = tf.unstack(images, axis=0)

        # Thêm kênh toàn 0 vào images (kênh thứ 4)
        zeros_channel = tf.zeros_like(images[:, :, :, :1])  # Tạo kênh thứ 4 toàn 0
        images_with_new_channel = tf.concat([images, zeros_channel], axis=-1)  # Thêm kênh vào cuối cùng

        outputs = []  # Mảng lưu output của 3 lần forward
        predictions = images_with_new_channel  # Bắt đầu với input có kênh 0

        # Thêm GradCAM heatmap vào ảnh (the_images)
        for i in range(batch_size):
            # Thêm kênh toàn 0 cho ảnh đầu tiên
            zeros_channel = np.zeros_like(the_images[0][:, :, :1])
            the_images[i] = np.concatenate([the_images[i], zeros_channel], axis=-1)

        # Tiến hành forward 3 lần (Không cần GradientTape trong evaluate)
        for i in range(3):
            prediction = model(predictions, training=False)  # Forward pass (tính toán dự đoán)
            outputs.append(prediction)

            mymaps = []
            for j in range(batch_size):
                # Tính GradCAM heatmap cho ảnh

                prop_from_layer = model.layers[-1].name
                prop_to_layer = 'center_block'
                cls = 0

                clsroi = ClassRoI(model=model, image=the_images[j], cls=cls)
                newsgc = SegGradCAM(model, the_images[j], cls, prop_to_layer, prop_from_layer, roi=clsroi,
                                    normalize=True, abs_w=False, posit_w=False)
                mymap = newsgc.SGC()  # Heatmap với shape (H, W)
                mymaps.append(mymap)

            for j in range(batch_size):
                mymap_3d = np.expand_dims(mymaps[j], axis=-1)  # Thêm chiều kênh cho mymap
                the_images[j] = np.concatenate([the_images[j][:, :, :-1], mymap_3d], axis=-1)

            # Chuyển GradCAM heatmap thành tensor và thêm vào ảnh
            mymaps_tensor = np.stack(mymaps, axis=0)  # Gộp mymaps thành tensor có shape (5, H, W)
            mymaps_tensor = np.expand_dims(mymaps_tensor, axis=-1)  # Thêm chiều kênh cho mymaps_tensor
            predictions = np.concatenate([images, mymaps_tensor], axis=-1)  # Nối heatmap vào ảnh

        # Average các output của 3 lần forward
        stacked_outputs = tf.stack(outputs, axis=0)
        averaged_output = tf.reduce_mean(stacked_outputs, axis=0) # (batch_size, H, W, 1)

        # Áp dụng sigmoid
        # sigmoid_output = tf.sigmoid(averaged_output)

        # Threshold 0.5 để chuyển thành nhị phân
        # labels_tmp = labels
        labels = tf.cast(labels > 0.5, tf.float32)
        binary_output = tf.cast(averaged_output > 0.5, tf.float32)

        print(labels.shape)
        print(binary_output.shape)

        binary_output_flattened = binary_output.numpy().flatten()  # Chuyển đổi tensor thành NumPy array và flatten
        labels_flattened = labels.numpy().flatten()  # Flatten nhãn

        print(labels_flattened)
        print(binary_output_flattened)


        # Tính accuracy, IoU và F1 score
        accuracy = accuracy_score(labels.numpy().flatten(), binary_output.numpy().flatten())
        iou = jaccard_score(labels.numpy().flatten(), binary_output.numpy().flatten())
        # iou = sm.base.functional.iou_score(labels, binary_output, threshold=0.5)
        f1 = f1_score(labels.numpy().flatten(), binary_output.numpy().flatten())

        print(accuracy, iou, f1)

        # Tính AUC, giữ nguyên giá trị sau sigmoid
        auc = roc_auc_score(labels.numpy().flatten(), averaged_output.numpy().flatten())

        # Cộng dồn các metrics
        total_accuracy += accuracy
        total_iou += iou
        total_f1 += f1
        total_auc += auc
        num_samples += 1

    # Tính trung bình các metrics
    avg_accuracy = total_accuracy / num_samples
    avg_iou = total_iou / num_samples
    avg_f1 = total_f1 / num_samples
    avg_auc = total_auc / num_samples

    print(f"Validation Accuracy: {avg_accuracy}")
    print(f"Validation IoU: {avg_iou}")
    print(f"Validation F1 Score: {avg_f1}")
    print(f"Validation AUC: {avg_auc}")

    return avg_accuracy, avg_iou, avg_f1, avg_auc

def crop_image_and_compute_means(arr, crop_size=(304, 288)):
    # Lấy kích thước ảnh
    h, w = arr.shape

    # Tính số lượng crop trên chiều cao và chiều rộng
    crop_h, crop_w = crop_size
    n_crops_h = h // crop_h
    n_crops_w = w // crop_w

    # Khởi tạo danh sách để lưu các giá trị mean của mỗi crop
    means = []

    # Cắt ảnh thành các crop nhỏ và tính mean của mỗi crop
    for i in range(n_crops_h):
        for j in range(n_crops_w):
            crop = arr[i*crop_h:(i+1)*crop_h, j*crop_w:(j+1)*crop_w]
            crop_mean = np.mean(crop)
            means.append(crop_mean)

    # Trả về 4 giá trị mean của 4 crop
    return means

def compute_error_uncertainty(model, image_path, mask_path):
    outputs = []
    gradcams = []

    the_image = mpimg.imread(image_path)
    the_label = mpimg.imread(mask_path)

    image_with_batchsize = np.expand_dims(the_image, axis=0)

    # Thêm kênh toàn 0 vào image (kênh thứ 4)
    zeros_channel = tf.zeros_like(image_with_batchsize[:, :, :, :1])  # Tạo kênh thứ 4 toàn 0
    input = tf.concat([image_with_batchsize, zeros_channel], axis=-1)  # Thêm kênh vào cuối cùng

    # Thêm GradCAM heatmap vào ảnh
    zeros_channel = np.zeros_like(the_image[:, :, :1])
    the_image = np.concatenate([the_image, zeros_channel], axis=-1)

    # Tiến hành forward 3 lần (Không cần GradientTape trong evaluate)
    for i in range(3):
        print("Input shape: ", input.shape)
        prediction = model(input, training=False)  # Forward pass (tính toán dự đoán) shape (batch size=1, H, W, 1)
        print("Prediction shape: ", prediction.shape)
        outputs.append(prediction)

        prop_from_layer = model.layers[-1].name
        prop_to_layer = 'center_block'
        cls = 0

        clsroi = ClassRoI(model=model, image=the_image, cls=cls)
        newsgc = SegGradCAM(model, the_image, cls, prop_to_layer, prop_from_layer, roi=clsroi,
                            normalize=True, abs_w=False, posit_w=False)
        mymap = newsgc.SGC()  # Heatmap với shape (H, W)

        mymap_3d = np.expand_dims(mymap, axis=-1)  # Thêm chiều kênh cho mymap
        the_image = np.concatenate([the_image[:, :, :-1], mymap_3d], axis=-1)

        # Chuyển GradCAM heatmap thành tensor và thêm vào ảnh
        mymaps_tensor = np.expand_dims(mymap, axis=0)
        mymaps_tensor = np.expand_dims(mymaps_tensor, axis=-1)  # Thêm chiều kênh cho mymaps_tensor

        print("Mymap tensor shape: ", mymaps_tensor.shape)
        print("Image with batch size shape: ", image_with_batchsize.shape)

        input = np.concatenate([image_with_batchsize, mymaps_tensor], axis=-1)  # Nối heatmap vào ảnh
        gradcams.append(mymaps_tensor)

    # # Average các output của 3 lần forward
    stacked_outputs = tf.stack(outputs, axis=0)
    stacked_outputs = np.squeeze(stacked_outputs)
    averaged_output = np.mean(stacked_outputs, axis=0)  # (H, W)
    error_map = np.abs(averaged_output - the_label) ** 2

    print("Averaged output shape: ", averaged_output.shape)

    stacked_gradcams = np.stack(gradcams, axis=0)
    stacked_gradcams = np.squeeze(stacked_gradcams)
    stacked_gradcams = np.transpose(stacked_gradcams, (1, 2, 0))
    uncertainty_map = np.std(stacked_gradcams, axis=-1)

    print("Error map shape: ", the_label.shape)
    print("Uncertainty map shape: ", uncertainty_map.shape)

    # crop_size = (152, 144)
    height = 608
    width = 576

    errors = []
    uncertainties = []
    for _ in range(4):
        error = crop_image_and_compute_means(error_map, crop_size=(height, width))
        uncertainty = crop_image_and_compute_means(uncertainty_map, crop_size=(height, width))

        errors.append(error)
        uncertainties.append(uncertainty)

        height //= 2
        width //= 2

    return errors, uncertainties

def evaluate_uncertainty(model, test_path):
    """
    Hàm evaluate mô hình với dữ liệu validation, chỉ tính toán error cho từng ảnh trong batch.
    """

    errors = [[] for _ in range(4)]
    uncertainties = [[] for _ in range(4)]

    for i in range(1, 21):
        id = str(i)
        if len(id) < 2:
                id = "0" + id

        image_path = os.path.join(test_path, "images", f"{id}.png")
        mask_path = os.path.join(test_path, "1st_manual", f"{id}.png")

        error, uncertainty = compute_error_uncertainty(model, image_path, mask_path)

        for j in range(4):
            errors[j] = np.append(errors[j], error[j])
            uncertainties[j] = np.append(uncertainties[j], uncertainty[j])

    return errors, uncertainties

def evaluate_uncertainty_chasedb1(model, test_path):
    """
    Hàm evaluate mô hình với dữ liệu validation, chỉ tính toán error cho từng ảnh trong batch.
    """

    errors = [[] for _ in range(4)]
    uncertainties = [[] for _ in range(4)]

    for i in range(1, 14):
        id = str(i)
        if len(id) < 2:
                id = "0" + id

        image_path = os.path.join(test_path, "images", f"Image_{id}L.jpg")
        mask_path = os.path.join(test_path, "1st_manual", f"Image_{id}L_1stHO.png")

        error, uncertainty = compute_error_uncertainty(model, image_path, mask_path)

        for j in range(4):
            errors[j] = np.append(errors[j], error[j])
            uncertainties[j] = np.append(uncertainties[j], uncertainty[j])

    return errors, uncertainties

def evaluate_uncertainty_stare(model, test_path):
    """
    Hàm evaluate mô hình với dữ liệu validation, chỉ tính toán error cho từng ảnh trong batch.
    """

    errors = [[] for _ in range(4)]
    uncertainties = [[] for _ in range(4)]

    img_dir = os.path.join(test_path, "images")
    mask_dir = os.path.join(test_path, "1st_manual")
    for filename in os.listdir(img_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Lọc các tệp hình ảnh
            image_path = os.path.join(img_dir, filename)
            mask_path = os.path.join(mask_dir, filename.replace('.png', '.ah.png'))

            error, uncertainty = compute_error_uncertainty(model, image_path, mask_path)

            for j in range(4):
                errors[j] = np.append(errors[j], error[j])
                uncertainties[j] = np.append(uncertainties[j], uncertainty[j])

    return errors, uncertainties

def AULC(uncs, error):
    idxs = np.argsort(uncs)
    uncs_s = uncs[idxs]
    error_s = error[idxs]
    mean_error = error_s.mean()
    error_csum = np.cumsum(error_s)
    Fs = error_csum / np.arange(1, len(error_s) + 1)
    Fs = mean_error / (Fs)
    s = 1 / len(Fs)
    return -1 + s * Fs.sum()
def rAULC(uncs, error):
    perf_aulc = AULC(error, error)
    curr_aulc = AULC(uncs, error)
    return curr_aulc / perf_aulc


if __name__ == "__main__":
    BATCH_SIZE = 4
    LR = 1e-3
    EPOCHS = 150
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
        input_shape=(608, 576, 4),
        save_path="/home/ltnghia02/MEDICAL_ITERATIVE/iterative_model/"
    )

    os.makedirs(trainparam.save_path, exist_ok=True)
    log_file = os.path.join(trainparam.save_path, "eval_log.txt")
    sys.stdout = open(log_file, "w")
    sys.stderr = sys.stdout  

    # Đường dẫn tới dữ liệu của bạn
    drive_test_image_dir = "/home/ltnghia02/MEDICAL_ITERATIVE/Dataset/DRIVE/test/images"
    drive_test_mask_dir = "/home/ltnghia02/MEDICAL_ITERATIVE/Dataset/DRIVE/test/1st_manual"
    drive_test_dataset = DRIVEDataset(image_dir=drive_test_image_dir, mask_dir=drive_test_mask_dir, batch_size=trainparam.batch_size, augment=False)

    # Đường dẫn tới dữ liệu của bạn
    chasedb1_test_image_dir = "/home/ltnghia02/MEDICAL_ITERATIVE/Dataset/CHASEDB1/images"
    chasedb1_test_mask_dir = "/home/ltnghia02/MEDICAL_ITERATIVE/Dataset/CHASEDB1/1st_manual"
    chasedb1_test_dataset = CHASEDB1Dataset(image_dir=chasedb1_test_image_dir, mask_dir=chasedb1_test_mask_dir, batch_size=trainparam.batch_size, augment=False)


    # Đường dẫn tới dữ liệu của bạn
    stare_test_image_dir = "/home/ltnghia02/MEDICAL_ITERATIVE/Dataset/STARE/images"
    stare_test_mask_dir = "/home/ltnghia02/MEDICAL_ITERATIVE/Dataset/STARE/1st_manual"
    stare_test_dataset = STAREDataset(image_dir=stare_test_image_dir, mask_dir=stare_test_mask_dir, batch_size=trainparam.batch_size, augment=False)

    # Đường dẫn tới dữ liệu của bạn
    stare_rotate_test_image_dir = "/home/ltnghia02/MEDICAL_ITERATIVE/Dataset/STARE_Rotate/images"
    stare_rotate_test_mask_dir = "/home/ltnghia02/MEDICAL_ITERATIVE/Dataset/STARE_Rotate/1st_manual"
    stare_rotate_test_dataset = STAREDataset(image_dir=stare_rotate_test_image_dir, mask_dir=stare_rotate_test_mask_dir, batch_size=trainparam.batch_size, augment=False)


    for i in range(20, 521, 20): 
        model_name = 'epoch_' + str(i)+ '_ver1.weights.h5'
        mdir = trainparam.save_path
        model = unet()
        model.load_weights(mdir + model_name)
        result_dir = mdir + 'results/' + model_name
        os.makedirs(result_dir, exist_ok=True)

        get_predictions(drive_test_dataset, "DRIVE", trainparam.batch_size)
        get_predictions(chasedb1_test_dataset, "CHASEDB1", trainparam.batch_size)
        get_predictions(stare_test_dataset, "STARE", trainparam.batch_size)
        get_predictions(stare_rotate_test_dataset, "STARE_Rotate", trainparam.batch_size)

        # Initialize variables for each dataset evaluation
        accuracy_1, iou_1, f1_1, auc_1 = evaluate_custom(model, drive_test_dataset, trainparam.batch_size, trainparam.validation_steps)
        accuracy_2, iou_2, f1_2, auc_2 = evaluate_custom(model, chasedb1_test_dataset, trainparam.batch_size, trainparam.validation_steps)
        accuracy_3, iou_3, f1_3, auc_3 = evaluate_custom(model, stare_test_dataset, trainparam.batch_size, trainparam.validation_steps)
        accuracy_4, iou_4, f1_4, auc_4 = evaluate_custom(model, stare_rotate_test_dataset, trainparam.batch_size, trainparam.validation_steps)

        # Create a DataFrame with 4 rows corresponding to 4 datasets
        df = pd.DataFrame([
            [accuracy_1, iou_1, f1_1, auc_1],
            [accuracy_2, iou_2, f1_2, auc_2],
            [accuracy_3, iou_3, f1_3, auc_3],
            [accuracy_4, iou_4, f1_4, auc_4]
        ], columns=["accuracy", "iou", "f1", "auc"], index=["drive_test", "chasedb1_test", "stare_test", "stare_rotate_test"])

        # Save the DataFrame to a CSV file
        save_path = os.path.join(result_dir, "segment_eval.csv")
        df.to_csv(save_path)

        test_path = "/home/ltnghia02/MEDICAL_ITERATIVE/Dataset/DRIVE/test/"
        drive_errors, drive_uncertainties = evaluate_uncertainty(model, test_path)
        
        path = "/home/ltnghia02/MEDICAL_ITERATIVE/Dataset/CHASEDB1"
        chasedb1_errors, chasedb1_uncertainties = evaluate_uncertainty_chasedb1(model, path)
        
        path = "/home/ltnghia02/MEDICAL_ITERATIVE/Dataset/STARE_Rotate"
        stare_rotate_errors, stare_rotate_uncertainties = evaluate_uncertainty_stare(model, path)

        path = "/home/ltnghia02/MEDICAL_ITERATIVE/Dataset/STARE"
        stare_errors, stare_uncertainties = evaluate_uncertainty_stare(model, path)

        # Define the result path
        result_path = os.path.join(result_dir, "UE_eval.csv")

        # Calculate Pearson Correlation Coefficients for different datasets
        corr_drive = [pearsonr(drive_errors[i], drive_uncertainties[i])[0] for i in range(4)]
        corr_chasedb1 = [pearsonr(chasedb1_errors[i], chasedb1_uncertainties[i])[0] for i in range(4)]
        corr_stare = [pearsonr(stare_errors[i], stare_uncertainties[i])[0] for i in range(4)]
        corr_rotate_stare = [pearsonr(stare_rotate_errors[i], stare_rotate_uncertainties[i])[0] for i in range(4)]

        # Calculate rAULC for different datasets
        rAULC_drive = [rAULC(drive_uncertainties[i]**0.5, drive_errors[i]) for i in range(4)]
        rAULC_chasedb1 = [rAULC(chasedb1_uncertainties[i]**0.5, chasedb1_errors[i]) for i in range(4)]
        rAULC_stare = [rAULC(stare_uncertainties[i]**0.5, stare_errors[i]) for i in range(4)]
        rAULC_rotate_stare = [rAULC(stare_rotate_uncertainties[i]**0.5, stare_rotate_errors[i]) for i in range(4)]

        # Define the column headers for the different image sizes with Pearson and rAULC next to each other
        column_headers = ["Dataset", 
                        "Pearson_608x576", "rAULC_608x576", 
                        "Pearson_304x288", "rAULC_304x288", 
                        "Pearson_152x144", "rAULC_152x144", 
                        "Pearson_76x72", "rAULC_76x72"]

        # Prepare data to write, with Pearson and rAULC side by side
        data = [
            ["Drive"] + corr_drive + rAULC_drive,
            ["Chasedb1"] + corr_chasedb1 + rAULC_chasedb1,
            ["Stare"] + corr_stare + rAULC_stare,
            ["Stare_rotate"] + corr_rotate_stare + rAULC_rotate_stare
        ]

        # Write results to the result file
        with open(result_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(column_headers)  # Write header row
            writer.writerows(data)  # Write the data rows

        print(f"Results saved to {result_path}")
