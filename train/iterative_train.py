# import os
# import sys
# import argparse

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import numpy as np
# import matplotlib.pyplot as plt
# from types import SimpleNamespace
# import keras
# from CustomDataset.CustomDataset import RTDataset
# from model.unet import standard_unet
# from utils import focal_loss, inference_train

# def parse_args():
#     # Tạo đối số dòng lệnh để nhận GPU
#     parser = argparse.ArgumentParser(description='Train Unet model on specific GPUs.')
#     parser.add_argument('--gpus', type=str, required=True, help='Comma separated list of GPU IDs to use (e.g., "5,6").')
#     return parser.parse_args()

# if __name__ == "__main__":
#     # Lấy đối số GPU từ dòng lệnh
#     args = parse_args()

#     # Set CUDA_VISIBLE_DEVICES để chọn GPU
#     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

#     BATCH_SIZE = 3
#     LR = 1e-3
#     EPOCHS = 10

#     # Ensure steps_per_epoch and validation_steps are integers
#     trainparam = SimpleNamespace(
#         dataset_name="RoadTracer",
#         learning_rate=LR,
#         n_classes=1,
#         epochs=EPOCHS,
#         batch_size=BATCH_SIZE,
#         input_shape=(1024, 1024, 4),
#         save_path="/home/ltnghia02/MEDICAL_ITERATIVE/model/RTdata_iterative_model"
#     )

#     data_dir = "/home/ltnghia02/MEDICAL_ITERATIVE/Dataset/RTdata_Crop"
#     train_dataset = RTDataset(data_dir, trainparam.batch_size, normalize=True, train=True, thin_label=False)

#     model = standard_unet(input_size=(None, None, 4), dropout_rate=0.0)

#     optim = keras.optimizers.Adam(learning_rate=trainparam.learning_rate)

#     inference_train(model, train_dataset, epochs=10, optimizer=optim, loss_fn=focal_loss())

import tensorflow as tf
import os

# Kiểm tra phiên bản TensorFlow
tensorflow_version = tf.__version__
print(f"TensorFlow Version: {tensorflow_version}")

# Kiểm tra xem TensorFlow có GPU hỗ trợ không
gpu_available = tf.config.list_physical_devices('GPU')
if gpu_available:
    print("TensorFlow is using GPU.")
else:
    print("TensorFlow is using CPU.")

# Kiểm tra phiên bản CUDA
try:
    cuda_version = os.popen("nvcc --version").read()
    cuda_version = cuda_version.split("release")[1].split(',')[0].strip() if 'release' in cuda_version else 'CUDA not installed'
    print(f"CUDA Version: {cuda_version}")
except Exception as e:
    print(f"Error in retrieving CUDA version: {e}")

# Kiểm tra phiên bản cuDNN (sử dụng file cudnn.h)
try:
    cudnn_version = os.popen('cat /usr/include/cudnn.h | grep CUDNN_MAJOR -A 2').read()
    if cudnn_version:
        cudnn_version = cudnn_version.split('\n')[0].split(' ')[-1]
        print(f"cuDNN Version: {cudnn_version}")
    else:
        print("cuDNN not installed or not found.")
except Exception as e:
    print(f"Error in retrieving cuDNN version: {e}")
