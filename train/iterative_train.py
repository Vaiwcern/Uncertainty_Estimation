import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from types import SimpleNamespace
import keras
from CustomDataset.CustomDataset import RTDatasetTF
from model.unet import StandardUNet
from utils import focal_loss, IoUMetric, F1ScoreMetric
import tensorflow as tf
import numpy as np
import imageio
from tqdm import tqdm



def parse_args():
    # Tạo đối số dòng lệnh để nhận GPU
    parser = argparse.ArgumentParser(description='Train Unet model on specific GPUs.')
    parser.add_argument('--gpus', type=str, required=True, help='Comma separated list of GPU IDs to use (e.g., "5,6").')
    return parser.parse_args()

class PrintLossCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get("loss")
        if loss is not None:
            print(f"\n✅ Epoch {epoch + 1}: Average Loss = {loss:.4f}")

class SaveEveryNEpoch(tf.keras.callbacks.Callback):
    def __init__(self, save_path, interval=5):
        super().__init__()
        self.save_path = save_path
        self.interval = interval

        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(save_path, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0:
            filename = os.path.join(self.save_path, f"model_epoch_{epoch + 1}.weights.h5")
            self.model.save_weights(filename)
            print(f"\n📦 Saved model to: {filename}")


def predict_and_save(model, dataset, image_files, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    idx = 0  # Chỉ số của ảnh gốc

    for (x, _) in tqdm(dataset, desc="Predicting"):
        x_orig = x[..., :3].numpy()  # Bỏ channel thứ 4, về numpy
        zero_channel = tf.zeros_like(x[..., :1])

        outputs = []

        for i in range(3):
            x_4ch = tf.concat([x[..., :3], zero_channel], axis=-1)
            y_pred = model(x_4ch, training=False)
            outputs.append(y_pred)
            zero_channel = y_pred

        batch_size = x.shape[0]

        for b in range(batch_size):
            if idx >= len(image_files):
                break

            image_name = os.path.basename(image_files[idx])
            name_without_ext = os.path.splitext(image_name)[0]

            # Lưu ảnh gốc (convert về [0,255])
            ori_image = (x_orig[b] * 255).astype("uint8")
            ori_path = os.path.join(save_dir, f"{name_without_ext}_input.png")
            imageio.imwrite(ori_path, ori_image)

            # Lưu từng output mà không threshold
            for i in range(3):
                pred = outputs[i][b]  # (H, W, 1)
                pred = tf.squeeze(pred, axis=-1)
                pred = (pred * 255).numpy().astype("uint8")  # scale lên [0, 255] mà không threshold
                output_path = os.path.join(save_dir, f"{name_without_ext}_output_{i}.png")
                imageio.imwrite(output_path, pred)

            idx += 1


if __name__ == "__main__":
    # Lấy đối số GPU từ dòng lệnh
    args = parse_args()

    # Set CUDA_VISIBLE_DEVICES để chọn GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    BATCH_SIZE = 12
    LR = 1e-3
    EPOCHS = 100

    # Ensure steps_per_epoch and validation_steps are integers
    trainparam = SimpleNamespace(
        dataset_name="RoadTracer",
        learning_rate=LR,
        n_classes=1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        input_shape=(1024, 1024, 4),
        save_path="/home/ltnghia02/MEDICAL_ITERATIVE/model/RTdata_iterative_model"
    )

    # Redirect stdout và stderr để log ra file
    log_file_path = os.path.join(trainparam.save_path, "log.txt")
    os.makedirs(trainparam.save_path, exist_ok=True)  # Đảm bảo thư mục tồn tại

    sys.stdout = open(log_file_path, "w")
    sys.stderr = sys.stdout

    # TRAIN
    # train_dataset_wrapper = RTDatasetTF(
    #     dataset_dir="/home/ltnghia02/MEDICAL_ITERATIVE/Dataset/RTdata_Crop",
    #     batch_size=trainparam.batch_size,
    #     normalize=True,
    #     train=True,
    #     thin_label=False
    # )

    # train_dataset = train_dataset_wrapper.dataset

    # print("Total images:", len(train_dataset_wrapper.image_files))
    # print("Steps per epoch:", train_dataset_wrapper.steps_per_epoch)

    # # Create a MirroredStrategy.
    # strategy = tf.distribute.MirroredStrategy()
    # print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # with strategy.scope():
    #     model = StandardUNet(input_channels=4, dropout_rate=0.0)
    #     optim = keras.optimizers.Adam(learning_rate=trainparam.learning_rate)
    #     model.compile(
    #         optimizer=optim,
    #         loss=focal_loss(),
    #         metrics=[
    #             'accuracy',
    #             IoUMetric(),
    #             F1ScoreMetric()
    #         ]
    #     )
        
    #     model.build(input_shape=(None, 1024, 1024, 4))

    #     dummy_x = tf.random.normal((1, 1024, 1024, 4))
    #     _ = model(dummy_x)


    # model.fit(
    #     train_dataset,  
    #     epochs=trainparam.epochs,
    #     steps_per_epoch=train_dataset_wrapper.steps_per_epoch,
    #     callbacks=[
    #         PrintLossCallback(),
    #         SaveEveryNEpoch(save_path=trainparam.save_path, interval=1)
    #     ]
    # )

    # PREDICT 
    test_dataset_wrapper = RTDatasetTF(
        dataset_dir="/home/ltnghia02/MEDICAL_ITERATIVE/Dataset/RTdata_Crop",
        batch_size=trainparam.batch_size,
        normalize=True,
        train=False,
        thin_label=False
    )

    test_dataset = test_dataset_wrapper.dataset
    image_files = [str(p) for p in test_dataset_wrapper.image_files]

    print("Total images:", len(test_dataset_wrapper.image_files))
    print("Steps per epoch:", test_dataset_wrapper.steps_per_epoch)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = StandardUNet(input_channels=4, dropout_rate=0.0)
        model.build((None, 1024, 1024, 4))
        model.load_weights(os.path.join(trainparam.save_path, "model_epoch_10.weights.h5"))

    predict_and_save(model, dataset=test_dataset, image_files=image_files, save_dir=os.path.join(trainparam.save_path, "predict"))
