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
            filename = os.path.join(self.save_path, f"model_epoch_{epoch + 1}.h5")
            self.model.save(filename)
            print(f"\n📦 Saved model to: {filename}")


if __name__ == "__main__":
    # Lấy đối số GPU từ dòng lệnh
    args = parse_args()

    # Set CUDA_VISIBLE_DEVICES để chọn GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    BATCH_SIZE = 12
    LR = 1e-3
    EPOCHS = 10

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

    train_dataset_wrapper = RTDatasetTF(
        dataset_dir="/home/ltnghia02/MEDICAL_ITERATIVE/Dataset/RTdata_Crop",
        batch_size=trainparam.batch_size,
        normalize=True,
        train=True,
        thin_label=False
    )

    train_dataset = train_dataset_wrapper.dataset

    print("Total images:", len(train_dataset_wrapper.image_files))
    print("Steps per epoch:", train_dataset_wrapper.steps_per_epoch)

    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        model = StandardUNet(input_channels=4, dropout_rate=0.0)
        optim = keras.optimizers.Adam(learning_rate=trainparam.learning_rate)
        model.compile(
            optimizer=optim,
            loss=focal_loss(),
            metrics=[
                'accuracy',
                IoUMetric(),
                F1ScoreMetric()
            ]
        )
        
        model.build(input_shape=(None, 1024, 1024, 4))

        dummy_x = tf.random.normal((1, 1024, 1024, 4))
        _ = model(dummy_x)


    model.fit(
        train_dataset,  
        epochs=trainparam.epochs,
        steps_per_epoch=train_dataset_wrapper.steps_per_epoch,
        callbacks=[
            PrintLossCallback(),
            SaveEveryNEpoch(save_path=trainparam.save_path, interval=1)
        ]
    )

