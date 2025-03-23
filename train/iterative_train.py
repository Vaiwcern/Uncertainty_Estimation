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

if __name__ == "__main__":
    # Lấy đối số GPU từ dòng lệnh
    args = parse_args()

    # Set CUDA_VISIBLE_DEVICES để chọn GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    BATCH_SIZE = 2
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

    # data_dir = "/home/ltnghia02/MEDICAL_ITERATIVE/Dataset/RTdata_Crop"
    # train_dataset = RTDataset(data_dir, trainparam.batch_size, normalize=True, train=True, thin_label=False)

    # model = standard_unet(input_size=(None, None, 4), dropout_rate=0.0)

    # optim = keras.optimizers.Adam(learning_rate=trainparam.learning_rate)

    # inference_train(model, train_dataset, epochs=trainparam.epochs, optimizer=optim, loss_fn=focal_loss(), save_path=trainparam.save_path)

    data_dir = "/content/RTdata_Crop"

    train_dataset_wrapper = RTDatasetTF(
        dataset_dir=data_dir,
        batch_size=trainparam.batch_size,
        normalize=True,
        train=True,
        thin_label=False
    )

    train_dataset = train_dataset_wrapper.dataset
    
    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    optim = keras.optimizers.Adam(learning_rate=trainparam.learning_rate)
    with strategy.scope():
        model = StandardUNet(input_channels=4, dropout_rate=0.0)
        model.compile(
            optimizer=optim,
            loss=focal_loss(),
            metrics=[
                'accuracy',
                IoUMetric(),
                F1ScoreMetric()
            ]
        )

    model.fit(train_dataset, epochs=trainparam.epochs)

