import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tensorflow as tf
from datetime import datetime
import keras

from model.unet import IterativeUnet, VanilaUnet
from training_utils import CustomCallbacks, CustomLosses

def train(
    model: str,
    train_dataset_wrapper: tf.data.Dataset,
    input_channels: int,
    num_epoch: int,
    batch_size: int,
    save_path: str,
    loss_function: str,
    use_batchnorm: bool = False,
    dropout_rate: float = False,
    learning_rate: float = 0.001,
    save_per_epoch: int = 5,
    devices: str = None,
) -> None:

    # Log setting
    os.makedirs(save_path, exist_ok=True)
    setting_log_path = os.path.join(save_path, "setting.log")
    with open(setting_log_path, "a") as f:
        f.write("=== Training Configuration ===\n")
        for key, value in locals().items():
            if key != "train_dataset_wrapper":  # tránh in dataset dài
                f.write(f"{key}: {value}\n")
        f.write("\n")

    # Log training process
    timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    train_log_dir = os.path.join(save_path, "train_logs")
    os.makedirs(train_log_dir, exist_ok=True)

    log_file_path = os.path.join(train_log_dir, f"training__{timestamp}.log")

    log_file = open(log_file_path, "w")
    sys.stdout = log_file
    sys.stderr = log_file

    # Train
    train_dataset = train_dataset_wrapper.dataset
    print("Total images:", len(train_dataset_wrapper.image_files))
    print("Steps per epoch:", train_dataset_wrapper.steps_per_epoch)

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    if loss_function == 'focal': 
        loss = CustomLosses.focal_loss()

    with strategy.scope():
        if model == "iterative": 
            myModel = IterativeUnet(input_channels=input_channels, dropout_rate=dropout_rate, use_batchnorm=use_batchnorm)
        elif model == "vanila": 
            myModel = VanilaUnet(input_channels=input_channels, dropout_rate=dropout_rate, use_batchnorm=use_batchnorm)
        optim = keras.optimizers.Adam(learning_rate=learning_rate)
        myModel.compile(
            optimizer=optim,
            loss=loss,
            metrics=[
                'accuracy',
            ]
        )

    myModel.fit(
        train_dataset,  
        epochs=num_epoch,
        steps_per_epoch=train_dataset_wrapper.steps_per_epoch,
        callbacks=[
            CustomCallbacks.PrintLossCallback(),
            CustomCallbacks.SaveEveryNEpoch(save_path=save_path, interval=save_per_epoch)
        ]
    )