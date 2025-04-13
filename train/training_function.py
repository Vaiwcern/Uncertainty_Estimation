import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tensorflow as tf
import keras

from model.unet import IterativeUnet, VanilaUnet
from training_utils import CustomCallbacks, CustomLosses

def train(
    model: str,
    train_dataset_wrapper: tf.data.Dataset,
    input_channels: int,
    num_epoch: int,
    save_path: str,
    loss_function: str,
    use_batchnorm: bool = False,
    dropout_rate: float = False,
    learning_rate: float = 0.001,
    save_per_epoch: int = 5,
) -> None:
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

    # if model == "iterative": 
    #     myModel = IterativeUnet(input_channels=input_channels, dropout_rate=dropout_rate, use_batchnorm=use_batchnorm)
    # elif model == "vanila": 
    #     myModel = VanilaUnet(input_channels=input_channels, dropout_rate=dropout_rate, use_batchnorm=use_batchnorm)
    # for layer in myModel.layers:
    #     print(layer.name, layer.dtype, layer.compute_dtype)

    myModel.fit(
        train_dataset,  
        epochs=num_epoch,
        steps_per_epoch=train_dataset_wrapper.steps_per_epoch,
        callbacks=[
            CustomCallbacks.PrintLossCallback(),
            CustomCallbacks.SaveEveryNEpoch(save_path=save_path, interval=save_per_epoch)
        ]
    )