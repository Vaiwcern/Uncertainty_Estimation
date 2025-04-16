import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tensorflow as tf
import keras
# from tensorflow.keras import mixed_precision

from model.unet import IterativeUnet, VanilaUnet
from training_utils import CustomCallbacks, CustomLosses
from evaluation.metric import IoUMetric, F1ScoreMetric


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
    elif loss_function == 'BCE': 
        loss = CustomLosses.binary_crossentropy_loss()
    elif loss_function == 'dice': 
        loss = CustomLosses.dice_loss()
    elif loss_function == 'dice_bce': 
        loss = CustomLosses.dice_bce_loss()
    elif loss_function == 'dice_focal': 
        loss = CustomLosses.dice_focal_loss()
    else: 
        raise ValueError("Loss function not supported.")

    with strategy.scope():
        if model == "iterative": 
            myModel = IterativeUnet(input_channels=input_channels, dropout_rate=dropout_rate, use_batchnorm=use_batchnorm)
        elif model == "vanila": 
            myModel = VanilaUnet(input_channels=input_channels, dropout_rate=dropout_rate, use_batchnorm=use_batchnorm)
        
        optim = keras.optimizers.Adam(learning_rate=learning_rate)
        # optim = mixed_precision.LossScaleOptimizer(optim)

        myModel.compile(
            optimizer=optim,
            loss=loss,
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name='acc'),
                IoUMetric(threshold=0.5),
                F1ScoreMetric(threshold=0.5),
            ]
        )

    for layer in myModel.layers:
        if isinstance(layer, tf.keras.Model) or isinstance(layer, tf.keras.Sequential):
            for sub_layer in layer.layers:
                if isinstance(sub_layer, tf.keras.layers.BatchNormalization):
                    print(f"Found BN layer: {sub_layer.name}")
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            print(f"Found BN layer: {layer.name}")

    myModel.fit(
        train_dataset,  
        epochs=num_epoch,
        steps_per_epoch=train_dataset_wrapper.steps_per_epoch,
        verbose=1,
        callbacks=[
            # CustomCallbacks.PrintLossCallback(),
            CustomCallbacks.SaveEveryNEpoch(save_path=save_path, interval=save_per_epoch)
        ]
    )