import tensorflow as tf
import os
import keras
import keras.backend as K
import time
import sys 

class CustomCallbacks:
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
            os.makedirs(save_path, exist_ok=True)

        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % self.interval == 0:
                filename = os.path.join(self.save_path, f"model_epoch_{epoch + 1}.weights.h5")
                self.model.save_weights(filename)
                print(f"\n📦 Saved model to: {filename}")

    class PrettyPrintMetrics(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start_time = time.time()

        def on_epoch_end(self, epoch, logs=None):
            duration = time.time() - self.epoch_start_time
            logs = logs or {}

            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in logs.items()])

            print(f"\n📊 Epoch {epoch + 1} - ⏱️ {duration:.2f}s")
            print(f"   {metrics_str}")
            sys.stdout.flush()


# def convert_model_to_functional(model, input_shape=(1024, 1024, 4)):
#     inputs = keras.Input(shape=input_shape)
#     outputs = model(inputs)
#     return keras.Model(inputs=inputs, outputs=outputs)

class CustomLosses: 
    @staticmethod
    def focal_loss(gamma=2.0, alpha=0.25):
        def loss(y_true, logits):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.sigmoid(logits)
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

            cross_entropy = - (y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
            focal_weight = alpha * tf.pow(1 - y_pred, gamma) * y_true + \
                           (1 - alpha) * tf.pow(y_pred, gamma) * (1 - y_true)

            return tf.reduce_mean(focal_weight * cross_entropy)
        return loss
    
    @staticmethod
    def binary_crossentropy_loss():
        def loss(y_true, logits):
            y_true = tf.cast(y_true, tf.float32)
            bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=logits)
            return tf.reduce_mean(bce)
        return loss

    @staticmethod
    def iou_loss(smooth=1e-6):
        def loss(y_true, logits):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.sigmoid(logits)  # Convert logits to probability

            intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
            union = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3]) - intersection
            iou = (intersection + smooth) / (union + smooth)
            return 1.0 - tf.reduce_mean(iou)
        return loss

    @staticmethod
    def dice_loss(smooth=1e-6):
        def loss(y_true, logits):
            y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
            y_pred_f = tf.keras.backend.flatten(tf.sigmoid(logits))  # Convert logits to probability

            intersection = tf.reduce_sum(y_true_f * y_pred_f)
            dice = (2. * intersection + smooth) / \
                   (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
            return 1.0 - dice
        return loss

    @staticmethod
    def dice_bce_loss(smooth=1e-6):
        def loss(y_true, logits):
            dice = CustomLosses.dice_loss(smooth)(y_true, logits)
            bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(y_true, tf.float32), logits=logits)
            return dice + tf.reduce_mean(bce)
        return loss

    @staticmethod
    def dice_focal_loss(smooth=1e-6, gamma=2.0, alpha=0.25):
        def loss(y_true, logits):
            dice = CustomLosses.dice_loss(smooth)(y_true, logits)
            focal = CustomLosses.focal_loss(gamma, alpha)(y_true, logits)
            return dice + focal
        return loss

