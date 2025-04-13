import tensorflow as tf
import os
import keras
import keras.backend as K

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


def convert_model_to_functional(model, input_shape=(1024, 1024, 4)):
    inputs = keras.Input(shape=input_shape)
    outputs = model(inputs)
    return keras.Model(inputs=inputs, outputs=outputs)

class CustomLosses:
    @staticmethod
    def focal_loss(gamma=2.0, alpha=0.25):
        def loss(y_true, y_pred):
            epsilon = K.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

            pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
            focal_weight = alpha * tf.pow((1 - pt), gamma)

            return -tf.reduce_mean(focal_weight * tf.math.log(pt))
        return loss

