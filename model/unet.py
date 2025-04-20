import tensorflow as tf
from tensorflow.keras import layers

import tensorflow as tf
from tensorflow.keras import layers

class IterativeUnet(tf.keras.Model):
    def __init__(self, input_channels=4, dropout_rate=0.5, use_batchnorm=True):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.use_batchnorm = use_batchnorm 

        # Encoder
        self.conv1 = self._conv_block(64, input_channels=input_channels)
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.conv2 = self._conv_block(128)
        self.pool2 = layers.MaxPooling2D((2, 2))
        self.conv3 = self._conv_block(256)
        self.pool3 = layers.MaxPooling2D((2, 2))
        self.conv4 = self._conv_block(512)
        self.pool4 = layers.MaxPooling2D((2, 2))

        self.bottleneck = self._conv_block(1024)

        # Decoder
        self.up6 = layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')
        self.conv6 = self._conv_block(512)
        self.up7 = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')
        self.conv7 = self._conv_block(256)
        self.up8 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')
        self.conv8 = self._conv_block(128)
        self.up9 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')
        self.conv9 = self._conv_block(64)

        self.output_layer = layers.Conv2D(1, (1, 1), activation=None, dtype='float32')

    def _conv_block(self, filters, input_channels=None):
        layers_list = []
        if input_channels is not None:
            layers_list.append(layers.Conv2D(filters, (3, 3), padding='same', input_shape=(None, None, input_channels)))
        else:
            layers_list.append(layers.Conv2D(filters, (3, 3), padding='same'))

        if self.use_batchnorm:
            layers_list.append(layers.BatchNormalization())
        layers_list.append(layers.Activation('relu'))

        layers_list.append(layers.Conv2D(filters, (3, 3), padding='same'))

        if self.use_batchnorm:
            layers_list.append(layers.BatchNormalization())
        layers_list.append(layers.Activation('relu'))

        if self.dropout_rate > 0:
            layers_list.append(layers.Dropout(self.dropout_rate))

        return tf.keras.Sequential(layers_list)

    def call(self, inputs, training=False):
        c1 = self.conv1(inputs, training=training)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1, training=training)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2, training=training)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3, training=training)
        p4 = self.pool4(c4)
        bn = self.bottleneck(p4, training=training)

        u6 = self.up6(bn)
        u6 = tf.concat([u6, c4], axis=3)
        c6 = self.conv6(u6, training=training)
        u7 = self.up7(c6)
        u7 = tf.concat([u7, c3], axis=3)
        c7 = self.conv7(u7, training=training)
        u8 = self.up8(c7)
        u8 = tf.concat([u8, c2], axis=3)
        c8 = self.conv8(u8, training=training)
        u9 = self.up9(c8)
        u9 = tf.concat([u9, c1], axis=3)
        c9 = self.conv9(u9, training=training)

        return self.output_layer(c9)

    @tf.function(reduce_retracing=True)
    def train_step(self, data):
        x, y = data
        x = tf.cast(x[..., :3], tf.float32)  # Remove feedback channel if present
        zero_channel = tf.zeros_like(x[..., :1])

        def iterative_forward(x, zero_channel, y):
            total_loss = 0.0
            for _ in tf.range(3):
                inputs = tf.concat([x, zero_channel], axis=-1)
                y_pred = self(inputs, training=True)
                loss = self.compute_loss(y=y, y_pred=y_pred)
                total_loss += loss
                zero_channel = y_pred
            return total_loss / 3.0, y_pred

        with tf.GradientTape() as tape:
            loss, y_pred = iterative_forward(x, zero_channel, y)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        for metric in self.metrics:
            metric.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics} | {"loss": loss}

    def build(self, input_shape):
        dummy = tf.zeros(input_shape, dtype=tf.float32)
        self.call(dummy)
        super().build(input_shape)

class VanilaUnet(tf.keras.Model):
    def __init__(self, input_channels=3, dropout_rate=0.5, use_batchnorm=True):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.use_batchnorm = use_batchnorm

        # Encoder
        self.conv1 = self._conv_block(64, input_channels=input_channels)
        self.pool1 = layers.MaxPooling2D((2, 2))

        self.conv2 = self._conv_block(128)
        self.pool2 = layers.MaxPooling2D((2, 2))

        self.conv3 = self._conv_block(256)
        self.pool3 = layers.MaxPooling2D((2, 2))

        self.conv4 = self._conv_block(512)
        self.pool4 = layers.MaxPooling2D((2, 2))

        self.bottleneck = self._conv_block(1024)

        # Decoder
        self.up6 = layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')
        self.conv6 = self._conv_block(512)

        self.up7 = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')
        self.conv7 = self._conv_block(256)

        self.up8 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')
        self.conv8 = self._conv_block(128)

        self.up9 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')
        self.conv9 = self._conv_block(64)

        # Output
        self.output_layer = layers.Conv2D(1, (1, 1), activation=None, dtype='float32')

    def _conv_block(self, filters, input_channels=None):
        layers_list = []

        # Conv2D đầu tiên
        layers_list.append(
            layers.Conv2D(filters, (3, 3), padding='same')
        )

        if self.use_batchnorm:
            layers_list.append(layers.BatchNormalization())

        layers_list.append(layers.Activation('relu'))

        # Conv2D thứ hai
        conv2 = layers.Conv2D(filters, (3, 3), padding='same', name='center_block' if filters == 1024 else None)
        layers_list.append(conv2)

        if self.use_batchnorm:
            layers_list.append(layers.BatchNormalization())

        layers_list.append(layers.Activation('relu'))

        # Dropout (chỉ thêm khi > 0.0)
        if self.dropout_rate > 0.0:
            layers_list.append(layers.Dropout(self.dropout_rate))

        return tf.keras.Sequential(layers_list)

    @tf.function(reduce_retracing=True)
    def call(self, inputs, training=False):
        # Encoder
        c1 = self.conv1(inputs, training=training)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1, training=training)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2, training=training)
        p3 = self.pool3(c3)

        c4 = self.conv4(p3, training=training)
        p4 = self.pool4(c4)

        bn = self.bottleneck(p4, training=training)

        # Decoder
        u6 = self.up6(bn)
        u6 = tf.concat([u6, c4], axis=3)
        c6 = self.conv6(u6, training=training)

        u7 = self.up7(c6)
        u7 = tf.concat([u7, c3], axis=3)
        c7 = self.conv7(u7, training=training)

        u8 = self.up8(c7)
        u8 = tf.concat([u8, c2], axis=3)
        c8 = self.conv8(u8, training=training)

        u9 = self.up9(c8)
        u9 = tf.concat([u9, c1], axis=3)
        c9 = self.conv9(u9, training=training)

        return self.output_layer(c9)

    def train_step(self, data):
        x, y = data
        x = tf.cast(x, tf.float32)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(y=y, y_pred=y_pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        for metric in self.metrics:
            metric.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics} | {"loss": loss}

    def build(self, input_shape):
        dummy = tf.zeros(input_shape, dtype=tf.float32)
        self.call(dummy, training=False)
        super().build(input_shape)
