import tensorflow as tf
from tensorflow.keras import layers

class StandardUNet(tf.keras.Model):
    def __init__(self, input_channels=4, dropout_rate=0.5):
        super(StandardUNet, self).__init__()
        self.dropout_rate = dropout_rate

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
        self.output_layer = layers.Conv2D(1, (1, 1), activation='sigmoid')

    def _conv_block(self, filters, input_channels=None):
        layers_list = []

        if input_channels is not None:
            layers_list.append(layers.Conv2D(filters, (3, 3), padding='same', input_shape=(None, None, input_channels)))
        else:
            layers_list.append(layers.Conv2D(filters, (3, 3), padding='same'))

        layers_list += [
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(filters, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(self.dropout_rate)
        ]

        return tf.keras.Sequential(layers_list)

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
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        # zero_channel = np.zeros((x.shape[0], x.shape[1], x.shape[2], 1))
        zero_channel = tf.zeros_like(x[..., :1])  # Cùng shape với 1 channel

        total_loss = 0.0

        with tf.GradientTape() as tape:
            for _ in range(3):
                # images_4ch = np.concatenate([x, zero_channel], axis=-1)
                images_4ch = tf.concat([x, zero_channel], axis=-1)
                print("HEHE", images_4ch.shape)
                y_pred = self(images_4ch, training=True)
                
                loss = self.compute_loss(y=y, y_pred=y_pred)
                
                # Cộng dồn loss
                total_loss += loss

                # Sử dụng output làm channel thứ 4 cho loop tiếp theo
                zero_channel = y_pred

            loss = total_loss / 3

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply(gradients, trainable_vars)

        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    
    def build(self, input_shape):
        # Tạo dummy input để kích hoạt các layers và đánh dấu model đã build
        dummy_input = tf.keras.Input(shape=input_shape[1:])
        _ = self.call(dummy_input, training=False)

        super().build(input_shape)


def unet(input_shape=(608, 576, 4), n_classes=1):
    """
    Defines a standard U-Net model for image segmentation.

    :param input_shape: Shape of the input image (height, width, channels)
    :param n_classes: Number of output classes (1 for binary segmentation)

    :return: Keras model object
    """

    inputs = layers.Input(shape=input_shape)

    # Contracting path (Encoder)
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)

    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)

    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)

    conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPooling2D((2, 2))(conv4)

    conv5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same', name='center_block')(conv5)

    # Expansive path (Decoder)
    up6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5)
    concat6 = layers.concatenate([up6, conv4], axis=3)
    conv6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(concat6)
    conv6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
    concat7 = layers.concatenate([up7, conv3], axis=3)
    conv7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(concat7)
    conv7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
    concat8 = layers.concatenate([up8, conv2], axis=3)
    conv8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(concat8)
    conv8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
    concat9 = layers.concatenate([up9, conv1], axis=3)
    conv9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(concat9)
    conv9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    # Output layer
    outputs = layers.Conv2D(n_classes, (1, 1), activation='sigmoid')(conv9)

    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

def dropout_unet(input_size=(608, 576, 3)):
    inputs = layers.Input(input_size)

    # Downsampling path (Encoder)
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = layers.Dropout(0.5)(conv1)  # Added Dropout
    pool1 = layers.MaxPooling2D((2, 2))(conv1)

    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = layers.Dropout(0.5)(conv2)  # Added Dropout
    pool2 = layers.MaxPooling2D((2, 2))(conv2)

    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = layers.Dropout(0.5)(conv3)  # Added Dropout
    pool3 = layers.MaxPooling2D((2, 2))(conv3)

    conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = layers.Dropout(0.5)(conv4)  # Added Dropout
    pool4 = layers.MaxPooling2D((2, 2))(conv4)

    # Bottleneck
    conv5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = layers.Dropout(0.5)(conv5)  # Added Dropout

    # Upsampling path (Decoder)
    up6 = layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(conv5)
    concat6 = layers.concatenate([up6, conv4], axis=3)
    conv6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(concat6)
    conv6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = layers.Dropout(0.5)(conv6)  # Added Dropout

    up7 = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(conv6)
    concat7 = layers.concatenate([up7, conv3], axis=3)
    conv7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(concat7)
    conv7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = layers.Dropout(0.5)(conv7)  # Added Dropout

    up8 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(conv7)
    concat8 = layers.concatenate([up8, conv2], axis=3)
    conv8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(concat8)
    conv8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = layers.Dropout(0.5)(conv8)  # Added Dropout

    up9 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(conv8)
    concat9 = layers.concatenate([up9, conv1], axis=3)
    conv9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(concat9)
    conv9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = layers.Dropout(0.5)(conv9)  # Added Dropout

    # Output layer
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = models.Model(inputs=[inputs], outputs=[outputs])

    return model