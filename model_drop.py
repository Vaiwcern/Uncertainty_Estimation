import tensorflow as tf
from tensorflow.keras import layers, models

def conv_block(x, filters, dropout_rate=0.5):
    """Khối convolution với Dropout"""
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = layers.Dropout(dropout_rate)(x)
    return x

def upconv_block(x, skip, filters, dropout_rate=0.5):
    """Khối upsampling + skip connection + convolution"""
    x = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.concatenate([x, skip])  # Skip connection từ encoder
    x = conv_block(x, filters, dropout_rate)
    return x

def unet(input_shape=(608, 576, 3), dropout_rate=0.5):
    inputs = layers.Input(input_shape)

    # Encoder (Downsampling)
    c1 = conv_block(inputs, 64, dropout_rate)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 128, dropout_rate)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 256, dropout_rate)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = conv_block(p3, 512, dropout_rate)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = conv_block(p4, 1024, dropout_rate)

    # Decoder (Upsampling)
    u6 = upconv_block(c5, c4, 512, dropout_rate)
    u7 = upconv_block(u6, c3, 256, dropout_rate)
    u8 = upconv_block(u7, c2, 128, dropout_rate)
    u9 = upconv_block(u8, c1, 64, dropout_rate)

    # Output layer
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(u9)

    model = models.Model(inputs, outputs)
    return model
