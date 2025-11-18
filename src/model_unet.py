%%writefile /content/drive/MyDrive/UNet-LIDC-Segmentation/src/model_unet.py
import tensorflow as tf
from tensorflow.keras import layers, models


def conv_block(x, filters):
    x = layers.Conv2D(filters, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(filters, (3, 3), activation="relu", padding="same")(x)
    return x


def build_unet(input_shape=(256, 256, 1)):
    """
    2D U-Net architecture used in the manuscript.
    """
    inputs = layers.Input(input_shape)

    # Encoder
    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 128)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = conv_block(p3, 256)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    bn = conv_block(p4, 512)

    # Decoder
    u5 = layers.UpSampling2D((2, 2))(bn)
    u5 = layers.Concatenate()([u5, c4])
    c5 = conv_block(u5, 256)

    u6 = layers.UpSampling2D((2, 2))(c5)
    u6 = layers.Concatenate()([u6, c3])
    c6 = conv_block(u6, 128)

    u7 = layers.UpSampling2D((2, 2))(c6)
    u7 = layers.Concatenate()([u7, c2])
    c7 = conv_block(u7, 64)

    u8 = layers.UpSampling2D((2, 2))(c7)
    u8 = layers.Concatenate()([u8, c1])
    c8 = conv_block(u8, 32)

    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(c8)

    model = models.Model(inputs=[inputs], outputs=[outputs], name="unet_lidc_2d")
    return model
