"""
Reference implementation of the 2D U-Net pipeline reported in

    Marino C. Automated segmentation of lung nodules in CT scans using a U-Net
    deep learning model: a study based on the LIDC-IDRI dataset.

Every definition in this module is reproduced verbatim from
notebooks/02_UNet_Training.ipynb, which remains the implementation of record:
the reported models were trained by executing that notebook (and its
seed-sensitivity counterpart 02B) rather than by importing this file. The module
exists so that the architecture, the loss and the data sequences can be read,
imported and checked without opening a notebook.

Configuration of the definitive runs, as reported in the manuscript:

    input                256 x 256 x 1
    encoder filters      32, 64, 128, 256   bottleneck 512
    parameters           7,768,353 total, 7,762,465 trainable
    loss                 focal Tversky, alpha = 0.30, beta = 0.70, gamma = 0.75
    optimiser            Adam, initial learning rate 1e-4
    batch size           8   (6 nodule-positive + 2 nodule-negative)
    steps per epoch      500
    early stopping       patience 8 on validation foreground Dice
    ReduceLROnPlateau    factor 0.5, patience 3, min_lr 1e-6
    augmentation         none
    seeds                42 (reference), 1, 2

Self-check:

    python src/unet_pipeline.py        # builds the network and verifies the
                                       # parameter count against the manuscript
"""

import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
CHANNELS = 1
BATCH_SIZE = 8
TRAIN_POSITIVE_FRACTION = 0.75
TRAIN_STEPS_PER_EPOCH = 500
SMOOTH = 1e-6
PREDICTION_THRESHOLD = 0.50



# ==========================================================================
# Balanced training sequence (verbatim from 02_UNet_Training.ipynb)
# ==========================================================================

class BalancedSegmentationSequence(tf.keras.utils.Sequence):

    def __init__(
        self,
        positive_df,
        negative_df,
        batch_size=8,
        positive_fraction=0.50,
        steps_per_epoch=400,
        seed=42,
        shuffle=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.positive_df = positive_df.reset_index(drop=True)
        self.negative_df = negative_df.reset_index(drop=True)

        self.batch_size = int(batch_size)
        self.positive_fraction = float(positive_fraction)
        self.steps_per_epoch = int(steps_per_epoch)
        self.shuffle = bool(shuffle)

        self.rng = np.random.default_rng(seed)

        self.n_positive = int(
            round(self.batch_size * self.positive_fraction)
        )

        self.n_negative = (
            self.batch_size - self.n_positive
        )

        if self.n_positive < 1:
            raise ValueError(
                "Each batch must contain at least one positive slice."
            )

        if self.n_negative < 1:
            raise ValueError(
                "Each batch must contain at least one negative slice."
            )

        print(
            f"Balanced generator: "
            f"{self.n_positive} positive + "
            f"{self.n_negative} negative per batch"
        )

    def __len__(self):
        return self.steps_per_epoch

    def _load_row(self, row):
        npz_path = row["npz_path"]
        slice_index = int(row["slice_index"])

        with np.load(npz_path) as data:
            image = data["volume_normalized"][
                :, :, slice_index
            ]

            mask = data["consensus_mask"][
                :, :, slice_index
            ]

        image = image.astype(np.float32)[..., None]
        mask = mask.astype(np.float32)[..., None]

        return image, mask

    def __getitem__(self, index):

        positive_indices = self.rng.integers(
            low=0,
            high=len(self.positive_df),
            size=self.n_positive,
        )

        negative_indices = self.rng.integers(
            low=0,
            high=len(self.negative_df),
            size=self.n_negative,
        )

        selected_rows = []

        for idx in positive_indices:
            selected_rows.append(
                self.positive_df.iloc[int(idx)]
            )

        for idx in negative_indices:
            selected_rows.append(
                self.negative_df.iloc[int(idx)]
            )

        if self.shuffle:
            self.rng.shuffle(selected_rows)

        images = []
        masks = []

        for row in selected_rows:
            image, mask = self._load_row(row)

            images.append(image)
            masks.append(mask)

        X = np.stack(images).astype(np.float32)
        Y = np.stack(masks).astype(np.float32)

        return X, Y


# ==========================================================================
# Deterministic evaluation sequence (verbatim)
# ==========================================================================

class EvaluationSegmentationSequence(tf.keras.utils.Sequence):

    def __init__(
        self,
        dataframe,
        batch_size=8,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dataframe = dataframe.reset_index(drop=True)
        self.batch_size = int(batch_size)

    def __len__(self):
        return int(
            np.ceil(
                len(self.dataframe) / self.batch_size
            )
        )

    def _load_row(self, row):
        npz_path = row["npz_path"]
        slice_index = int(row["slice_index"])

        with np.load(npz_path) as data:
            image = data["volume_normalized"][
                :, :, slice_index
            ]

            mask = data["consensus_mask"][
                :, :, slice_index
            ]

        image = image.astype(np.float32)[..., None]
        mask = mask.astype(np.float32)[..., None]

        return image, mask

    def __getitem__(self, index):

        start = index * self.batch_size
        end = min(
            start + self.batch_size,
            len(self.dataframe),
        )

        batch_df = self.dataframe.iloc[start:end]

        images = []
        masks = []

        for _, row in batch_df.iterrows():
            image, mask = self._load_row(row)

            images.append(image)
            masks.append(mask)

        X = np.stack(images).astype(np.float32)
        Y = np.stack(masks).astype(np.float32)

        return X, Y


# ==========================================================================
# Architecture configuration (verbatim)
# ==========================================================================

UNET_CONFIG = {
    "input_shape": (
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        CHANNELS,
    ),
    "base_filters": 32,
    "depth": 4,
    "dropout_encoder": 0.10,
    "dropout_bottleneck": 0.30,
    "batch_normalization": True,
    "kernel_initializer": "he_normal",
    "output_channels": 1,
    "output_activation": "sigmoid",
}


# ==========================================================================
# Convolution block (verbatim)
# ==========================================================================

def convolution_block(
    inputs,
    filters,
    dropout_rate=0.0,
    use_batch_norm=True,
    block_name=None,
):
    """
    Two-layer convolutional block used throughout the U-Net.

    Parameters
    ----------
    inputs:
        Input Keras tensor.
    filters:
        Number of convolutional filters.
    dropout_rate:
        Spatial dropout probability applied after the
        second activation.
    use_batch_norm:
        Whether to apply BatchNormalization after each
        convolution.
    block_name:
        Prefix used for layer names.
    """

    name = block_name or f"conv_{filters}"

    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer=UNET_CONFIG[
            "kernel_initializer"
        ],
        use_bias=not use_batch_norm,
        name=f"{name}_conv1",
    )(inputs)

    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization(
            name=f"{name}_bn1"
        )(x)

    x = tf.keras.layers.Activation(
        "relu",
        name=f"{name}_relu1",
    )(x)

    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer=UNET_CONFIG[
            "kernel_initializer"
        ],
        use_bias=not use_batch_norm,
        name=f"{name}_conv2",
    )(x)

    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization(
            name=f"{name}_bn2"
        )(x)

    x = tf.keras.layers.Activation(
        "relu",
        name=f"{name}_relu2",
    )(x)

    if dropout_rate > 0:
        x = tf.keras.layers.SpatialDropout2D(
            rate=dropout_rate,
            name=f"{name}_dropout",
        )(x)

    return x


# ==========================================================================
# Encoder block (verbatim)
# ==========================================================================

def encoder_block(
    inputs,
    filters,
    dropout_rate=0.0,
    use_batch_norm=True,
    block_name=None,
):
    """
    Encoder block returning:
    1. skip-connection tensor
    2. pooled tensor for the next encoder level
    """

    name = block_name or f"encoder_{filters}"

    skip = convolution_block(
        inputs=inputs,
        filters=filters,
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm,
        block_name=f"{name}_conv",
    )

    pooled = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        name=f"{name}_pool",
    )(skip)

    return skip, pooled


# ==========================================================================
# Decoder block (verbatim)
# ==========================================================================

def decoder_block(
    inputs,
    skip_connection,
    filters,
    dropout_rate=0.0,
    use_batch_norm=True,
    block_name=None,
):
    """
    Decoder block:
    Conv2DTranspose → concatenate skip connection
    → convolution block.
    """

    name = block_name or f"decoder_{filters}"

    x = tf.keras.layers.Conv2DTranspose(
        filters=filters,
        kernel_size=(2, 2),
        strides=(2, 2),
        padding="same",
        kernel_initializer=UNET_CONFIG[
            "kernel_initializer"
        ],
        name=f"{name}_upsample",
    )(inputs)

    x = tf.keras.layers.Concatenate(
        axis=-1,
        name=f"{name}_concat",
    )([x, skip_connection])

    x = convolution_block(
        inputs=x,
        filters=filters,
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm,
        block_name=f"{name}_conv",
    )

    return x


# ==========================================================================
# Network builder (verbatim)
# ==========================================================================

def build_unet(config=None):
    """
    Build a configurable four-level 2D U-Net.

    Returns
    -------
    tf.keras.Model
    """

    if config is None:
        config = UNET_CONFIG

    input_shape = tuple(config["input_shape"])
    base_filters = int(config["base_filters"])
    use_batch_norm = bool(
        config["batch_normalization"]
    )

    inputs = tf.keras.layers.Input(
        shape=input_shape,
        name="ct_input",
    )

    # Encoder level 1: 256 × 256
    skip1, pool1 = encoder_block(
        inputs=inputs,
        filters=base_filters,
        dropout_rate=0.0,
        use_batch_norm=use_batch_norm,
        block_name="encoder1",
    )

    # Encoder level 2: 128 × 128
    skip2, pool2 = encoder_block(
        inputs=pool1,
        filters=base_filters * 2,
        dropout_rate=0.0,
        use_batch_norm=use_batch_norm,
        block_name="encoder2",
    )

    # Encoder level 3: 64 × 64
    skip3, pool3 = encoder_block(
        inputs=pool2,
        filters=base_filters * 4,
        dropout_rate=config[
            "dropout_encoder"
        ],
        use_batch_norm=use_batch_norm,
        block_name="encoder3",
    )

    # Encoder level 4: 32 × 32
    skip4, pool4 = encoder_block(
        inputs=pool3,
        filters=base_filters * 8,
        dropout_rate=config[
            "dropout_encoder"
        ],
        use_batch_norm=use_batch_norm,
        block_name="encoder4",
    )

    # Bottleneck: 16 × 16
    bottleneck = convolution_block(
        inputs=pool4,
        filters=base_filters * 16,
        dropout_rate=config[
            "dropout_bottleneck"
        ],
        use_batch_norm=use_batch_norm,
        block_name="bottleneck",
    )

    # Decoder level 4: 32 × 32
    decoder4 = decoder_block(
        inputs=bottleneck,
        skip_connection=skip4,
        filters=base_filters * 8,
        dropout_rate=config[
            "dropout_encoder"
        ],
        use_batch_norm=use_batch_norm,
        block_name="decoder4",
    )

    # Decoder level 3: 64 × 64
    decoder3 = decoder_block(
        inputs=decoder4,
        skip_connection=skip3,
        filters=base_filters * 4,
        dropout_rate=config[
            "dropout_encoder"
        ],
        use_batch_norm=use_batch_norm,
        block_name="decoder3",
    )

    # Decoder level 2: 128 × 128
    decoder2 = decoder_block(
        inputs=decoder3,
        skip_connection=skip2,
        filters=base_filters * 2,
        dropout_rate=0.0,
        use_batch_norm=use_batch_norm,
        block_name="decoder2",
    )

    # Decoder level 1: 256 × 256
    decoder1 = decoder_block(
        inputs=decoder2,
        skip_connection=skip1,
        filters=base_filters,
        dropout_rate=0.0,
        use_batch_norm=use_batch_norm,
        block_name="decoder1",
    )

    outputs = tf.keras.layers.Conv2D(
        filters=config["output_channels"],
        kernel_size=(1, 1),
        padding="same",
        activation=config["output_activation"],
        name="segmentation_output",
    )(decoder1)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="PLOS_2D_UNet",
    )

    return model


# ==========================================================================
# Dice coefficient (verbatim)
# ==========================================================================

SMOOTH = 1e-6

def dice_coefficient(y_true, y_pred, smooth=SMOOTH):
    """
    Soft Dice coefficient averaged across the batch.

    y_true and y_pred shape:
    (batch, height, width, 1)
    """

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    axes = (1, 2, 3)

    intersection = tf.reduce_sum(
        y_true * y_pred,
        axis=axes,
    )

    denominator = (
        tf.reduce_sum(y_true, axis=axes)
        +
        tf.reduce_sum(y_pred, axis=axes)
    )

    dice_per_sample = (
        2.0 * intersection + smooth
    ) / (
        denominator + smooth
    )

    return tf.reduce_mean(dice_per_sample)


# ==========================================================================
# Foreground Dice (verbatim)
# ==========================================================================

SMOOTH = 1e-6
PREDICTION_THRESHOLD = 0.50

def foreground_dice(y_true, y_pred, smooth=SMOOTH):
    y_true = tf.cast(y_true >= 0.5, tf.float32)
    y_pred = tf.cast(
        y_pred >= PREDICTION_THRESHOLD,
        tf.float32,
    )

    axes = (1, 2, 3)

    true_area = tf.reduce_sum(y_true, axis=axes)
    pred_area = tf.reduce_sum(y_pred, axis=axes)

    intersection = tf.reduce_sum(
        y_true * y_pred,
        axis=axes,
    )

    dice_per_sample = (
        2.0 * intersection + smooth
    ) / (
        true_area + pred_area + smooth
    )

    foreground_samples = true_area > 0
    foreground_values = tf.boolean_mask(
        dice_per_sample,
        foreground_samples,
    )

    return tf.cond(
        tf.size(foreground_values) > 0,
        lambda: tf.reduce_mean(foreground_values),
        lambda: tf.constant(0.0, dtype=tf.float32),
    )


# ==========================================================================
# Intersection over union (verbatim)
# ==========================================================================

PREDICTION_THRESHOLD = 0.50

def binarize_predictions(y_pred):
    return tf.cast(
        y_pred >= PREDICTION_THRESHOLD,
        tf.float32,
    )


def thresholded_dice(y_true, y_pred, smooth=SMOOTH):
    y_true = tf.cast(y_true >= 0.5, tf.float32)
    y_pred = binarize_predictions(y_pred)

    axes = (1, 2, 3)

    intersection = tf.reduce_sum(
        y_true * y_pred,
        axis=axes,
    )

    denominator = (
        tf.reduce_sum(y_true, axis=axes)
        +
        tf.reduce_sum(y_pred, axis=axes)
    )

    dice_per_sample = (
        2.0 * intersection + smooth
    ) / (
        denominator + smooth
    )

    return tf.reduce_mean(dice_per_sample)


def intersection_over_union(
    y_true,
    y_pred,
    smooth=SMOOTH,
):
    y_true = tf.cast(y_true >= 0.5, tf.float32)
    y_pred = binarize_predictions(y_pred)

    axes = (1, 2, 3)

    intersection = tf.reduce_sum(
        y_true * y_pred,
        axis=axes,
    )

    union = (
        tf.reduce_sum(y_true, axis=axes)
        +
        tf.reduce_sum(y_pred, axis=axes)
        -
        intersection
    )

    iou_per_sample = (
        intersection + smooth
    ) / (
        union + smooth
    )

    return tf.reduce_mean(iou_per_sample)


# ==========================================================================
# Precision, recall and specificity (verbatim)
# ==========================================================================

def segmentation_precision(
    y_true,
    y_pred,
    smooth=SMOOTH,
):
    y_true = tf.cast(y_true >= 0.5, tf.float32)
    y_pred = binarize_predictions(y_pred)

    axes = (1, 2, 3)

    true_positive = tf.reduce_sum(
        y_true * y_pred,
        axis=axes,
    )

    false_positive = tf.reduce_sum(
        (1.0 - y_true) * y_pred,
        axis=axes,
    )

    precision_per_sample = (
        true_positive + smooth
    ) / (
        true_positive
        +
        false_positive
        +
        smooth
    )

    return tf.reduce_mean(
        precision_per_sample
    )


def segmentation_recall(
    y_true,
    y_pred,
    smooth=SMOOTH,
):
    y_true = tf.cast(y_true >= 0.5, tf.float32)
    y_pred = binarize_predictions(y_pred)

    axes = (1, 2, 3)

    true_positive = tf.reduce_sum(
        y_true * y_pred,
        axis=axes,
    )

    false_negative = tf.reduce_sum(
        y_true * (1.0 - y_pred),
        axis=axes,
    )

    recall_per_sample = (
        true_positive + smooth
    ) / (
        true_positive
        +
        false_negative
        +
        smooth
    )

    return tf.reduce_mean(
        recall_per_sample
    )


def segmentation_specificity(
    y_true,
    y_pred,
    smooth=SMOOTH,
):
    y_true = tf.cast(y_true >= 0.5, tf.float32)
    y_pred = binarize_predictions(y_pred)

    axes = (1, 2, 3)

    true_negative = tf.reduce_sum(
        (1.0 - y_true)
        *
        (1.0 - y_pred),
        axis=axes,
    )

    false_positive = tf.reduce_sum(
        (1.0 - y_true)
        *
        y_pred,
        axis=axes,
    )

    specificity_per_sample = (
        true_negative + smooth
    ) / (
        true_negative
        +
        false_positive
        +
        smooth
    )

    return tf.reduce_mean(
        specificity_per_sample
    )


# ==========================================================================
# Focal Tversky loss (verbatim)
# ==========================================================================

def tversky_index(
    y_true,
    y_pred,
    alpha=0.30,
    beta=0.70,
    smooth=1e-6,
):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    axes = (1, 2, 3)

    true_positive = tf.reduce_sum(
        y_true * y_pred,
        axis=axes,
    )

    false_positive = tf.reduce_sum(
        (1.0 - y_true) * y_pred,
        axis=axes,
    )

    false_negative = tf.reduce_sum(
        y_true * (1.0 - y_pred),
        axis=axes,
    )

    score = (
        true_positive + smooth
    ) / (
        true_positive
        + alpha * false_positive
        + beta * false_negative
        + smooth
    )

    return tf.reduce_mean(score)


def focal_tversky_loss(
    y_true,
    y_pred,
    alpha=0.30,
    beta=0.70,
    gamma=0.75,
):
    tversky = tversky_index(
        y_true,
        y_pred,
        alpha=alpha,
        beta=beta,
    )

    return tf.pow(
        1.0 - tversky,
        gamma,
    )


# ==========================================================================
# Self-check
# ==========================================================================

EXPECTED_TOTAL_PARAMETERS = 7_768_353
EXPECTED_TRAINABLE_PARAMETERS = 7_762_465


def _self_check():
    model = build_unet()
    trainable = int(sum(np.prod(v.shape) for v in model.trainable_weights))
    total = trainable + int(sum(np.prod(v.shape) for v in model.non_trainable_weights))
    print(f"input  {model.input_shape}")
    print(f"output {model.output_shape}")
    print(f"total parameters     {total:,} (manuscript: {EXPECTED_TOTAL_PARAMETERS:,})")
    print(f"trainable parameters {trainable:,} (manuscript: {EXPECTED_TRAINABLE_PARAMETERS:,})")
    assert model.input_shape == (None, 256, 256, 1)
    assert model.output_shape == (None, 256, 256, 1)
    assert total == EXPECTED_TOTAL_PARAMETERS, "architecture does not match the manuscript"
    assert trainable == EXPECTED_TRAINABLE_PARAMETERS
    print("OK: architecture matches the manuscript.")


if __name__ == "__main__":
    _self_check()
