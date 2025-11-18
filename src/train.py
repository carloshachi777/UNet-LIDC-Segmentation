%%writefile /content/drive/MyDrive/UNet-LIDC-Segmentation/src/train.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from model_unet import build_unet
from metrics import dice_coefficient, dice_loss, binary_iou
from dataloader import SliceDataGenerator


def load_data(data_dir):
    """
    Expects preprocessed NumPy arrays:
      images_train.npy, masks_train.npy, pos_idx_train.npy, neg_idx_train.npy, etc.
    Adjust names to match your preprocessing notebook.
    """
    x_train = np.load(os.path.join(data_dir, "images_train.npy"))
    y_train = np.load(os.path.join(data_dir, "masks_train.npy"))
    x_val = np.load(os.path.join(data_dir, "images_val.npy"))
    y_val = np.load(os.path.join(data_dir, "masks_val.npy"))

    pos_idx_train = np.load(os.path.join(data_dir, "pos_idx_train.npy"))
    neg_idx_train = np.load(os.path.join(data_dir, "neg_idx_train.npy"))

    return x_train, y_train, x_val, y_val, pos_idx_train, neg_idx_train


def main():
    data_dir = "data/preprocessed"  # update if needed
    os.makedirs("results/models", exist_ok=True)

    x_train, y_train, x_val, y_val, pos_idx_train, neg_idx_train = load_data(data_dir)

    train_gen = SliceDataGenerator(
        x_train,
        y_train,
        pos_idx_train,
        neg_idx_train,
        batch_size=16,
        shuffle=True,
    )

    # simple non-balanced generator for validation
    x_val = np.expand_dims(x_val, -1).astype("float32")
    y_val = np.expand_dims(y_val, -1).astype("float32")

    model = build_unet(input_shape=(256, 256, 1))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=dice_loss,
        metrics=[dice_coefficient, binary_iou, "accuracy"],
    )

    checkpoint = ModelCheckpoint(
        "results/models/unet_lidc_best.h5",
        monitor="val_dice_coefficient",
        mode="max",
        save_best_only=True,
        verbose=1,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_dice_coefficient",
        factor=0.5,
        patience=5,
        verbose=1,
        min_lr=1e-6,
    )
    early_stop = EarlyStopping(
        monitor="val_dice_coefficient",
        patience=10,
        mode="max",
        restore_best_weights=True,
        verbose=1,
    )

    model.fit(
        train_gen,
        validation_data=(x_val, y_val),
        epochs=50,
        callbacks=[checkpoint, reduce_lr, early_stop],
        verbose=1,
    )

    model.save("results/models/unet_lidc_final.h5")


if __name__ == "__main__":
    main()
