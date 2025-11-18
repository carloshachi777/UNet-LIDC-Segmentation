%%writefile /content/drive/MyDrive/UNet-LIDC-Segmentation/src/dataloader.py
import numpy as np
from tensorflow.keras.utils import Sequence


class SliceDataGenerator(Sequence):
    """
    Balanced data generator for 2D CT slices and masks.

    Expects:
      images: np.ndarray [N, H, W]
      masks:  np.ndarray [N, H, W]
      pos_indices, neg_indices: arrays of indices for slices with / without nodules
    """

    def __init__(
        self,
        images,
        masks,
        pos_indices,
        neg_indices,
        batch_size=16,
        shuffle=True,
    ):
        self.images = images
        self.masks = masks
        self.pos_indices = np.array(pos_indices)
        self.neg_indices = np.array(neg_indices)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor((len(self.pos_indices) + len(self.neg_indices)) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.pos_indices)
            np.random.shuffle(self.neg_indices)

    def __getitem__(self, idx):
        # half positive, half negative (or as close as possible)
        half = self.batch_size // 2

        pos_sel = self.pos_indices[
            (idx * half) % len(self.pos_indices) : ((idx + 1) * half) % len(self.pos_indices)
        ]
        neg_sel = self.neg_indices[
            (idx * half) % len(self.neg_indices) : ((idx + 1) * half) % len(self.neg_indices)
        ]

        batch_indices = np.concatenate([pos_sel, neg_sel])
        np.random.shuffle(batch_indices)

        batch_x = self.images[batch_indices].astype("float32")
        batch_y = self.masks[batch_indices].astype("float32")

        # add channel dimension
        batch_x = np.expand_dims(batch_x, axis=-1)
        batch_y = np.expand_dims(batch_y, axis=-1)

        return batch_x, batch_y
