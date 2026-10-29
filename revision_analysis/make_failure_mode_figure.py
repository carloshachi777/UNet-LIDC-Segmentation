"""
Figure: the seed-2 catastrophic over-segmentation failure mode.

Run this on the machine that holds the models and the processed volumes:

    cd <your clone of UNet-LIDC-Segmentation>
    python revision_analysis/make_failure_mode_figure.py

It writes figures/seed_failure_mode.png.

Why these two patients: LIDC-IDRI-0834 and LIDC-IDRI-0919 were held out of
training in BOTH the seed-1 and the seed-2 runs, so the comparison is like for
like. Under the seed-1 model they are segmented normally (mean 108 and 42
predicted pixels); under the seed-2 model the prediction floods the thorax
(up to 36,322 of 65,536 pixels against a 156-pixel reference).

Models are loaded with compile=False, so the custom focal Tversky loss and the
custom metrics do not need to be redefined here.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from pathlib import Path

from _paths import PROJECT, OUTPUT, results_dir, slice_index, metadata_file, per_slice_metrics, selected_threshold, model_path


PROJECT = find_project_root()
FIGURES = PROJECT / "figures"
THRESHOLD = 0.05          # the validation-selected threshold, identical in all three runs
MODEL_NAME = "best_plos_2d_unet_focal_tversky.keras"

# patient, scan id, slice index  (worst runaway slice for each patient under seed 2)
CASES = [
    ("LIDC-IDRI-0834", 821, 36),
    ("LIDC-IDRI-0919", 736, 66),
]

MODELS = {
    "Seed 1": PROJECT / "models" / "seed1" / MODEL_NAME,
    "Seed 2": PROJECT / "models" / "seed2" / MODEL_NAME,
}


def load_slice(patient_id, scan_id, slice_index):
    path = PROJECT / "processed" / f"{patient_id}_scan{scan_id}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as npz:
        image = npz["volume_normalized"][:, :, slice_index].astype(np.float32)
        mask = npz["consensus_mask"][:, :, slice_index].astype(np.uint8)
    return image, mask


def main():
    models = {}
    for label, path in MODELS.items():
        if not path.exists():
            raise FileNotFoundError(f"{path}\nCheck that the per-seed models were saved here.")
        models[label] = tf.keras.models.load_model(path, compile=False)
        print(f"loaded {label}: {path}")

    n_rows = len(CASES)
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4.2 * n_rows), squeeze=False)

    for row, (patient_id, scan_id, slice_index) in enumerate(CASES):
        image, mask = load_slice(patient_id, scan_id, slice_index)
        model_input = image[np.newaxis, ..., np.newaxis]

        axes[row, 0].imshow(image, cmap="gray")
        axes[row, 0].set_title(f"{patient_id}\nslice {slice_index}", fontsize=10)

        axes[row, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title(f"Consensus reference\n{int(mask.sum())} pixels", fontsize=10)

        for column, label in enumerate(["Seed 1", "Seed 2"], start=2):
            probability = models[label].predict(model_input, verbose=0)[0, :, :, 0]
            binary = (probability >= THRESHOLD).astype(np.uint8)

            intersection = int(np.logical_and(mask > 0, binary > 0).sum())
            denominator = int(mask.sum()) + int(binary.sum())
            dice = (2.0 * intersection / denominator) if denominator else 0.0

            axes[row, column].imshow(image, cmap="gray")
            axes[row, column].imshow(
                np.ma.masked_where(binary == 0, binary), cmap="autumn", alpha=0.55, vmin=0, vmax=1
            )
            if mask.any():
                axes[row, column].contour(mask, levels=[0.5], colors="lime", linewidths=1.8)
            axes[row, column].set_title(
                f"{label} model\n{int(binary.sum()):,} pixels predicted, Dice = {dice:.3f}",
                fontsize=10,
            )
            print(f"{patient_id} slice {slice_index} {label}: "
                  f"{int(binary.sum())} px, dice {dice:.4f}, max prob {probability.max():.3f}")

        for ax in axes[row]:
            ax.axis("off")

    fig.suptitle(
        "Run-to-run instability: identical protocol, identical held-out patients, different training seed\n"
        f"Predictions binarised at the validation-selected threshold of {THRESHOLD:.2f}; "
        "green contour is the consensus reference",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    FIGURES.mkdir(parents=True, exist_ok=True)
    output = FIGURES / "seed_failure_mode.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"\nsaved {output}")


if __name__ == "__main__":
    main()
