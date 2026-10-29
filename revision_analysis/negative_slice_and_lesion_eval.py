"""
Evaluation of the already-trained checkpoints on the parts of the held-out test
sets that were never tabulated. No training, no architecture change: this only
runs the three saved models over test slices and writes CSVs.

It answers two of the fourth-round review points at once.

  Point 4  False positives on nodule-negative test slices. Every test slice is
           scored, not only the nodule-positive ones, so the manuscript can
           report how often the model invents foreground where no nodule exists,
           and how far the seed-2 over-segmentation failure extends.

  Point 2  Exact lesion-level recall. The consensus mask of each scan is split
           into 3D connected components, and each component's true-positive and
           false-negative voxels are counted directly, so lesion detection is
           measured per lesion rather than per slice.

Run it from anywhere:

    cd /Volumes/KINGSTON/PLOS_2D_UNet_Revision
    python revision_analysis/negative_slice_and_lesion_eval.py

Expect a few minutes per seed on the M4. Outputs go to results/<seed>/ and to
results/negative_slice_summary.json. Send me those files and I will fold the
numbers into the manuscript.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
import tensorflow as tf

from _paths import PROJECT, OUTPUT, results_dir, slice_index, metadata_file, per_slice_metrics, selected_threshold, model_path


# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------

def find_project_root():
    here = Path(__file__).resolve()
    for candidate in [here.parent.parent, Path.cwd(), Path.cwd().parent]:
        if (candidate / "metadata").exists() and (candidate / "processed").exists():
            return candidate
    raise FileNotFoundError("Could not locate the project root; run from the project directory.")

PROJECT = find_project_root()
PROCESSED = PROJECT / "processed"
BATCH = 16
MODEL_NAME = "best_plos_2d_unet_focal_tversky.keras"
RUNAWAY = 10_000          # same definition used in the manuscript
PIXELS = 256 * 256

def _seed_config(seed):
    return dict(index=slice_index(seed),
                model=model_path(seed, MODEL_NAME),
                results=results_dir(seed),
                threshold=selected_threshold(seed))


SEED_LIST = (42, 1, 2)

STRUCT = ndimage.generate_binary_structure(3, 3)     # 26-connectivity


def resolve(npz_path):
    """The released slice index stores absolute paths from the authoring machine."""
    return PROCESSED / Path(str(npz_path)).name


# ----------------------------------------------------------------------------
# per-seed evaluation
# ----------------------------------------------------------------------------

def evaluate_seed(seed, cfg):
    print(f"\n=== seed {seed} " + "=" * 50)
    for key in ("index", "model", "threshold"):
        if not cfg[key].exists():
            raise FileNotFoundError(f"seed {seed}: missing {cfg[key]}")

    threshold = float(json.load(open(cfg["threshold"]))["selected_threshold"])
    model = tf.keras.models.load_model(cfg["model"], compile=False)
    print(f"model: {cfg['model'].name} | threshold {threshold:.2f}")

    index = pd.read_csv(cfg["index"])
    test = index[index.split == "test"].copy()
    print(f"test slices: {len(test):,} in {test.patient_id.nunique()} patients")

    slice_rows, lesion_rows = [], []

    for (pid, sid), grp in test.groupby(["patient_id", "scan_id"], sort=False):
        path = resolve(grp.npz_path.iloc[0])
        with np.load(path) as z:
            volume = z["volume_normalized"]
            mask = z["consensus_mask"]

        wanted = np.sort(grp.slice_index.astype(int).values)
        images = volume[:, :, wanted].transpose(2, 0, 1)[..., np.newaxis].astype(np.float32)
        probabilities = model.predict(images, batch_size=BATCH, verbose=0)[..., 0]
        binary = (probabilities >= threshold).astype(np.uint8)

        truth = (mask[:, :, wanted].transpose(2, 0, 1) > 0).astype(np.uint8)
        tp = np.logical_and(truth == 1, binary == 1).sum(axis=(1, 2))
        fp = np.logical_and(truth == 0, binary == 1).sum(axis=(1, 2))
        fn = np.logical_and(truth == 1, binary == 0).sum(axis=(1, 2))

        for k, si in enumerate(wanted):
            slice_rows.append(dict(
                patient_id=pid, scan_id=int(sid), slice_index=int(si),
                is_positive=bool(truth[k].any()),
                gt_pixels=int(truth[k].sum()),
                predicted_pixels=int(binary[k].sum()),
                true_positive_pixels=int(tp[k]),
                false_positive_pixels=int(fp[k]),
                false_negative_pixels=int(fn[k]),
                max_probability=float(probabilities[k].max()),
            ))

        # lesion-level recall, computed on the lesion's own voxels
        labels, n = ndimage.label(mask > 0, structure=STRUCT)
        if n:
            predicted_volume = np.zeros_like(mask, dtype=np.uint8)
            predicted_volume[:, :, wanted] = binary.transpose(1, 2, 0)
            for lid in range(1, n + 1):
                comp = labels == lid
                voxels = int(comp.sum())
                hit = int(np.logical_and(comp, predicted_volume > 0).sum())
                zs = np.flatnonzero(comp.any(axis=(0, 1)))
                lesion_rows.append(dict(
                    patient_id=pid, scan_id=int(sid), lesion_id=int(lid),
                    volume_vox=voxels, n_slices=int(len(zs)),
                    max_cross_section=int(comp.sum(axis=(0, 1)).max()),
                    true_positive_vox=hit,
                    false_negative_vox=voxels - hit,
                    recall=hit / voxels,
                    detected=bool(hit > 0),
                ))

    slices = pd.DataFrame(slice_rows)
    lesions = pd.DataFrame(lesion_rows)

    cfg["results"].mkdir(parents=True, exist_ok=True)
    slices.to_csv(cfg["results"] / "all_test_slice_predictions.csv", index=False)
    lesions.to_csv(cfg["results"] / "lesion_level_recall.csv", index=False)

    negative = slices[~slices.is_positive]
    positive = slices[slices.is_positive]

    summary = dict(
        seed=seed, threshold=threshold,
        n_test_slices=int(len(slices)),
        n_negative_slices=int(len(negative)),
        n_positive_slices=int(len(positive)),
        # ---- false positives on nodule-free slices
        negative_slices_with_any_fp=int((negative.predicted_pixels > 0).sum()),
        negative_fp_rate=float((negative.predicted_pixels > 0).mean()),
        negative_mean_fp_pixels=float(negative.predicted_pixels.mean()),
        negative_median_fp_pixels_when_nonzero=float(
            negative.loc[negative.predicted_pixels > 0, "predicted_pixels"].median()
            if (negative.predicted_pixels > 0).any() else 0.0),
        negative_max_fp_pixels=int(negative.predicted_pixels.max()),
        negative_runaway_slices=int((negative.predicted_pixels > RUNAWAY).sum()),
        negative_runaway_patients=int(
            negative.loc[negative.predicted_pixels > RUNAWAY, "patient_id"].nunique()),
        negative_total_fp_pixels=int(negative.false_positive_pixels.sum()),
        positive_total_fp_pixels=int(positive.false_positive_pixels.sum()),
        fp_pixels_per_negative_slice=float(negative.false_positive_pixels.mean()),
        negative_fp_fraction_of_area=float(negative.false_positive_pixels.sum()
                                           / (len(negative) * PIXELS)),
        # ---- whole-scan view
        whole_test_global_dice=float(
            2 * slices.true_positive_pixels.sum()
            / (2 * slices.true_positive_pixels.sum()
               + slices.false_positive_pixels.sum()
               + slices.false_negative_pixels.sum())),
        positive_only_global_dice=float(
            2 * positive.true_positive_pixels.sum()
            / (2 * positive.true_positive_pixels.sum()
               + positive.false_positive_pixels.sum()
               + positive.false_negative_pixels.sum())),
        # ---- lesion level
        n_lesions=int(len(lesions)),
        lesion_detection_rate=float(lesions.detected.mean()),
        lesion_mean_recall=float(lesions.recall.mean()),
        lesion_median_volume=float(lesions.volume_vox.median()),
    )

    for name, value in summary.items():
        print(f"  {name:42s} {value}")
    return summary


def main():
    print(f"project root: {PROJECT}")
    summaries = {}
    for seed in SEED_LIST:
        try:
            cfg = _seed_config(seed)
        except FileNotFoundError as error:
            print(f'!! seed {seed} skipped: {error}')
            continue
        try:
            summaries[seed] = evaluate_seed(seed, cfg)
        except FileNotFoundError as error:
            print(f"!! seed {seed} skipped: {error}")
    out = PROJECT / "results/negative_slice_summary.json"
    json.dump(summaries, open(out, "w"), indent=2, default=str)
    print(f"\nwrote {out}")
    print("Also written, per seed: all_test_slice_predictions.csv and lesion_level_recall.csv")


if __name__ == "__main__":
    main()
