"""Lesion-level (3D) analysis of the reference run, seed 42.

Addresses Reviewer 3 point 2: slice-level foreground area is a cross-section,
not the size of a nodule. Here each connected 3D component of the consensus
mask is treated as one lesion.
"""
import json
import numpy as np, pandas as pd
from scipy import ndimage
from pathlib import Path

from pathlib import Path

from _paths import PROJECT, OUTPUT, results_dir, slice_index, metadata_file, per_slice_metrics, selected_threshold, model_path


ROOT = PROJECT

idx = pd.read_csv(slice_index(42))
met = pd.read_csv(per_slice_metrics(42))
test_pos = idx[(idx.split == "test") & (idx.is_positive.astype(bool))].reset_index(drop=True)
assert len(test_pos) == len(met), (len(test_pos), len(met))
slices = pd.concat([test_pos[["patient_id", "scan_id", "slice_index", "npz_path"]],
                    met.drop(columns=["slice_index"])], axis=1)

# released paths are absolute from the author's machine; resolve against the project root
def local(p):
    return ROOT / Path(p).name if False else ROOT / "processed" / Path(p).name

STRUCT = ndimage.generate_binary_structure(3, 3)   # 26-connectivity

lesion_rows, slice_rows = [], []
for (pid, sid), grp in slices.groupby(["patient_id", "scan_id"]):
    with np.load(local(grp.npz_path.iloc[0])) as z:
        mask = z["consensus_mask"]                      # H x W x S
    labels, n = ndimage.label(mask > 0, structure=STRUCT)
    if n == 0:
        continue
    # per-lesion geometry
    for lid in range(1, n + 1):
        comp = labels == lid
        zs = np.flatnonzero(comp.any(axis=(0, 1)))
        areas = comp.sum(axis=(0, 1))[zs]
        lesion_rows.append(dict(
            patient_id=pid, scan_id=sid, lesion_id=lid,
            volume_vox=int(comp.sum()),
            n_slices=int(len(zs)),
            slice_min=int(zs.min()), slice_max=int(zs.max()),
            max_cross_section=int(areas.max()),
            median_cross_section=float(np.median(areas)),
            min_cross_section=int(areas.min()),
            equiv_diam_vox=float(2 * (3 * comp.sum() / (4 * np.pi)) ** (1 / 3)),
        ))
    # attribute each evaluated slice to the lesion holding most of its foreground
    for _, r in grp.iterrows():
        si = int(r.slice_index)
        lab = labels[:, :, si]
        present = np.bincount(lab[lab > 0].ravel())
        if present.size <= 1:
            continue
        lid = int(np.argmax(present))
        comp = labels == lid
        zs = np.flatnonzero(comp.any(axis=(0, 1)))
        areas_all = comp.sum(axis=(0, 1))
        slice_rows.append(dict(
            patient_id=pid, scan_id=sid, slice_index=si, lesion_id=lid,
            slice_area=int(r.ground_truth_pixels),
            lesion_volume_vox=int(comp.sum()),
            lesion_n_slices=int(len(zs)),
            lesion_max_cross_section=int(areas_all.max()),
            area_fraction_of_max=float(areas_all[si] / areas_all.max()),
            position_in_lesion=float((si - zs.min()) / max(len(zs) - 1, 1)),
            dice=float(r.dice), detected=bool(r.detected),
            tp=int(r.true_positive_pixels), fp=int(r.false_positive_pixels), fn=int(r.false_negative_pixels),
        ))

les = pd.DataFrame(lesion_rows)
sl = pd.DataFrame(slice_rows)

# lesion-level Dice: pool the confusion matrix over the lesion's evaluated slices
agg = (sl.groupby(["patient_id", "scan_id", "lesion_id"])
         .agg(tp=("tp", "sum"), fp=("fp", "sum"), fn=("fn", "sum"),
              n_eval_slices=("dice", "size"), slices_detected=("detected", "sum"),
              mean_slice_dice=("dice", "mean"))
         .reset_index())
agg["lesion_dice"] = 2 * agg.tp / (2 * agg.tp + agg.fp + agg.fn).replace(0, np.nan)
agg["lesion_detected"] = agg.tp > 0
les_eval = les.merge(agg, on=["patient_id", "scan_id", "lesion_id"], how="inner")

out = {}
out["n_lesions_total"] = int(len(les))
out["n_lesions_evaluated"] = int(len(les_eval))
out["lesion_volume_median"] = float(les_eval.volume_vox.median())
out["lesion_volume_iqr"] = [float(les_eval.volume_vox.quantile(.25)), float(les_eval.volume_vox.quantile(.75))]
out["lesion_volume_range"] = [int(les_eval.volume_vox.min()), int(les_eval.volume_vox.max())]
out["lesion_nslices_median"] = float(les_eval.n_slices.median())
out["lesion_nslices_iqr"] = [float(les_eval.n_slices.quantile(.25)), float(les_eval.n_slices.quantile(.75))]
out["lesion_nslices_max"] = int(les_eval.n_slices.max())
out["lesion_dice_mean"] = float(les_eval.lesion_dice.mean())
out["lesion_dice_median"] = float(les_eval.lesion_dice.median())
out["lesion_detection_rate"] = float(les_eval.lesion_detected.mean())

# the reviewer's specific point, quantified
out["slices_below_half_max_cross_section"] = float((sl.area_fraction_of_max < 0.5).mean())
out["slices_below_quarter_max"] = float((sl.area_fraction_of_max < 0.25).mean())
big = sl[sl.lesion_volume_vox >= sl.lesion_volume_vox.quantile(.75)]
out["in_largest_quartile_lesions_slices_in_smallest_area_tertile"] = float(
    (big.slice_area <= sl.slice_area.quantile(1/3)).mean())
out["corr_slice_area_vs_lesion_volume_spearman"] = float(
    sl[["slice_area", "lesion_volume_vox"]].corr(method="spearman").iloc[0, 1])

les_eval.to_csv(OUTPUT / "lesion_level_metrics_seed42.csv", index=False)
sl.to_csv(OUTPUT / "slice_to_lesion_seed42.csv", index=False)
json.dump(out, open(OUTPUT / "lesion_summary.json", "w"), indent=2)
for k, v in out.items():
    print(f"{k:55s} {v}")
