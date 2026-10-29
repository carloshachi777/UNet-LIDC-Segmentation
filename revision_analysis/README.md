# Revision analyses

Scripts written during peer review to answer specific reviewer questions. They
sit outside the main pipeline: none of them trains a model or alters a reported
result. They either re-analyse the reference annotations, recompute statistics
from the per-slice metrics in `results/`, or evaluate the existing checkpoints on
test slices that were already in the held-out sets.

Every script locates the repository root by itself, so they can be run from
anywhere inside a clone:

```bash
python3 revision_analysis/<script>.py
```

On macOS there is no `python` command, only `python3`. If you are working inside
an activated virtual environment either name works.

Outputs are written back into this directory, except where noted.

## What produced which reported number

| Script | Needs | Produces | Reported in |
|---|---|---|---|
| `revision_statistics.py` | `metadata/`, `results/` | `analysis_summary.json`, `per_patient_test_dice.csv`, `test_slice_metrics_with_patient.csv` | Reference-run test metrics, patient-clustered bootstrap intervals, size tertiles, pixel confusion matrix |
| `multiseed_stability.py` | `metadata/`, `results/seed*/` | `multiseed_summary.json` | Run-to-run stability across seeds 42, 1 and 2 |
| `lesion_geometry_analysis.py` | `processed/`, `metadata/`, `results/` | `lesion_summary.json`, `lesion_level_metrics_seed42.csv`, `slice_to_lesion_seed42.csv` | Lesion geometry: 226 lesions, median volume 37 voxels, 26.8% of slices below half the maximum cross-section |
| `clustered_size_inference.py` | `metadata/`, `results/` | `clustered_size_summary.json` | Patient-clustered intervals for the area–Dice association, and the patient-level Spearman check |
| `negative_slice_and_lesion_eval.py` | `processed/`, `models/`, `metadata/`, `results/` | `results/negative_slice_summary.json`, `results/*/all_test_slice_predictions.csv`, `results/*/lesion_level_recall.csv` | False positives on nodule-free slices; lesion-level detection and recall |
| `revision_figures.py` | `results/`, outputs of `revision_statistics.py` | figures in `figures/` | Cohort size distribution, test-set variability |
| `architecture_figure.py` | none | `figures/unet_architecture_diagram.png` | The architecture diagram |
| `make_failure_mode_figure.py` | `processed/`, `models/seed1/`, `models/seed2/` | `figures/seed_failure_mode.png` | The same held-out patients under two seeds |

## Requirements

All scripts need `numpy`, `pandas` and `scipy`; the figure scripts also need
`matplotlib`. Two of them load the trained checkpoints and therefore need
`tensorflow`, and the LIDC-IDRI-derived `processed/` archive, which is not
redistributed here (see `REPRODUCING.md`):

- `negative_slice_and_lesion_eval.py`
- `make_failure_mode_figure.py`

`lesion_geometry_analysis.py` needs `processed/` but not TensorFlow: it reads the
consensus masks only, and runs no model.

The remaining scripts need nothing beyond what is committed to this repository,
so their outputs can be regenerated from a clone with no imaging download.

## Verification

`revision_statistics.py`, `multiseed_stability.py`, `lesion_geometry_analysis.py`
and `clustered_size_inference.py` were re-run from a clean clone and reproduce
the committed JSON summaries key for key, with every reported statistic
identical.

Four entries in `analysis_summary.json` may differ in their final digits across
SciPy builds: `spearman_p`, `ks_clusters_p`, `ks_train_test_size_p` and
`ks_clusters_elig_p`. The discrepancy is of order one part in 10^15 and arises
from the underlying linear-algebra library, not from the analysis. None of these
four values is reported in the manuscript; the associations they relate to are
reported as patient-clustered confidence intervals instead, because slices from
one patient are not independent and a slice-level p-value would be
anti-conservative. Any disagreement beyond those four entries is worth
reporting.

Exact bitwise reproduction of trained weights is not guaranteed across hardware
and TensorFlow builds, so `negative_slice_and_lesion_eval.py` run against your
own retrained checkpoints may differ slightly from the committed CSVs. Run
against the released checkpoints it is deterministic.
