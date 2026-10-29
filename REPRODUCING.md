# Reproducing the results

This document states which code produced which reported number, what a fresh
clone can reproduce without any download, and what additionally requires the
LIDC-IDRI imaging data.

## The implementation of record

**The notebooks in `notebooks/` are the implementation of record.** Every model,
table and figure in the manuscript was produced by executing them in order. They
are committed with their outputs intact, so the logs behind the reported numbers
can be read without re-running anything.

| Notebook | Produces |
|---|---|
| `01_Preprocessing.ipynb` | Scan inventories, the 500-scan cohort, consensus masks, the processed `.npz` archive, the slice-level index |
| `01B_Seed_Sensitivity_Splits.ipynb` | The patient-level partitions for seeds 1 and 2 |
| `02_UNet_Training.ipynb` | The reference model (seed 42) and its training logs |
| `02B_Seed_Sensitivity_Training.ipynb` | The seed 1 and seed 2 models and logs |
| `03_UNet_Evaluation.ipynb` | Threshold selection, per-slice test metrics and the figures, seed 42 |
| `03B_Seed_Sensitivity_Evaluation.ipynb` | The same for seeds 1 and 2 |
| `cohort_selection_and_table1_audit.ipynb` | Table 1 in full, with assertions |

`src/unet_pipeline.py` is a convenience module: the architecture, the focal
Tversky loss, the metrics and the two data sequences, reproduced **verbatim**
from `02_UNet_Training.ipynb` so that they can be read and imported outside a
notebook. It does not re-train anything, and the notebooks remain authoritative.
Running it verifies that the architecture it defines has the 7,768,353
parameters reported in the manuscript:

```bash
python3 src/unet_pipeline.py
```

> **Note on earlier versions of this repository.** `src/` previously contained
> `train.py`, `model_unet.py`, `metrics.py` and `dataloader.py`. Those files were
> scaffolding from an early exploratory stage: they used a batch size of 16, a
> Dice loss, `.h5` checkpoints and a `.npy` data layout, none of which match the
> reported experiments, and `train.py` even retained a Colab `%%writefile`
> directive. They did not produce any reported result and have been removed to
> eliminate the contradiction.

## Revision analyses

`revision_analysis/` holds the analyses added during peer review, with their own
[README](revision_analysis/README.md) mapping each script to the quantity it
produces. None of them trains a model or changes a reported result: they
re-analyse the reference annotations, recompute statistics from the committed
per-slice metrics, or evaluate the existing checkpoints on test slices that were
always inside the held-out sets.

Four of them need nothing beyond this repository and regenerate their committed
JSON summaries, with every reported statistic identical (four p-values that the
manuscript does not report may vary in their final digits across SciPy builds;
see `revision_analysis/README.md`):

```bash
python3 revision_analysis/revision_statistics.py      # analysis_summary.json
python3 revision_analysis/multiseed_stability.py      # multiseed_summary.json
python3 revision_analysis/clustered_size_inference.py # clustered_size_summary.json
```

(`lesion_geometry_analysis.py` also needs the `processed/` archive, since it
reads the consensus masks, but it runs no model.)

Each writes its output beside the committed copy, so a mismatch is a one-line
`diff` away. All four were verified to produce identical results whether run
from a clone, where seed-42 files sit under `metadata/seed42/` and
`results/seed42/`, or from a working tree where they sit at the top level;
`revision_analysis/_paths.py` resolves both layouts.

The whole-test-set predictions released in this round deserve a specific note.
`results/seed*/all_test_slice_predictions.csv` scores **every** test slice, not
only the nodule-positive ones, giving per-slice predicted area and the full
confusion matrix. These are what the reported false-positive behaviour on
nodule-free slices is computed from, and they can be re-aggregated without
rerunning any model. `results/seed*/lesion_level_recall.csv` does the same at the
level of three-dimensional connected components of the reference masks.

## What reproduces with no download at all

The cohort audit reproduces **Table 1 of the manuscript in full**, from files
committed here. From a fresh clone:

```bash
pip install -r requirements.txt
jupyter notebook notebooks/cohort_selection_and_table1_audit.ipynb   # Run All
```

It reads only `metadata/lidc_annotation_inventory.csv`,
`metadata/lidc_download_inventory.csv` and
`metadata/selected_500_scan_cohort_seed42.csv`, and it ends with assertions on
every reported figure, so a silent mismatch fails loudly. Expected output:

```
Eligible decomposition: 883 = 500 + 378 + 5
Selected   : 443 (270-580)
Comparator : 470 (270-604)
Mann-Whitney U = 89461.0 ; two-sided p = 0.233373
ALL COHORT AND TABLE 1 AUDIT ASSERTIONS PASSED
```

The per-slice test metrics in `results/seed*/` likewise support re-deriving every
reported performance number, confidence interval and figure without the imaging
data, since they contain the full confusion matrix of each evaluated slice. This
now covers the whole test set rather than its nodule-positive part: see
`all_test_slice_predictions.csv` and `lesion_level_recall.csv` in each seed
directory, and `results/negative_slice_summary.json` for the per-run summary.

## What requires the LIDC-IDRI imaging data

Re-running preprocessing, training or inference needs the images, which cannot
be redistributed here.

1. Obtain LIDC-IDRI from The Cancer Imaging Archive:
   <https://www.cancerimagingarchive.net/collection/lidc-idri/>
2. Configure `pylidc` by creating `~/.pylidcrc` pointing at the DICOM root:

   ```ini
   [dicom]
   path = /path/to/manifest-xxxxxxxx/LIDC-IDRI
   warn = True
   ```

3. Run `01_Preprocessing.ipynb`. It writes one compressed `.npz` per scan into
   `processed/`, holding `volume_hu`, `volume_normalized`, `consensus_mask` and
   the slice-level metadata. **This directory is roughly 24 GB for the
   500-scan cohort and is deliberately not committed**; `.gitignore` excludes it.
4. Run the training and evaluation notebooks in the order of the table above.

Trained checkpoints are approximately 93 MB each and are not committed either;
see `models/README.md`.

## Paths

The slice-level indices in `metadata/seed*/` reference the processed archive
with **repository-relative** paths, for example
`processed/LIDC-IDRI-0001_scan12.npz`, so they resolve correctly from the
repository root on any machine. Earlier releases of these files carried absolute
paths from the authoring machine (`/Volumes/.../processed/...`); if you hold an
older copy, resolve each entry by filename against your own `processed/`
directory. Scripts in `revision_analysis/` do this automatically.

`lidc_download_inventory.csv` is published with the per-patient DICOM file
counts used in Table 1; the authoring machine's local directory paths have been
removed, as they are neither useful nor portable.

## Determinism

Seeds are fixed for Python, NumPy and TensorFlow. The three reported runs use
seeds 42 (reference), 1 and 2. The cohort of 500 scans is drawn with seed 42 and
is **held fixed across all three runs**; the seed under test governs the
patient-level partition, the weight initialisation and the sampling order only.

Exact bitwise reproduction of the trained weights is not guaranteed across
different hardware, TensorFlow builds or GPU kernels. The committed per-slice
metrics and training logs are therefore the authoritative record of the reported
results. The reported run-to-run spread (mean per-slice Dice 0.416, SD 0.026)
gives the scale against which any re-run should be compared.

## Environment

Reported results were produced with TensorFlow 2.21.0 and NumPy 1.26.4 on an
Apple M4 workstation. See `requirements.txt`.
