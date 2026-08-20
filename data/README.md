# LIDC-IDRI Dataset Access

This directory contains instructions for obtaining and preparing the **Lung Image Database Consortium and Image Database Resource Initiative (LIDC-IDRI)** dataset used in this project.

The repository does **not** redistribute the original LIDC-IDRI DICOM images.

---

## Dataset Source

The LIDC-IDRI collection is publicly available through **The Cancer Imaging Archive (TCIA)**.

Collection:

https://www.cancerimagingarchive.net/collection/lidc-idri/

To obtain the dataset:

1. Access the LIDC-IDRI collection through TCIA.
2. Download the collection using the **NBIA Data Retriever** or another TCIA-supported download method.
3. Preserve the original patient and scan directory organization.

A typical local directory structure is:

```text
UNet-LIDC-Segmentation/
└── data/
    └── LIDC-IDRI/
        ├── LIDC-IDRI-0001/
        ├── LIDC-IDRI-0002/
        ├── LIDC-IDRI-0003/
        └── ...
```

The `LIDC-IDRI/` directory is excluded from version control and should **not** be committed to GitHub.

---

## Cohort Construction

The experiments reported in the associated manuscript use a reproducibly selected cohort of:

```text
500 CT scans
498 unique patients
```

Two patients contribute two scans each.

The cohort is selected using a fixed cohort-selection seed:

```python
COHORT_SEED = 42
```

The cohort-selection seed remains fixed across the reference experiment and the seed-sensitivity experiments.

Patient-level partitioning is performed separately to ensure that **no patient contributes scans or slices to more than one of the training, validation, or test partitions**.

---

## Preprocessing and Consensus Masks

Run:

```text
notebooks/01_Preprocessing.ipynb
```

The preprocessing pipeline performs the main data-preparation steps required for the experiments, including:

- loading the selected LIDC-IDRI CT scans
- accessing radiologist annotations using **PyLIDC**
- grouping annotations associated with the same nodule
- generating consensus segmentation masks
- extracting axial CT slices
- preprocessing and resizing slices to `256 × 256`
- identifying nodule-positive and nodule-negative slices
- generating slice-level metadata required by the downstream training and evaluation pipeline

A slice is considered **nodule-positive** when its consensus reference mask contains at least one foreground pixel.

---

## Patient-Level Data Partitioning

The cohort is partitioned at the **patient level**, rather than at the slice level.

This is important because a single CT scan contributes many highly correlated axial slices. Slice-level random splitting could therefore introduce information leakage between training, validation, and testing.

The approximate target allocation is:

```text
Training:   70%
Validation: 15%
Test:       15%
```

The reference experiment and seed-sensitivity experiments use separate run seeds while retaining the same fixed cohort.

The reference experiment uses:

```python
RUN_SEED = 42
```

Additional sensitivity experiments use:

```python
RUN_SEED = 1
RUN_SEED = 2
```

Seed-specific partition information is stored separately so that each experimental run can be independently reproduced and audited.

---

## Seed-Sensitivity Splits

The additional patient partitions used for the seed-sensitivity analysis can be generated using:

```text
notebooks/01B_Seed_Sensitivity_Splits.ipynb
```

The fixed cohort does not change between runs.

Instead, the run seed controls the stochastic components associated with the experiment, including patient partitioning and subsequent run-specific randomization.

Generated split information should be stored under:

```text
splits/
├── seed1/
├── seed2/
└── seed42/
```

Corresponding metadata can be stored under:

```text
metadata/
├── cohort/
├── seed1/
├── seed2/
└── seed42/
```

---

## Local Data vs. Repository Data

Large imaging and intermediate preprocessing files should remain local.

The following should **not** be uploaded to GitHub:

```text
raw DICOM files
large NumPy arrays
temporary preprocessing files
local caches
large model checkpoints
```

The repository instead provides lightweight reproducibility artifacts such as:

- cohort identifiers
- patient-level partition assignments
- slice indices
- positive/negative slice indicators
- training logs
- threshold-selection results
- evaluation summaries

This allows the experimental design to be audited without redistributing the original imaging dataset.

---

## Example Images

The directory:

```text
data/examples/
```

contains only example PNG images used to illustrate the preprocessing and segmentation pipeline.

These images:

- contain no DICOM headers
- contain no original DICOM metadata
- are provided only for research documentation and visualization

The original LIDC-IDRI imaging data must be obtained directly from TCIA.

---

## Reproducing the Experimental Pipeline

After obtaining the LIDC-IDRI dataset, the main workflow is:

```text
LIDC-IDRI DICOM data
        │
        ▼
01_Preprocessing.ipynb
        │
        ├── cohort construction
        ├── PyLIDC annotations
        ├── consensus masks
        └── slice-level metadata
        │
        ▼
02_UNet_Training.ipynb
        │
        ├── patient-level partitions
        ├── balanced sampling
        └── U-Net training
        │
        ▼
03_UNet_Evaluation.ipynb
        │
        ├── validation threshold selection
        ├── held-out test evaluation
        └── segmentation metrics
```

Seed-sensitivity experiments use:

```text
01B_Seed_Sensitivity_Splits.ipynb
        ↓
02B_Seed_Sensitivity_Training.ipynb
        ↓
03B_Seed_Sensitivity_Evaluation.ipynb
```

---

## Data Availability

The LIDC-IDRI dataset is publicly available from **The Cancer Imaging Archive (TCIA)**.

This repository provides the code and reproducibility artifacts necessary to reconstruct the experimental pipeline but does not redistribute the original LIDC-IDRI DICOM data.

---

## Citation

Use of LIDC-IDRI should cite the original dataset publication:

> Armato SG III, McLennan G, Bidaut L, et al.  
> The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): A completed reference database of lung nodules on CT scans.  
> *Medical Physics*. 2011;38(2):915–931.

Please also follow the current citation and data-usage requirements specified by TCIA when using the LIDC-IDRI collection.

---

## Research Use

The data-processing and segmentation code in this repository is provided for **research and educational purposes only**.

The resulting models are not intended for clinical diagnosis, treatment planning, or other clinical decision-making.
