# UNet-LIDC-Segmentation

**Reproducible 2D U-Net Baseline for Lung Nodule Segmentation on LIDC-IDRI**

This repository provides an open-source and reproducible implementation of a **2D U-Net baseline for lung nodule segmentation in CT scans using the LIDC-IDRI dataset**.

It accompanies the manuscript:

**“Automated Segmentation of Lung Nodules in CT Scans Using a U-Net Deep Learning Model: A Study Based on the LIDC-IDRI Dataset.”**

The objective of this project is not to claim state-of-the-art segmentation performance. Instead, it provides a **transparent and reproducible experimental baseline** for studying lung nodule segmentation under substantial foreground/background class imbalance.

The repository supports reproduction of the main experimental workflow, including:

- LIDC-IDRI cohort construction
- PyLIDC-based consensus mask generation
- Patient-level train/validation/test partitioning
- Slice-level preprocessing
- Balanced positive/negative slice sampling
- 2D U-Net training
- Focal Tversky optimization
- Validation-based segmentation threshold selection
- Positive-slice and full-test-set evaluation
- Patient-clustered bootstrap confidence intervals
- Seed-sensitivity experiments
- Qualitative prediction visualization



## 🔍 Project Overview

The experimental pipeline is organized around a fixed cohort of **500 LIDC-IDRI CT scans corresponding to 498 unique patients**.

Two patients contribute two scans each. To prevent information leakage, all partitions are constructed at the **patient level**, ensuring that no patient contributes scans or slices to more than one partition.

The reference experiment uses a fixed cohort-selection seed and a separate run seed. Additional runs are performed using different run seeds to evaluate the sensitivity of the pipeline to patient partitioning, model initialization, and balanced sampling.

### Main Experimental Configuration

| Component | Configuration |
|---|---|
| Dataset | LIDC-IDRI |
| Cohort | 500 CT scans |
| Unique patients | 498 |
| Input representation | 2D axial CT slices |
| Image resolution | 256 × 256 |
| Channels | 1 |
| Segmentation model | 2D U-Net |
| Ground truth | PyLIDC consensus masks |
| Partitioning | Patient-level |
| Split ratio | Approximately 70% / 15% / 15% |
| Training batch size | 8 |
| Positive training fraction | 0.50 |
| Training steps per epoch | 400 |
| Loss | Focal Tversky loss |
| Optimizer | Adam |
| Threshold selection | Validation set only |
| Primary evaluation population | Nodule-positive test slices |
| Additional evaluation | Full held-out test set |
| Sensitivity runs | Multiple run seeds |
| Reference run | Seed 42 |

The reference run uses:

```text
COHORT_SEED = 42
RUN_SEED = 42
```

Additional seed-sensitivity experiments use:

```text
RUN_SEED = 1
RUN_SEED = 2
```

The cohort itself remains fixed while the patient partition, TensorFlow/Python/NumPy random state, model initialization, and balanced sampling sequence vary according to the run seed.



## Experimental Design

### Fixed Cohort

A reproducible cohort of 500 scans is selected from LIDC-IDRI using:

```text
COHORT_SEED = 42
```

The cohort contains **498 unique patients**, because two patients contribute two scans each.

The cohort-selection seed is held fixed for all experiments.

### Patient-Level Partitioning

Data are partitioned at the **patient level** rather than at the slice level.

This prevents slices from the same patient from appearing in multiple experimental partitions.

The approximate allocation is:

```text
Training:   70%
Validation: 15%
Testing:    15%
```

Patient assignments are generated using the run-specific seed.

### Seed-Sensitivity Analysis

The primary/reference experiment uses:

```text
RUN_SEED = 42
```

Two additional experiments use:

```text
RUN_SEED = 1
RUN_SEED = 2
```

These runs are used to examine the sensitivity of the baseline to stochastic variation in:

- patient partitioning
- model initialization
- random-number generation
- balanced batch construction
- training dynamics

Each seed is stored in a separate output directory to prevent accidental overwriting of results.



## Ground-Truth Mask Generation

Radiologist annotations are processed using **PyLIDC**.

Annotations corresponding to the same physical nodule are grouped into nodule clusters, and consensus masks are generated from the available radiologist annotations.

A slice is classified as **nodule-positive** when its consensus segmentation mask contains at least one foreground pixel.

This produces two training pools:

```text
positive slices
negative slices
```

The distinction is used for balanced training batch construction.



## Balanced Training Strategy

Lung nodule segmentation presents substantial foreground/background imbalance because the majority of CT slices contain no nodule pixels.

To reduce domination of training by nodule-free slices, the training pipeline uses a custom balanced Keras sequence:

```python
BalancedSegmentationSequence
```

With the default configuration:

```text
BATCH_SIZE = 8
TRAIN_POSITIVE_FRACTION = 0.50
TRAIN_STEPS_PER_EPOCH = 400
```

each batch contains approximately:

```text
4 positive slices
4 negative slices
```

Therefore, each epoch processes:

```text
400 batches × 8 slices = 3,200 sampled training slices
```

Sampling is performed from the complete training positive and negative pools rather than permanently discarding negative slices.



## Loss Function

The model is optimized using **Focal Tversky loss**, which is designed to address severe class imbalance and permits explicit control of false-positive and false-negative penalties.

The Tversky index is defined as:

```text
TI = (TP + ε) / (TP + αFP + βFN + ε)
```

and the Focal Tversky loss as:

```text
L = (1 - TI)^γ
```

The experimental configuration uses:

```text
α = 0.30
β = 0.70
γ = 0.75
```

The larger penalty assigned to false negatives reflects the importance of detecting small nodule regions.



## Validation-Based Threshold Selection

The probability threshold used to convert predicted probability maps into binary segmentation masks is **not selected using the test set**.

Instead, candidate thresholds are evaluated exclusively on the validation partition.

The selected threshold is then frozen and applied once to the held-out test set.

This separation prevents test-set information from influencing model or threshold selection.

Threshold-selection results for each seed are saved as reproducibility artifacts.



## Evaluation Protocol

Evaluation distinguishes between two complementary populations.

### Primary Evaluation — Nodule-Positive Slices

Segmentation quality is evaluated on held-out test slices containing reference nodule pixels.

Metrics include:

- Dice coefficient
- Intersection over Union (IoU)
- Precision
- Recall
- Specificity
- Detection rate

This evaluation focuses directly on slices for which a lesion is present and segmentation is clinically meaningful.

### Full Test-Set Evaluation

The model is also evaluated across the complete held-out test partition, including nodule-free slices.

This analysis is used to characterize background behavior and false-positive predictions.

### Patient-Clustered Uncertainty

Because multiple slices may originate from the same patient, slices cannot be assumed to be statistically independent.

Where confidence intervals are reported, bootstrap resampling is therefore performed at the **patient level**, preserving within-patient clustering.



## Repository Structure

```text
UNet-LIDC-Segmentation/
│
├── data/
│   ├── README.md
│   └── examples/
│       └── anonymized PNG examples
│
├── notebooks/
│   ├── 01_Preprocessing.ipynb
│   ├── 02_UNet_Training.ipynb
│   ├── 03_UNet_Evaluation.ipynb
│   ├── 01B_Seed_Sensitivity_Splits.ipynb
│   ├── 02B_Seed_Sensitivity_Training.ipynb
│   └── 03B_Seed_Sensitivity_Evaluation.ipynb
│
├── metadata/
│   ├── cohort/
│   ├── seed1/
│   ├── seed2/
│   └── seed42/
│
├── splits/
│   ├── seed1/
│   ├── seed2/
│   └── seed42/
│
├── results/
│   ├── seed1/
│   ├── seed2/
│   └── seed42/
│
├── models/
│   ├── seed1/
│   ├── seed2/
│   └── seed42/
│
├── figures/
│   ├── seed1/
│   ├── seed2/
│   └── seed42/
│
├── src/
│   ├── dataloader.py
│   ├── model_unet.py
│   ├── metrics.py
│   └── train.py
│
├── requirements.txt
├── LICENSE
├── CITATION.cff
└── README.md
```

Large imaging data and derived NumPy arrays are intentionally excluded from version control.



## Installation

### Option A — Google Colab

The notebooks can be adapted for execution in Google Colab after mounting or otherwise providing access to a local copy of the LIDC-IDRI dataset.

### Option B — Local Installation

Clone the repository:

```bash
git clone https://github.com/carloshachi777/UNet-LIDC-Segmentation.git
cd UNet-LIDC-Segmentation
```

Create and activate a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The definitive experiments reported in the revised manuscript were executed using:

```text
TensorFlow/Keras 2.21.0
NumPy 1.26.4
```

on a local Apple Silicon workstation.



## Dataset Access — LIDC-IDRI

The LIDC-IDRI imaging data are publicly available through **The Cancer Imaging Archive (TCIA)**.

**This repository does not redistribute LIDC-IDRI DICOM files.**

To reproduce the experiments:

1. Obtain the LIDC-IDRI collection from TCIA.
2. Download the collection using the NBIA Data Retriever or another TCIA-supported method.
3. Preserve the original patient/scan directory organization.
4. Configure the local dataset path in the preprocessing notebook.
5. Run the cohort construction and preprocessing pipeline.

A typical local structure is:

```text
data/
└── LIDC-IDRI/
    ├── LIDC-IDRI-0001/
    ├── LIDC-IDRI-0002/
    ├── LIDC-IDRI-0003/
    └── ...
```

The raw imaging data should remain local and should **not** be committed to GitHub.



## Reproducing the Reference Experiment

### Step 1 — Preprocessing

Run:

```text
notebooks/01_Preprocessing.ipynb
```

The preprocessing pipeline performs:

- fixed cohort construction
- PyLIDC annotation processing
- nodule clustering
- consensus mask generation
- CT slice preprocessing
- resizing to 256 × 256
- slice-level metadata generation
- identification of positive and negative slices



### Step 2 — Reference Training Run

Run:

```text
notebooks/02_UNet_Training.ipynb
```

with:

```python
COHORT_SEED = 42
RUN_SEED = 42
```

The notebook performs:

- patient-level partitioning
- balanced training batch generation
- 2D U-Net construction
- Focal Tversky optimization
- model checkpointing
- learning-rate reduction
- early stopping
- training log generation

Seed-specific outputs are written to directories such as:

```text
models/seed42/
results/seed42/
figures/seed42/
```



### Step 3 — Evaluation

Run:

```text
notebooks/03_UNet_Evaluation.ipynb
```

The evaluation pipeline performs:

- validation threshold selection
- threshold freezing
- positive-test-slice evaluation
- full-test-set evaluation
- Dice computation
- IoU computation
- precision and recall computation
- specificity computation
- detection-rate analysis
- patient-clustered bootstrap analysis
- qualitative visualization



## Reproducing the Seed-Sensitivity Experiments

The seed-sensitivity analysis is implemented using:

```text
01B_Seed_Sensitivity_Splits.ipynb
02B_Seed_Sensitivity_Training.ipynb
03B_Seed_Sensitivity_Evaluation.ipynb
```

Run the pipeline separately with:

```python
RUN_SEED = 1
```

and:

```python
RUN_SEED = 2
```

The cohort-selection seed remains:

```python
COHORT_SEED = 42
```

Outputs should remain isolated by seed:

```text
results/seed1/
results/seed2/
results/seed42/

models/seed1/
models/seed2/
models/seed42/
```

This prevents results from one experimental run from overwriting another.


## Reproducibility Artifacts

The repository is intended to provide the non-imaging artifacts required to reproduce and audit the experiments.

These may include:

- fixed cohort identifiers
- patient-level partition assignments
- slice-level indices
- positive/negative slice labels
- training logs
- validation threshold sweeps
- selected thresholds
- per-slice test metrics
- patient identifiers associated with evaluation records
- bootstrap summaries
- seed-specific results

Raw LIDC-IDRI DICOM images are **not** included.



## Example Images

The directory:

```text
data/examples/
```

may contain anonymized PNG representations of CT slices and segmentation outputs for demonstration.

Examples can include:

```text
ten_random_slices.png
prediction_examples.png
```

These images contain no DICOM headers or DICOM metadata.



## Important Interpretation Note

This repository implements a **baseline research pipeline**, not a clinical diagnostic system.

The model is intentionally based on a conventional 2D U-Net architecture to provide an interpretable and reproducible reference point.

Performance can vary substantially across patient partitions and random seeds. The seed-sensitivity experiments are therefore an important component of the study rather than merely a software validation step.

In particular, the experiments illustrate that apparently strong validation performance does not necessarily imply stable generalization across alternative patient partitions.



## Data and Privacy

The repository does not contain:

- original LIDC-IDRI DICOM files
- protected health information
- original DICOM metadata
- independently identifiable patient information

LIDC-IDRI data should be obtained directly from TCIA and handled according to the terms associated with the dataset.



## Citation

If you use this repository, please cite the associated manuscript when available.

Please also cite the original LIDC-IDRI dataset publication:

> Armato SG III, McLennan G, Bidaut L, et al.  
> The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI):  
> A completed reference database of lung nodules on CT scans.  
> *Medical Physics*. 2011;38(2):915–931.

Citation metadata for this repository are also provided in:

```text
CITATION.cff
```



## License

This project is released under the **MIT License**.

You may use, modify, and distribute the source code in accordance with the terms of the license.



## Acknowledgments

This project uses:

- **LIDC-IDRI** for thoracic CT imaging data and expert annotations
- **The Cancer Imaging Archive (TCIA)** for public data distribution
- **PyLIDC** for accessing and processing LIDC-IDRI annotations
- **TensorFlow/Keras** for deep-learning model development
- **NumPy, pandas, SciPy, and related Python scientific-computing libraries** for data processing and statistical analysis

We thank the LIDC-IDRI investigators, participating radiologists, and TCIA for making this research resource publicly available.



## Research Purpose

This repository is provided for **research and educational purposes only**.

The segmentation model has not been validated for clinical use and should not be used for diagnosis, treatment planning, or other clinical decision-making.

