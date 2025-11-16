# UNet-LIDC-Segmentation
**Reproducible Baseline Pipeline for Lung Nodule Segmentation using 2D U-Net on LIDC-IDRI**

This repository provides an **open-source, fully reproducible implementation** of a 2D U-Net
segmentation model for detecting lung nodules in CT scans.
It accompanies the manuscript:

**“Automated Segmentation of Lung Nodules in CT Scans Using a U-Net Deep Learning Model:
A Study Based on the LIDC-IDRI Dataset.”**

The primary goal is to provide a **transparent baseline**, not a state-of-the-art model, and
to enable researchers to replicate all steps: preprocessing, mask generation, model training,
evaluation, and visualization.

---

## 🔍 Project Overview

This repository includes:

- **PyLIDC-based consensus mask generation**
- **Train/validation split and dataloader implementation**
- **U-Net model definition** (2D CNN with encoder–decoder architecture)
- **Model training pipeline** with logging
- **Evaluation and visualization notebooks**
- **Examples of anonymized slices** (PNG only, no DICOMs)

This project is aligned with PLOS ONE’s requirements for **code availability and reproducibility**.

---

# 📁 Folder Structure

UNet-LIDC-Segmentation/
│
├── data/
│ ├── README.md ← How to download LIDC-IDRI from TCIA
│ ├── examples/ ← Non-sensitive PNG slices for visualization
│
├── notebooks/
│ ├── preprocessing.ipynb ← PyLIDC consensus masks & numpy exports
│ ├── training_unet.ipynb ← U-Net training and validation
│ ├── evaluation_visualization.ipynb
│
├── src/
│ ├── dataloader.py
│ ├── model_unet.py
│ ├── metrics.py
│ ├── train.py
│
├── requirements.txt ← For pip install
├── LICENSE ← MIT License
└── CITATION.cff ← Citation metadata (optional)


---

## 🛠 Installation

### Option A — Use Google Colab (**recommended**)
Simply open the notebooks in `/notebooks/`, and the environment will install dependencies.

### Option B — Local installation

```bash
git clone https://github.com/carloshachi777/UNet-LIDC-Segmentation.git
cd UNet-LIDC-Segmentation
pip install -r requirements.txt
```

# 📁 Dataset Access (LIDC-IDRI)

This repository does NOT redistribute any DICOM files.
To obtain the LIDC-IDRI dataset:
1. Create a free account on The Cancer Imaging Archive (TCIA)
https://www.cancerimagingarchive.net/collection/lidc-idri/
2. Install the NBIA Data Retriever
3. Download the collection:

LIDC-IDRI/
  ├── LIDC-IDRI-0001
  ├── LIDC-IDRI-0002
  └── ...
4. Place the downloaded folders under:


```bash
UNet-LIDC-Segmentation/data/LIDC-IDRI/
```

5. Run the notebook:

```bash
cd notebooks
jupyter notebook preprocessing.ipynb
```

This generates:
- Preprocessed slices
- PyLIDC consensus masks
- NumPy arrays ready for model training

## ⚠️ Disclaimer
All example images in data/examples/ are **anonymized PNG slices** created for demonstration.
They contain no **DICOM metadata**.

# 🧪 How to Run the Pipeline
## 1. Preprocessing and Mask Generation
Uses PyLIDC to merge radiologists’ annotations:

```bash
notebooks/preprocessing.ipynb
```

Outputs:
- X_train.npy, Y_train.npy
- X_val.npy, Y_val.npy

## 2. Model Training

```bash
notebooks/training_unet.ipynb
```

or command-line:

```bash
cd src
python train.py
```

Outputs:

* trained model (.h5)
* training logs
* loss/metric plots

## 3. Evaluation and Visualization

```bash
notebooks/evaluation_visualization.ipynb
```

Includes:
* Dice coefficient computation
* Prediction overlays
* Error inspection
* Qualitative side-by-side comparisons

# Example Images

The folder:

```bash
data/examples/
```

contains:
* ten_random_slices.png — montage of anonymized CT slices
* optional predicted mask visualizations
These files are safe to share since they do not include any DICOM headers.

# Citation

If you use this repository in your work, please cite the LIDC-IDRI dataset:

Armato SG, McLennan G, Bidaut L, et al.
The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI):
A Completed Reference Database of Lung Nodules on CT Scans.
Medical Physics. 2011;38(2):915–931.

# License

This project is released under the MIT License.
You are free to use, modify, and distribute this code with attribution.

# Acknowledgments

This project uses:
* PyLIDC for consensus annotations
* TCIA for imaging data
* TensorFlow/Keras for deep learning models
We thank the LIDC-IDRI team for making the dataset publicly available.

---

