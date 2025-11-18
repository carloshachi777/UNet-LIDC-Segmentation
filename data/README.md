# LIDC-IDRI Dataset Access

This repository does **not** include DICOM files from LIDC-IDRI.
To reproduce experiments:

1. Create a free TCIA account
   https://www.cancerimagingarchive.net/collection/lidc-idri/
2. Download using the **NBIA Data Retriever**
3. Place folders under:

```text
UNet-LIDC-Segmentation/data/LIDC-IDRI/
├── LIDC-IDRI-0001/
├── LIDC-IDRI-0002/
└── ...
```
---
4. Run:

notebooks/preprocessing.ipynb


This will:
- load DICOMs
- generate PyLIDC consensus masks
- export numpy arrays for training/validation

---

## Example Images

The folder `examples/` contains **non-sensitive PNG slices only**.
These images contain **no metadata** and cannot be used to reconstruct DICOM files.

---

## Citation

LIDC-IDRI must be cited as:

> Armato SG, McLennan G, Bidaut L, et al.
> The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI).
> *Medical Physics*. 2011;38(2):915–931.
