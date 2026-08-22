## Trained Models

Trained Keras model checkpoints are **not stored directly in this repository** because of their file size.

All reported models can be reproduced from the provided notebooks and metadata. The repository includes the preprocessing pipeline, patient-level data partitions, training configuration, random seeds, validation-based threshold selection, evaluation procedures, and reported results for each experimental run.

### Model checkpoints

Three independent training runs were performed using random seeds **42, 1, and 2**. Seed 42 corresponds to the reference run reported in the main analysis, while seeds 1 and 2 were used for the sensitivity analysis.

When the training notebooks are executed, model checkpoints are saved locally using the following structure:

```text
models/
├── seed42/
│   └── best_plos_2d_unet.keras
├── seed1/
│   └── best_plos_2d_unet.keras
└── seed2/
    └── best_plos_2d_unet.keras
```

The `.keras` checkpoint files are excluded from version control because of their size.

### Reproducing the models

The checkpoints can be regenerated using the training notebooks in:

```text
notebooks/
├── 02_UNet_Training.ipynb
└── 02B_Seed_Sensitivity_Training.ipynb
```

The reference model (**seed 42**) is trained with `02_UNet_Training.ipynb`. The sensitivity runs (**seeds 1 and 2**) are reproduced with `02B_Seed_Sensitivity_Training.ipynb`.

Corresponding configurations, patient-level splits, training logs, evaluation outputs, and summary statistics are provided in the repository to support reproducibility.

> **Note:** Model checkpoints are generated artifacts and are not required to reproduce the reported experiments from the provided code and metadata.
