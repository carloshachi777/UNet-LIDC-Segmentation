"""Path resolution shared by the revision-analysis scripts.

Two directory layouts exist in the wild and both are supported:

  repository layout      metadata/seed42/slice_level_index_seed42.csv
                         results/seed42/positive_test_per_slice_metrics.csv

  working layout         metadata/slice_level_index_seed42.csv
                         results/positive_test_per_slice_metrics.csv

The reference run (seed 42) is nested in a clone but sits at the top level in
the authoring working tree. Seeds 1 and 2 are nested in both. Every lookup below
tries the nested form first and falls back to the flat one, so the scripts work
unchanged in either.
"""

from pathlib import Path


def find_project_root(start=None):
    """Locate the repository root by its marker directories."""
    here = Path(start or __file__).resolve()
    for candidate in [here.parent.parent, *here.parents, Path.cwd(), *Path.cwd().parents]:
        if (candidate / "metadata").is_dir() and (candidate / "results").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate the repository root. Run this script from inside a clone."
    )


PROJECT = find_project_root()
OUTPUT = PROJECT / "revision_analysis"


def _first_existing(*candidates, description=""):
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find {description or 'file'}. Looked in:\n  "
        + "\n  ".join(str(c) for c in candidates)
    )


def results_dir(seed):
    """Directory holding one run's result files."""
    return _first_existing(
        PROJECT / "results" / f"seed{seed}",
        PROJECT / "results",
        description=f"results directory for seed {seed}",
    )


def slice_index(seed):
    """Slice-level index CSV for one run."""
    return _first_existing(
        PROJECT / "metadata" / f"seed{seed}" / f"slice_level_index_seed{seed}.csv",
        PROJECT / "metadata" / f"slice_level_index_seed{seed}.csv",
        description=f"slice-level index for seed {seed}",
    )


def metadata_file(name):
    """A cohort-level metadata file, which may sit at metadata/ or metadata/cohort/."""
    return _first_existing(
        PROJECT / "metadata" / name,
        PROJECT / "metadata" / "cohort" / name,
        description=f"metadata/{name}",
    )


def model_path(seed, filename="best_plos_2d_unet_focal_tversky.keras"):
    """Trained checkpoint for one run. Not redistributed; see REPRODUCING.md."""
    return _first_existing(
        PROJECT / "models" / f"seed{seed}" / filename,
        PROJECT / "models" / filename,
        description=f"model checkpoint for seed {seed}",
    )


def per_slice_metrics(seed):
    return results_dir(seed) / "positive_test_per_slice_metrics.csv"


def selected_threshold(seed):
    return results_dir(seed) / "selected_prediction_threshold.json"
