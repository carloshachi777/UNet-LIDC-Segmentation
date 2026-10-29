"""Multi-seed stability analysis: seeds 42, 1, 2."""
import pandas as pd, numpy as np, json, statistics as st
from pathlib import Path

from pathlib import Path

from _paths import PROJECT, OUTPUT, results_dir, slice_index, metadata_file, per_slice_metrics, selected_threshold, model_path


R = PROJECT
SEEDS = {seed: (results_dir(seed), slice_index(seed)) for seed in (42, 1, 2)}
RUNAWAY = 10000
out = {}

def boot(df, B=2000, seed=42):
    rng = np.random.default_rng(seed)
    pats = df.patient_id.unique()
    grp = {p: df[df.patient_id == p] for p in pats}
    res = {k: [] for k in ["dice", "iou", "precision", "recall", "detected"]}
    for _ in range(B):
        idx = rng.integers(0, len(pats), len(pats))
        s = pd.concat([grp[pats[i]] for i in idx])
        for k in res:
            res[k].append(s[k].mean())
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in res.items()}

per_seed = {}
for s, (d, idxf) in SEEDS.items():
    m = pd.read_csv(d / "positive_test_per_slice_metrics.csv")
    idx = pd.read_csv(idxf)
    meta = idx[(idx.split == "test") & (idx.is_positive.astype(bool))].reset_index(drop=True)
    assert len(meta) == len(m), (s, len(meta), len(m))
    df = pd.concat([meta[["patient_id", "scan_id", "mask_pixels"]],
                    m.drop(columns=[c for c in ["slice_index"] if c in m])], axis=1)
    thr = json.load(open(d / "selected_prediction_threshold.json"))
    sl = idx.groupby("split").agg(slices=("is_positive", "size"),
                                  pos=("is_positive", "sum"),
                                  patients=("patient_id", "nunique"))
    runaway = df[df.predicted_pixels > RUNAWAY]
    aff = sorted(runaway.patient_id.unique())
    clean = df[~df.patient_id.isin(aff)]
    TP, FP, FN = df.true_positive_pixels.sum(), df.false_positive_pixels.sum(), df.false_negative_pixels.sum()
    per_seed[s] = {
        "threshold": thr["selected_threshold"],
        "val_mean_dice": thr["validation_mean_dice"],
        "test_patients": int(df.patient_id.nunique()),
        "test_pos_slices": int(len(df)),
        "train_patients": int(sl.loc["train", "patients"]), "val_patients": int(sl.loc["validation", "patients"]),
        "mean_dice": float(df.dice.mean()), "median_dice": float(df.dice.median()),
        "mean_iou": float(df.iou.mean()), "mean_precision": float(df.precision.mean()),
        "mean_recall": float(df.recall.mean()), "detection_rate": float(df.detected.mean()),
        "global_dice": float(2 * TP / (2 * TP + FP + FN)),
        "patient_mean_dice_sd": float(df.groupby("patient_id").dice.mean().std()),
        "runaway_slices": int(len(runaway)),
        "runaway_patients": aff,
        "n_runaway_patients": len(aff),
        "slices_from_affected": int(df.patient_id.isin(aff).sum()),
        "mean_dice_affected": float(df[df.patient_id.isin(aff)].dice.mean()) if aff else None,
        "mean_dice_excl_affected": float(clean.dice.mean()),
        "max_predicted_px": int(df.predicted_pixels.max()),
        "boot": boot(df),
    }
    print(f"seed {s}: dice {per_seed[s]['mean_dice']:.4f} "
          f"CI {per_seed[s]['boot']['dice'][0]:.4f}-{per_seed[s]['boot']['dice'][1]:.4f} | "
          f"{per_seed[s]['test_patients']} pts, {per_seed[s]['test_pos_slices']} slices | "
          f"runaway {per_seed[s]['runaway_slices']} slices / {per_seed[s]['n_runaway_patients']} pts")

agg = {}
for k in ["mean_dice", "median_dice", "mean_iou", "mean_precision", "mean_recall", "detection_rate", "global_dice"]:
    v = [per_seed[s][k] for s in [42, 1, 2]]
    agg[k] = {"values": v, "mean": st.mean(v), "sd": st.stdev(v), "min": min(v), "max": max(v), "range": max(v) - min(v)}
    print(f"{k:16s} mean {agg[k]['mean']:.4f}  SD {agg[k]['sd']:.4f}  range {agg[k]['min']:.4f}-{agg[k]['max']:.4f}")

json.dump({"per_seed": per_seed, "across_seeds": agg}, open(OUTPUT / "multiseed_summary.json", "w"), indent=2)
print("\nsaved multiseed_summary.json")
