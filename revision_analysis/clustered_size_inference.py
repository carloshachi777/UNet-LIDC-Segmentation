"""Patient-clustered inference for the lesion-size analysis (Reviewer 3, point 3).

Slices from the same patient are not independent, so the naive Spearman and
Mann-Whitney p-values are anti-conservative. Here every statistic is recomputed
on bootstrap replicates that resample PATIENTS with replacement, keeping all of
a patient's slices together.
"""
import json
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, mannwhitneyu

from pathlib import Path

from _paths import PROJECT, OUTPUT, results_dir, slice_index, metadata_file, per_slice_metrics, selected_threshold, model_path


ROOT = PROJECT
idx = pd.read_csv(slice_index(42))
met = pd.read_csv(per_slice_metrics(42))
tp = idx[(idx.split == "test") & (idx.is_positive.astype(bool))].reset_index(drop=True)
df = pd.concat([tp[["patient_id"]], met], axis=1)

B, rng = 2000, np.random.default_rng(42)
pats = df.patient_id.unique()
grp = {p: df[df.patient_id == p] for p in pats}
out = {"n_slices": int(len(df)), "n_patients": int(len(pats)), "n_bootstrap": B}

# observed values
rho, p_rho = spearmanr(df.ground_truth_pixels, df.dice)
det, mis = df[df.detected], df[~df.detected]
u, p_u = mannwhitneyu(det.ground_truth_pixels, mis.ground_truth_pixels, alternative="two-sided")
n1, n2 = len(det), len(mis)
rb = 2 * u / (n1 * n2) - 1          # rank-biserial, positive = detected larger
out.update(spearman_rho=float(rho), spearman_p_naive=float(p_rho),
           mw_p_naive=float(p_u), rank_biserial=float(rb),
           median_size_detected=float(det.ground_truth_pixels.median()),
           median_size_missed=float(mis.ground_truth_pixels.median()))

# patient-level means, an independent-unit alternative to the slice-level test
pm = df.groupby("patient_id").agg(mean_area=("ground_truth_pixels", "mean"),
                                  mean_dice=("dice", "mean")).reset_index()
rho_pat, p_pat = spearmanr(pm.mean_area, pm.mean_dice)
out.update(spearman_rho_patient_level=float(rho_pat), spearman_p_patient_level=float(p_pat),
           n_patient_units=int(len(pm)))

boot = {"rho": [], "rb": [], "diff_median": [], "tertile_gap": []}
for _ in range(B):
    s = pd.concat([grp[pats[i]] for i in rng.integers(0, len(pats), len(pats))])
    r = spearmanr(s.ground_truth_pixels, s.dice)[0]
    boot["rho"].append(r)
    d, m = s[s.detected], s[~s.detected]
    if len(d) and len(m):
        uu = mannwhitneyu(d.ground_truth_pixels, m.ground_truth_pixels, alternative="two-sided")[0]
        boot["rb"].append(2 * uu / (len(d) * len(m)) - 1)
        boot["diff_median"].append(d.ground_truth_pixels.median() - m.ground_truth_pixels.median())
    q1, q2 = s.ground_truth_pixels.quantile([1/3, 2/3])
    lo, hi = s[s.ground_truth_pixels <= q1], s[s.ground_truth_pixels > q2]
    boot["tertile_gap"].append(hi.dice.mean() - lo.dice.mean())

for k, v in boot.items():
    v = np.asarray(v)
    out[f"boot_{k}_ci"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    out[f"boot_{k}_median"] = float(np.median(v))
out["boot_rho_excludes_zero"] = bool(out["boot_rho_ci"][0] > 0)
out["boot_rb_excludes_zero"] = bool(out["boot_rb_ci"][0] > 0)
out["boot_tertile_gap_excludes_zero"] = bool(out["boot_tertile_gap_ci"][0] > 0)

json.dump(out, open(OUTPUT / "clustered_size_summary.json", "w"), indent=2)
for k, v in out.items():
    print(f"{k:42s} {v}")
