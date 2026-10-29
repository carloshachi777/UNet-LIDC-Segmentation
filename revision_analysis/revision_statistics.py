import numpy as np, pandas as pd, json
from pathlib import Path
from scipy.stats import mannwhitneyu, ks_2samp, spearmanr

from pathlib import Path

from _paths import PROJECT, OUTPUT, results_dir, slice_index, metadata_file, per_slice_metrics, selected_threshold, model_path


R = PROJECT
out = {}

full = pd.read_csv(metadata_file("lidc_annotation_inventory.csv"))
coh  = pd.read_csv(metadata_file("selected_500_scan_cohort_seed42.csv"))
dl   = pd.read_csv(metadata_file("lidc_download_inventory.csv"))

fullok = full[full.status=="OK"]
out["full_inventory_n"] = len(full)
out["full_ok_n"] = len(fullok)
out["full_with_nodules"] = int(fullok.has_nodule_annotations.sum())
out["cohort_n"] = len(coh)

nonsel = fullok[~fullok.patient_id.isin(set(coh.patient_id))]
for name, df in [("cohort",coh),("nonselected",nonsel),("full",fullok)]:
    out[f"{name}_clusters_mean"]=float(df.n_nodule_clusters.mean())
    out[f"{name}_clusters_median"]=float(df.n_nodule_clusters.median())
    out[f"{name}_clusters_iqr"]=[float(df.n_nodule_clusters.quantile(.25)),float(df.n_nodule_clusters.quantile(.75))]
    out[f"{name}_ann_mean"]=float(df.n_annotations.mean())
    out[f"{name}_ann_median"]=float(df.n_annotations.median())

u,p = mannwhitneyu(coh.n_nodule_clusters, nonsel.n_nodule_clusters, alternative="two-sided")
out["mwu_clusters_p"]=float(p); out["mwu_clusters_U"]=float(u)
u2,p2 = mannwhitneyu(coh.n_annotations, nonsel.n_annotations, alternative="two-sided")
out["mwu_annotations_p"]=float(p2)
ks,pks = ks_2samp(coh.n_nodule_clusters, nonsel.n_nodule_clusters)
out["ks_clusters_D"]=float(ks); out["ks_clusters_p"]=float(pks)

# slices per patient from download inventory
dl_c = dl[dl.patient_id.isin(set(coh.patient_id))]
dl_n = dl[~dl.patient_id.isin(set(coh.patient_id))]
out["dicom_cohort_median"]=float(dl_c.dcm_files_with_extension.median())
out["dicom_nonsel_median"]=float(dl_n.dcm_files_with_extension.median())
out["mwu_dicom_p"]=float(mannwhitneyu(dl_c.dcm_files_with_extension, dl_n.dcm_files_with_extension, alternative="two-sided")[1])
out["dicom_total"]=int(dl.dcm_files_with_extension.sum())

# ---- nodule size distribution (positive slices) ----
sl = pd.read_csv(slice_index(42))
pos = sl[sl.is_positive.astype(bool)]
out["n_slices_total"]=int(len(sl)); out["n_pos_slices"]=int(len(pos))
q = pos.mask_pixels.quantile([.05,.25,.5,.75,.95]).to_dict()
out["mask_pixels_quantiles"]={str(k):float(v) for k,v in q.items()}
out["mask_pixels_mean"]=float(pos.mask_pixels.mean())
out["mask_pixels_min"]=int(pos.mask_pixels.min()); out["mask_pixels_max"]=int(pos.mask_pixels.max())
eqd = 2*np.sqrt(pos.mask_pixels/np.pi)
out["eqdiam_px_quantiles"]={str(k):float(v) for k,v in eqd.quantile([.25,.5,.75]).to_dict().items()}
# per split
for s in ["train","validation","test"]:
    ps = pos[pos.split==s]
    out[f"{s}_maskpx_median"]=float(ps.mask_pixels.median())
    out[f"{s}_maskpx_iqr"]=[float(ps.mask_pixels.quantile(.25)),float(ps.mask_pixels.quantile(.75))]
ktr = ks_2samp(pos[pos.split=="train"].mask_pixels, pos[pos.split=="test"].mask_pixels)
out["ks_train_test_size_D"]=float(ktr[0]); out["ks_train_test_size_p"]=float(ktr[1])

# per-scan nodule burden
rep = pd.read_csv(metadata_file("processed_dataset_report.csv"))
repok = rep[rep.status=="OK"]
out["processed_scans"]=int(len(repok))
out["pos_slices_per_scan_median"]=float(repok.positive_slices.median())
out["mask_voxels_per_scan_median"]=float(repok.mask_voxels.median())

# ---- test metrics with patient-clustered bootstrap ----
m = pd.read_csv(per_slice_metrics(42))
meta = sl[(sl.split=="test") & (sl.is_positive.astype(bool))].reset_index(drop=True)
assert len(m)==len(meta), (len(m),len(meta))
m = pd.concat([meta[["patient_id","scan_id","slice_index","mask_pixels"]].reset_index(drop=True),
               m.drop(columns=["slice_index"]).reset_index(drop=True)], axis=1)
out["test_n_slices"]=int(len(m)); out["test_n_patients"]=int(m.patient_id.nunique())
out["test_mean_dice"]=float(m.dice.mean()); out["test_median_dice"]=float(m.dice.median())
out["test_mean_iou"]=float(m.iou.mean()); out["test_mean_prec"]=float(m.precision.mean())
out["test_mean_rec"]=float(m.recall.mean()); out["test_detection_rate"]=float(m.detected.mean())
out["test_dice_iqr"]=[float(m.dice.quantile(.25)),float(m.dice.quantile(.75))]

pat = m.groupby("patient_id").agg(mean_dice=("dice","mean"), n=("dice","size"),
                                  det=("detected","mean")).reset_index()
out["patient_mean_dice_mean"]=float(pat.mean_dice.mean())
out["patient_mean_dice_median"]=float(pat.mean_dice.median())
out["patient_mean_dice_sd"]=float(pat.mean_dice.std())
out["patient_mean_dice_iqr"]=[float(pat.mean_dice.quantile(.25)),float(pat.mean_dice.quantile(.75))]
out["patient_mean_dice_min"]=float(pat.mean_dice.min()); out["patient_mean_dice_max"]=float(pat.mean_dice.max())
out["patient_det_median"]=float(pat.det.median())

rng = np.random.default_rng(42)
pats = m.patient_id.unique()
groups = {p: m[m.patient_id==p] for p in pats}
B=2000
stats={"mean_dice":[], "detection_rate":[], "mean_iou":[], "mean_precision":[], "mean_recall":[],
       "patient_mean_dice":[], "global_dice":[]}
for _ in range(B):
    idx = rng.integers(0,len(pats),len(pats))
    samp = pd.concat([groups[pats[i]] for i in idx])
    stats["mean_dice"].append(samp.dice.mean())
    stats["detection_rate"].append(samp.detected.mean())
    stats["mean_iou"].append(samp.iou.mean())
    stats["mean_precision"].append(samp.precision.mean())
    stats["mean_recall"].append(samp.recall.mean())
    stats["patient_mean_dice"].append(np.mean([groups[pats[i]].dice.mean() for i in idx]))
    tp=samp.true_positive_pixels.sum(); fp=samp.false_positive_pixels.sum(); fn=samp.false_negative_pixels.sum()
    stats["global_dice"].append(2*tp/(2*tp+fp+fn))
for k,v in stats.items():
    lo,hi=np.percentile(v,[2.5,97.5])
    out[f"boot_{k}_ci"]=[float(lo),float(hi)]
    out[f"boot_{k}_sd"]=float(np.std(v))

rho,pv = spearmanr(m.ground_truth_pixels, m.dice)
out["spearman_rho"]=float(rho); out["spearman_p"]=float(pv)
det=m[m.detected==True]; mis=m[m.detected==False]
out["n_detected"]=int(len(det)); out["n_missed"]=int(len(mis))
out["median_size_detected"]=float(det.ground_truth_pixels.median())
out["median_size_missed"]=float(mis.ground_truth_pixels.median())
out["iqr_size_detected"]=[float(det.ground_truth_pixels.quantile(.25)),float(det.ground_truth_pixels.quantile(.75))]
out["iqr_size_missed"]=[float(mis.ground_truth_pixels.quantile(.25)),float(mis.ground_truth_pixels.quantile(.75))]
uu,pp = mannwhitneyu(det.ground_truth_pixels, mis.ground_truth_pixels, alternative="two-sided")
out["mwu_detect_size_U"]=float(uu); out["mwu_detect_size_p"]=float(pp)
# rank-biserial effect size
n1,n2=len(det),len(mis); out["rank_biserial"]=float(1-2*uu/(n1*n2)) if n1*n2 else None

# dice by size tertile
m["size_bin"]=pd.qcut(m.ground_truth_pixels,3,labels=["small","medium","large"])
out["dice_by_size_tertile"]={str(k):{"n":int(v.shape[0]),"median_px":float(v.ground_truth_pixels.median()),
    "mean_dice":float(v.dice.mean()),"det_rate":float(v.detected.mean())} for k,v in m.groupby("size_bin",observed=True)}

# global pixel metrics
TP=int(m.true_positive_pixels.sum()); FP=int(m.false_positive_pixels.sum()); FN=int(m.false_negative_pixels.sum())
TN=len(m)*256*256-TP-FP-FN
out["TP"],out["FP"],out["FN"],out["TN"]=TP,FP,FN,TN
out["global_dice"]=float(2*TP/(2*TP+FP+FN)); out["global_iou"]=float(TP/(TP+FP+FN))
out["global_prec"]=float(TP/(TP+FP)); out["global_rec"]=float(TP/(TP+FN))
out["global_spec"]=float(TN/(TN+FP)); out["global_acc"]=float((TP+TN)/(TP+TN+FP+FN))
# trivial-all-background accuracy baseline
out["all_background_accuracy"]=float((TN+FP)/(len(m)*256*256))

# ---- representativeness within the eligible pool (feeds Table 1) ----
elig_pool = full[full.has_nodule_annotations]
non_elig = elig_pool[~elig_pool.patient_id.isin(set(coh.patient_id))]
out["eligible_pool_n"] = int(len(elig_pool))
out["nonselected_eligible_n"] = int(len(non_elig))
out["elig_nonsel_clusters_median"] = float(non_elig.n_nodule_clusters.median())
out["elig_nonsel_clusters_mean"] = float(non_elig.n_nodule_clusters.mean())
out["elig_nonsel_clusters_iqr"] = [float(non_elig.n_nodule_clusters.quantile(.25)),
                                   float(non_elig.n_nodule_clusters.quantile(.75))]
out["elig_nonsel_ann_median"] = float(non_elig.n_annotations.median())
out["elig_nonsel_ann_mean"] = float(non_elig.n_annotations.mean())
out["mwu_clusters_elig_p"] = float(mannwhitneyu(coh.n_nodule_clusters, non_elig.n_nodule_clusters)[1])
out["mwu_ann_elig_p"] = float(mannwhitneyu(coh.n_annotations, non_elig.n_annotations)[1])
_ks = ks_2samp(coh.n_nodule_clusters, non_elig.n_nodule_clusters)
out["ks_clusters_elig_D"] = float(_ks[0])
out["ks_clusters_elig_p"] = float(_ks[1])
out["rank_biserial_detected_larger"] = abs(out["rank_biserial"])

m.to_csv(OUTPUT / "test_slice_metrics_with_patient.csv", index=False)
pat.to_csv(OUTPUT / "per_patient_test_dice.csv", index=False)
json.dump(out, open(OUTPUT / "analysis_summary.json","w"), indent=2)
for k,v in out.items(): print(k,"=",v)
