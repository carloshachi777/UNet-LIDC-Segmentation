import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

from pathlib import Path

from _paths import PROJECT, OUTPUT, results_dir, slice_index, metadata_file, per_slice_metrics, selected_threshold, model_path


R = PROJECT
F = R/"figures"

# ------------------------------------------------------------------
# Figure: corrected U-Net architecture diagram
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6.2))
ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")

levels = [
    ("256 x 256", 32),
    ("128 x 128", 64),
    ("64 x 64", 128),
    ("32 x 32", 256),
]
bott = ("16 x 16", 512)

enc_x, dec_x = 17, 66
y0, dy = 50, 10.5
bw, bh = 17, 6.0

def box(x, y, text, fc):
    p = FancyBboxPatch((x-bw/2, y-bh/2), bw, bh,
                       boxstyle="round,pad=0.35,rounding_size=0.8",
                       linewidth=1.2, edgecolor="black", facecolor=fc)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=8.4)

def arrow(x1, y1, x2, y2, style="-|>", ls="-", lw=1.2, color="black"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=11, linewidth=lw,
                                 linestyle=ls, color=color,
                                 shrinkA=0, shrinkB=0))

enc_c, dec_c, bot_c, io_c = "#dbe8f5", "#e8e2f0", "#f3e2dc", "#e6e6e6"

# input
box(enc_x, y0+7.0, "Input CT slice\n256 x 256 x 1", io_c)
arrow(enc_x, y0+7.0-bh/2, enc_x, y0+bh/2)

enc_y = []
for i, (dim, f) in enumerate(levels):
    y = y0 - i*dy
    enc_y.append(y)
    box(enc_x, y, f"Encoder block {i+1}\n2 x [Conv 3x3 ({f}) + BN + ReLU]\n{dim} x {f}", enc_c)
    if i < len(levels)-1:
        arrow(enc_x, y-bh/2, enc_x, y-dy+bh/2)
        ax.text(enc_x+1.0, y-dy/2, "max-pool 2x2", fontsize=7.2, ha="left", va="center")

yb = y0 - len(levels)*dy
arrow(enc_x, enc_y[-1]-bh/2, enc_x, yb+bh/2)
ax.text(enc_x+1.0, enc_y[-1]-dy/2, "max-pool 2x2", fontsize=7.2, ha="left", va="center")
box((enc_x+dec_x)/2, yb, f"Bottleneck: 2 x [Conv 3x3 ({bott[1]}) + BN + ReLU] + spatial dropout 0.30\n{bott[0]} x {bott[1]}", bot_c)
arrow((enc_x+dec_x)/2+bw/2+8, yb, dec_x, yb+bh/2, style="-|>")

for i, (dim, f) in enumerate(levels):
    y = y0 - i*dy
    box(dec_x, y, f"Decoder block {4-i}\nConv2DTranspose 2x2 ({f}), stride 2 + concat\n2 x [Conv 3x3 ({f}) + BN + ReLU]\n{dim} x {f}", dec_c)
    # skip connection
    arrow(enc_x+bw/2, y, dec_x-bw/2, y, style="-|>", ls=(0,(4,2)), lw=1.0, color="#444444")
    ax.text((enc_x+dec_x)/2, y+2.1, "skip connection (concatenate)", fontsize=7.0,
            ha="center", va="bottom", color="#444444")
    if i > 0:
        arrow(dec_x, y-dy+bh/2, dec_x, y-bh/2)

box(dec_x, y0+7.0, "Output: Conv 1x1 (1), sigmoid\n256 x 256 x 1", io_c)
arrow(dec_x, y0+bh/2, dec_x, y0+7.0-bh/2)

ax.text(50, 60.5, "2D U-Net architecture (4 resolution levels, 7,768,353 parameters)",
        ha="center", va="center", fontsize=11, fontweight="bold")
ax.text(50, 2.0,
        "Spatial dimensions are halved by 2x2 max-pooling in the encoder and restored exactly by\n"
        "2x2 transposed convolutions with stride 2 in the decoder, so encoder and decoder feature\n"
        "maps always match in size and no cropping or padding is required before concatenation.",
        ha="center", va="center", fontsize=8.0)

plt.tight_layout()
plt.savefig(F/"unet_architecture_diagram.png", dpi=300, bbox_inches="tight")
plt.close()
print("saved architecture diagram")

# ------------------------------------------------------------------
# Figure: nodule size distribution in the 500-scan cohort
# ------------------------------------------------------------------
sl = pd.read_csv(R/"metadata/slice_level_index_seed42.csv")
pos = sl[sl.is_positive.astype(bool)]
rep = pd.read_csv(R/"metadata/processed_dataset_report.csv")
rep = rep[rep.status == "OK"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

ax = axes[0]
ax.hist(pos.mask_pixels, bins=np.logspace(0, np.log10(pos.mask_pixels.max()+1), 40),
        edgecolor="black")
ax.set_xscale("log")
ax.axvline(pos.mask_pixels.median(), color="red", linestyle="--", linewidth=2,
           label=f"Median = {pos.mask_pixels.median():.0f} px")
ax.set_xlabel("Consensus mask area per positive slice (pixels, log scale)")
ax.set_ylabel("Number of slices")
ax.set_title("A. Slice-level nodule area")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
data = [pos[pos.split == s].mask_pixels for s in ["train", "validation", "test"]]
ax.boxplot(data, showfliers=False)
ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["Train", "Validation", "Test"])
ax.set_ylabel("Consensus mask area per slice (pixels)")
ax.set_title("B. Nodule area by data split")
ax.grid(alpha=0.3)

ax = axes[2]
ax.hist(rep.positive_slices, bins=30, edgecolor="black")
ax.axvline(rep.positive_slices.median(), color="red", linestyle="--", linewidth=2,
           label=f"Median = {rep.positive_slices.median():.0f} slices")
ax.set_xlabel("Nodule-positive slices per scan")
ax.set_ylabel("Number of scans")
ax.set_title("C. Nodule burden per scan")
ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(F/"cohort_nodule_size_distribution.png", dpi=300, bbox_inches="tight")
plt.close()
print("saved size distribution figure")

# ------------------------------------------------------------------
# Figure: per-patient Dice distribution (bootstrap / variability)
# ------------------------------------------------------------------
pat = pd.read_csv(OUTPUT / "per_patient_test_dice.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
ax = axes[0]
ax.hist(pat.mean_dice, bins=20, edgecolor="black")
ax.axvline(pat.mean_dice.mean(), color="red", linestyle="--", linewidth=2,
           label=f"Mean = {pat.mean_dice.mean():.3f}")
ax.set_xlabel("Mean per-slice Dice within patient")
ax.set_ylabel("Number of test patients")
ax.set_title("A. Between-patient variability (n = 74)")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
m = pd.read_csv(OUTPUT / "test_slice_metrics_with_patient.csv")
rng = np.random.default_rng(42)
pats = m.patient_id.unique()
groups = {p: m[m.patient_id == p].dice.values for p in pats}
boot = [np.concatenate([groups[pats[i]] for i in rng.integers(0, len(pats), len(pats))]).mean()
        for _ in range(2000)]
lo, hi = np.percentile(boot, [2.5, 97.5])
ax.hist(boot, bins=40, edgecolor="black")
ax.axvline(m.dice.mean(), color="red", linestyle="--", linewidth=2,
           label=f"Observed = {m.dice.mean():.3f}")
ax.axvline(lo, color="green", linestyle=":", linewidth=2, label=f"95% CI = [{lo:.3f}, {hi:.3f}]")
ax.axvline(hi, color="green", linestyle=":", linewidth=2)
ax.set_xlabel("Bootstrap mean per-slice Dice")
ax.set_ylabel("Bootstrap replicates")
ax.set_title("B. Patient-clustered bootstrap (2,000 replicates)")
ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(F/"test_dice_variability.png", dpi=300, bbox_inches="tight")
plt.close()
print("saved variability figure", lo, hi)
