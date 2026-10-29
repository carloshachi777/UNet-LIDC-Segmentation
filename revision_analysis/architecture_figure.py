import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

from pathlib import Path

from _paths import PROJECT, OUTPUT, results_dir, slice_index, metadata_file, per_slice_metrics, selected_threshold, model_path


F = PROJECT / "figures"

fig, ax = plt.subplots(figsize=(12.5, 7.4))
ax.set_xlim(0, 124); ax.set_ylim(0, 76); ax.axis("off")

enc_c, dec_c, bot_c, io_c = "#dbe8f5", "#ece5f3", "#f6e3da", "#e9e9e9"
BW, BH = 34, 7.2
enc_x, dec_x = 26, 96
y0, dy = 58, 11.5

def box(x, y, text, fc, w=BW, h=BH, fs=8.0):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
                 boxstyle="round,pad=0.3,rounding_size=1.0",
                 lw=1.2, ec="black", fc=fc))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs)

def arr(x1, y1, x2, y2, ls="-", color="black", lw=1.3):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=12, lw=lw, ls=ls, color=color, shrinkA=0, shrinkB=0))

levels = [("256 x 256", 32), ("128 x 128", 64), ("64 x 64", 128), ("32 x 32", 256)]

box(enc_x, y0 + 9.0, "Input CT slice\n256 x 256 x 1", io_c, h=6.2)
arr(enc_x, y0 + 9.0 - 3.1, enc_x, y0 + BH/2)

for i, (dim, f) in enumerate(levels):
    y = y0 - i*dy
    box(enc_x, y, f"Encoder level {i+1}\n2 x [Conv 3x3, {f} filters + BN + ReLU]\nfeature map: {dim} x {f}", enc_c)
    if i < 3:
        arr(enc_x, y - BH/2, enc_x, y - dy + BH/2)
        ax.text(enc_x + 1.5, y - dy/2, "max-pool 2x2", fontsize=7.6, ha="left", va="center")

yb = y0 - 4*dy
arr(enc_x, y0 - 3*dy - BH/2, 60 - 26, yb + 1.5)
ax.text(enc_x + 1.5, y0 - 3*dy - dy/2, "max-pool 2x2", fontsize=7.6, ha="left", va="center")
box(60, yb, "Bottleneck\n2 x [Conv 3x3, 512 filters + BN + ReLU] + spatial dropout 0.30\nfeature map: 16 x 16 x 512", bot_c, w=52, h=6.6, fs=8.0)
arr(60 + 26, yb, dec_x, yb + 4.3)

for i, (dim, f) in enumerate(levels):
    y = y0 - i*dy
    box(dec_x, y, f"Decoder level {i+1}\nConv2DTranspose 2x2, {f} filters, stride 2\nconcatenate skip + 2 x [Conv 3x3, {f} filters + BN + ReLU]\nfeature map: {dim} x {f}", dec_c, w=41, h=8.6, fs=7.1)
    arr(enc_x + BW/2, y + 1.2, dec_x - 20.5, y + 1.2, ls=(0, (5, 3)), color="#3a3a3a", lw=1.0)
    ax.text((enc_x + dec_x)/2, y + 2.4, "skip connection (concatenate)", fontsize=7.6,
            ha="center", va="bottom", color="#3a3a3a")
    if i > 0:
        arr(dec_x, y - dy + 4.3, dec_x, y - 4.3)

box(dec_x, y0 + 9.0, "Output layer\nConv 1x1, 1 filter, sigmoid -> 256 x 256 x 1", io_c, w=38, h=6.2)
arr(dec_x, y0 + 4.3, dec_x, y0 + 9.0 - 3.1)

ax.text(60, 74.0, "2D U-Net: four resolution levels, 7,768,353 parameters (7,762,465 trainable)",
        ha="center", va="center", fontsize=11.5, fontweight="bold")
ax.text(60, 3.2,
        "Spatial dimensions are halved by 2x2 max-pooling in the encoder and restored exactly by 2x2 transposed convolutions with stride 2 in the decoder.\n"
        "Encoder and decoder feature maps therefore always have identical spatial dimensions, and no cropping or padding is required before concatenation.",
        ha="center", va="center", fontsize=8.2)

plt.savefig(F/"unet_architecture_diagram.png", dpi=300, bbox_inches="tight")
print("ok")
