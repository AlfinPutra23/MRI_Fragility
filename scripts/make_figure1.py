"""Publication Figure 1 (paper-style): (a) real-image forward pipeline; (b) predict-from-anatomy law chain.
Helvetica-style font, large readable type, bold arrows, (a)/(b) panel labels, tight layout, short central connector.
  python make_figure1.py  ->  outputs/plots/figure1.png
"""
import os, sys, numpy as np
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
sys.path.insert(0, "scripts" if os.path.isdir("scripts") else ".")
from kspace import vd_cartesian_mask, undersample_slice
from make_schematics_visual import pick_slice, norm, kspace_log, overlay

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": "#17181c",
})
INK, SLATE, CORAL = "#15171c", "#2f4058", "#a83226"
LAW = ["#f4ddd6", "#eab6a6", "#df8a73", "#cf5b43", "#a83226"]     # light -> deep

def box(ax, x, y, w, h, title, sub, fc, tc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.5",
                                fc=fc, ec="white", lw=2, mutation_aspect=0.55))
    ax.text(x + w/2, y + h*0.62, title, ha="center", va="center", fontsize=15, fontweight="bold", color=tc)
    if sub: ax.text(x + w/2, y + h*0.30, sub, ha="center", va="center", fontsize=11.5, color=tc)

def arrow(host, x0, y0, x1, y1, color=INK, lw=4.0, tf=None):
    kw = dict(arrowstyle="-|>,head_length=10,head_width=6.5", lw=lw, color=color,
              shrinkA=0, shrinkB=0, capstyle="round", joinstyle="round")
    if tf is not None: kw["transform"] = tf
    (host.add_artist if hasattr(host, "add_artist") else host.add_patch)(FancyArrowPatch((x0, y0), (x1, y1), **kw))

def main():
    img, lab = pick_slice(); n = img.shape[0]; R = 8
    m = vd_cartesian_mask(n, R); ali = undersample_slice(img, m, pe_axis=0)
    Km = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(img)) * m[:, None]))

    fig = plt.figure(figsize=(16, 9.3), facecolor="white")
    gs = fig.add_gridspec(2, 5, height_ratios=[1.0, 0.80], hspace=0.40, wspace=0.06,
                          left=0.022, right=0.985, top=0.80, bottom=0.055)
    axt = [fig.add_subplot(gs[0, i]) for i in range(5)]
    panels = [(norm(img), "gray", "Clean MRI", "full scan"),
              (kspace_log(img), "magma", "k-space", "Fourier domain"),
              (Km, "magma", "Undersample 8×", "keep 32 of 256 lines"),
              (norm(ali), "gray", "Fast image", "8× · aliased"),
              (None, None, "Segmentation", "per-organ Dice")]
    for i, (im, cmap, t, s) in enumerate(panels):
        a = axt[i]
        if im is not None: a.imshow(im, cmap=cmap)
        else: overlay(a, norm(img), lab)
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values(): sp.set_edgecolor("#b9bdc4"); sp.set_linewidth(1.2)
        a.set_title(t, fontsize=16, fontweight="bold", pad=8)
        a.text(0.5, -0.055, s, transform=a.transAxes, ha="center", va="top", fontsize=12.5, color="#5c6470")
    for i in range(4):
        p0, p1 = axt[i].get_position(), axt[i+1].get_position(); yc = (p0.y0 + p0.y1)/2
        arrow(fig, p0.x1 + .002, yc, p1.x0 - .002, yc, tf=fig.transFigure, lw=4.2, color="#111")

    # (a) panel label
    ty1 = max(a.get_position().y1 for a in axt); tx0 = axt[0].get_position().x0
    fig.text(tx0, ty1 + 0.075, "a", fontsize=21, fontweight="bold", color="#111")
    fig.text(tx0 + 0.022, ty1 + 0.088, "MEASURE", fontsize=15.5, fontweight="bold", color=SLATE, va="center")
    fig.text(tx0 + 0.112, ty1 + 0.088, "how much each organ breaks as the scan speeds up  (R: 1 → 8)",
             fontsize=13.5, color="#5c6470", va="center")

    # ---- (b) predict / law chain ----
    axb = fig.add_subplot(gs[1, :]); axb.set_xlim(0, 100); axb.set_ylim(0, 26); axb.axis("off")
    steps = [("Organ anatomy", "shape of one organ", LAW[0], "#3a2a26"),
             ("Spectral centroid", "how fine its detail is", LAW[1], "#3a2a26"),
             ("Fragility law", "r = 0.84  (5-fold)", LAW[2], "white"),
             ("Safe limit  R*", "per organ", LAW[3], "white"),
             ("Flag fragile organs", "adrenals · gallbladder", LAW[4], "white")]
    xb = [1.5, 21, 40.5, 61.5, 82]; wb = [17, 17.5, 19, 18, 16.5]; by, bh = 3, 16
    for (t, s, c, tc), x, w in zip(steps, xb, wb):
        box(axb, x, by, w, bh, t, s, c, tc)
    for i in range(len(xb) - 1):
        arrow(axb, xb[i] + wb[i], by + bh/2, xb[i+1], by + bh/2, color=CORAL, lw=4.2)

    pb = axb.get_position(); box_top = pb.y0 + (by + bh)/26 * pb.height; bx0 = pb.x0 + xb[0]/100 * pb.width
    fig.text(bx0, box_top + 0.062, "b", fontsize=21, fontweight="bold", color="#111")
    fig.text(bx0 + 0.022, box_top + 0.073, "PREDICT", fontsize=15.5, fontweight="bold", color=CORAL, va="center")
    fig.text(bx0 + 0.108, box_top + 0.073, "the same fragility — from anatomy alone, no fast scan",
             fontsize=13.5, color="#8a5a50", va="center")

    # short central connector between the two rows
    ty0 = min(a.get_position().y0 for a in axt)
    arrow(fig, 0.5, ty0 - 0.052, 0.5, box_top + 0.052, tf=fig.transFigure, color="#7f8894", lw=3.4)
    fig.text(0.517, (ty0 - 0.052 + box_top + 0.052)/2, "predict\na priori", fontsize=12.5, style="italic",
             color="#636b78", va="center", ha="left")

    fig.suptitle("Predicting which organs break under fast MRI — and their safe scan speed",
                 fontsize=19, fontweight="bold", y=0.965)
    fig.savefig("outputs/plots/figure1.png", dpi=175, bbox_inches="tight", facecolor="white")
    print("wrote outputs/plots/figure1.png")

if __name__ == "__main__":
    main()
