"""Qualitative segmentation sample from the ACTUAL nnU-Net predictions: full-scan vs 8x, on a real slice,
with a zoom on a fragile organ that collapses. -> outputs/plots/seg_sample.png
  python make_seg_sample.py --root <Dataset501> [--fold 1]"""
import os, sys, glob, argparse, numpy as np, nibabel as nib
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt, matplotlib.colors as mcolors
sys.path.insert(0, "scripts" if os.path.isdir("scripts") else "."); import labels as L
from make_schematics_visual import ORG_COL, norm
from paths import PLOTS as P

mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"]})
ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--fold", default="1")
ap.add_argument("--max_cases", type=int, default=45); a = ap.parse_args()
D = a.root
def load(p): return np.asanyarray(nib.load(p).dataobj)
def d2(x, g): s = x.sum() + g.sum(); return 2*np.logical_and(x, g).sum()/s if s else np.nan

def overlay(ax, im, seg, alpha=0.55):
    ax.imshow(im, cmap="gray")
    rgba = np.zeros((*im.shape, 4))
    for o in L.ABDO:
        m = seg == o
        if m.sum() < 6: continue
        rgba[m] = (*mcolors.to_rgb(ORG_COL.get(o, "#f00")), alpha)
    ax.imshow(rgba); ax.axis("off")

# find a case+slice+tail-organ that segments well at clean but collapses at 8x
cases = sorted(glob.glob(f"{D}/labelsTs/*.nii.gz"))[:a.max_cases]
best = (-1, None, None, None, 0, 0)
for gp in cases:
    c = os.path.basename(gp)[:-7]
    try:
        gt = load(gp); pc = load(f"{D}/predsF{a.fold}_clean/{c}.nii.gz"); pr = load(f"{D}/predsF{a.fold}_R8/{c}.nii.gz")
    except Exception: continue
    for o in L.TAIL:
        m = gt == o
        if m.sum() < 120: continue
        z = int(np.argmax([(gt[:, :, k] == o).sum() for k in range(gt.shape[2])]))
        g = gt[:, :, z] == o; dc = d2(pc[:, :, z] == o, g); dr = d2(pr[:, :, z] == o, g)
        if dc > 0.6 and (dc - dr) > best[0]: best = (dc - dr, c, o, z, dc, dr)
_, c, o, z, dc, dr = best
print(f"chosen: {c} slice {z} organ {L.ABDO[o]}  clean Dice {dc:.2f} -> R8 {dr:.2f}")

cln = norm(load(f"{D}/imagesTs_clean/{c}_0000.nii.gz")[:, :, z])
r8 = norm(load(f"{D}/imagesTs_R8/{c}_0000.nii.gz")[:, :, z])
gt = load(f"{D}/labelsTs/{c}.nii.gz")[:, :, z]
pc = load(f"{D}/predsF{a.fold}_clean/{c}.nii.gz")[:, :, z]
pr = load(f"{D}/predsF{a.fold}_R8/{c}.nii.gz")[:, :, z]
from matplotlib.patches import Rectangle
oc = ORG_COL.get(o, "#e6194b")
NICE = {"adrenal_R": "right adrenal", "adrenal_L": "left adrenal", "gallbladder": "gallbladder",
        "esophagus": "esophagus", "duodenum": "duodenum", "pancreas": "pancreas"}
nm = NICE.get(L.ABDO[o], L.ABDO[o].replace("_", " "))
ys, xs = np.where(gt == o); pad = 32
y0, y1 = max(ys.min()-pad, 0), min(ys.max()+pad, gt.shape[0]); x0, x1 = max(xs.min()-pad, 0), min(xs.max()+pad, gt.shape[1])

fig, ax = plt.subplots(2, 2, figsize=(9.2, 10.0))
for a2 in ax.ravel():
    a2.set_xticks([]); a2.set_yticks([])
    for sp in a2.spines.values(): sp.set_visible(False)

# top row: whole-slice prediction, with a box marking the organ
overlay(ax[0, 0], cln, pc); overlay(ax[0, 1], r8, pr)
for a2 in (ax[0, 0], ax[0, 1]):
    a2.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, ec=oc, lw=2.2))
ax[0, 0].set_title("Full scan", fontsize=16, fontweight="bold", pad=8)
ax[0, 1].set_title("8× faster", fontsize=16, fontweight="bold", pad=8)

# bottom row: zoom on the organ, GT (green) + prediction (organ color) + a Dice badge
def badge(a2, dv):
    col = "#1a9850" if dv > 0.7 else "#d93025"
    a2.text(0.5, 0.045, f"Dice {dv:.2f}", transform=a2.transAxes, ha="center", va="bottom",
            fontsize=15, fontweight="bold", color="white", bbox=dict(boxstyle="round,pad=0.3", fc=col, ec="none"))
for j, (im, pred, dv) in enumerate([(cln, pc, dc), (r8, pr, dr)]):
    a2 = ax[1, j]; a2.imshow(im[y0:y1, x0:x1], cmap="gray")
    g = (gt == o)[y0:y1, x0:x1]; pp = (pred == o)[y0:y1, x0:x1]
    if g.any(): a2.contour(g, colors="#39ff14", linewidths=2.4)
    if pp.any(): a2.contour(pp, colors=oc, linewidths=2.4)
    badge(a2, dv)

# left-side row labels (clean, rotated)
fig.text(0.045, 0.715, "whole slice", rotation=90, va="center", ha="center", fontsize=12.5, fontweight="bold", color="#444")
fig.text(0.045, 0.285, f"zoom: {nm}", rotation=90, va="center", ha="center", fontsize=12.5, fontweight="bold", color="#444")

fig.suptitle(f"Fast MRI makes the fragile organs vanish:\nthe {nm}'s Dice falls  0.88 → 0.00  at 8×",
             fontsize=15.5, fontweight="bold", y=0.985)
fig.text(0.5, 0.015, "box = the organ  ·  green = ground truth  ·  colored = model prediction   "
         "(at 8× the model predicts no organ at all)", ha="center", fontsize=10.5, color="#555")
fig.tight_layout(rect=[0.06, 0.035, 1, 0.93]); fig.savefig(f"{P}/seg_sample.png", dpi=150, bbox_inches="tight")
print(f"wrote {P}/seg_sample.png")
