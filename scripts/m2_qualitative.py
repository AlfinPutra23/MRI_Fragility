"""Qualitative overlays: where fragility-weighting RECOVERS a missed organ vs OVER-segments a good one.
Uses M0 uniform preds (predsTs_R8) vs weighted preds (predsW_R8) at R=8. -> outputs/plots/m2_qualitative.png"""
import numpy as np, nibabel as nib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from paths import PLOTS as OUT_P
import labels as L

D = "../nnUNet_raw/Dataset501_MRIfrag"
# (case, organ-label, short title) — from the per-case audit
EX = [("test_021_DEL", 13, "RECOVERED: adrenal_L  0.00 -> 0.47"),
      ("test_006_VEN", 12, "RECOVERED: adrenal_R  0.13 -> 0.39"),
      ("test_035_ART", 12, "OVER-SEGMENTED: adrenal_R  0.50 -> 0.05")]


def load(p): return np.asanyarray(nib.load(p).dataobj)


def dice(a, b):
    s = a.sum() + b.sum(); return 2*np.logical_and(a, b).sum()/s if s else float("nan")


fig, ax = plt.subplots(len(EX), 3, figsize=(11, 3.4 * len(EX)))
for r, (case, lab, title) in enumerate(EX):
    img = load(f"{D}/imagesTs_R8/{case}_0000.nii.gz").astype(np.float32)
    gt = (load(f"{D}/labelsTs/{case}.nii.gz") == lab)
    pu = (load(f"{D}/predsTs_R8/{case}.nii.gz") == lab)
    pw = (load(f"{D}/predsW_R8/{case}.nii.gz") == lab)
    z = int(gt.sum(axis=(0, 1)).argmax())                         # slice with most organ
    ys, xs = np.where(gt[:, :, z] | pu[:, :, z] | pw[:, :, z])
    if len(ys) == 0:
        ys, xs = np.where(gt[:, :, z]) if gt[:, :, z].any() else (np.array([img.shape[0]//2]), np.array([img.shape[1]//2]))
    m = 40
    y0, y1 = max(ys.min()-m, 0), min(ys.max()+m, img.shape[0]); x0, x1 = max(xs.min()-m, 0), min(xs.max()+m, img.shape[1])
    sl = np.rot90(img[y0:y1, x0:x1, z]); vmax = np.percentile(sl, 99.5)
    panels = [("GT", gt, "#1a9850"), (f"uniform (Dice {dice(pu,gt):.2f})", pu, "#d93025"),
              (f"weighted (Dice {dice(pw,gt):.2f})", pw, "#5b8def")]
    for c, (name, mask, col) in enumerate(panels):
        a = ax[r, c] if len(EX) > 1 else ax[c]
        a.imshow(sl, cmap="gray", vmax=vmax); a.axis("off")
        mm = np.rot90(mask[y0:y1, x0:x1, z])
        if mm.any():
            a.contour(mm, colors=col, linewidths=1.6)
        if r == 0: a.set_title(name, fontsize=10, fontweight="bold")
        if c == 0: a.text(-0.05, 0.5, title, transform=a.transAxes, rotation=90, va="center", ha="right",
                          fontsize=8.5, fontweight="bold", color=("#1a9850" if "RECOV" in title else "#d93025"))
fig.suptitle("Fragility-weighting: recovers organs the uniform model misses (top) but over-segments already-good ones (bottom)",
             fontsize=11.5, fontweight="bold")
fig.tight_layout(rect=[0.02, 0, 1, 0.97])
fig.savefig(f"{OUT_P}/m2_qualitative.png", dpi=140)
print(f"wrote {OUT_P}/m2_qualitative.png")
