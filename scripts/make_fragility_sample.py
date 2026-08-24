"""Visual fragility sample: a small organ (adrenal/gallbladder) collapses clean->R4->R8 while the liver holds.
Uses the M0 model preds (predsTs) on Dataset501. -> outputs/plots/fragility_sample.png"""
import glob, os, numpy as np, nibabel as nib, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "scripts" if os.path.isdir("scripts") else "."); import labels as L
D = "nnUNet_raw/Dataset501_MRIfrag" if os.path.isdir("nnUNet_raw") else "../nnUNet_raw/Dataset501_MRIfrag"
TAGS = [("clean", "R=1 (clean)"), ("R4", "R=4"), ("R8", "R=8")]


def load(p): return np.asanyarray(nib.load(p).dataobj)
def dice(a, b):
    s = a.sum() + b.sum(); return 2 * np.logical_and(a, b).sum() / s if s else float("nan")


# pick the case where the adrenal COLLAPSES most (clean Dice high -> R8 low), liver present
ADR, LIV = 13, 5
case, best_drop = None, -1
for gp in sorted(glob.glob(f"{D}/labelsTs/*.nii.gz"))[:70]:
    g = load(gp); c = os.path.basename(gp)[:-7]
    if (g == ADR).sum() < 250 or (g == LIV).sum() < 5000:
        continue
    pc, p8 = f"{D}/predsTs_clean/{c}.nii.gz", f"{D}/predsTs_R8/{c}.nii.gz"
    if not (os.path.exists(pc) and os.path.exists(p8)):
        continue
    dc = dice(load(pc) == ADR, g == ADR); d8 = dice(load(p8) == ADR, g == ADR)
    if dc > 0.6 and dc - d8 > best_drop:
        best_drop, case = dc - d8, c
g = load(f"{D}/labelsTs/{case}.nii.gz")

fig, ax = plt.subplots(2, 3, figsize=(11.5, 7.6))
for row, (lab, name, color) in enumerate([(ADR, "adrenal (small)", "#d93025"), (LIV, "liver (large)", "#2c5fb0")]):
    z = int((g == lab).sum(axis=(0, 1)).argmax())
    ys, xs = np.where(g[:, :, z] == lab); m = 55
    y0, y1 = max(ys.min()-m, 0), min(ys.max()+m, g.shape[0]); x0, x1 = max(xs.min()-m, 0), min(xs.max()+m, g.shape[1])
    gt = np.rot90((g[:, :, z] == lab)[y0:y1, x0:x1])
    for col, (tag, tlabel) in enumerate(TAGS):
        img = np.rot90(load(f"{D}/imagesTs_{tag}/{case}_0000.nii.gz")[y0:y1, x0:x1, z].astype(np.float32))
        pr = np.rot90((load(f"{D}/predsTs_{tag}/{case}.nii.gz")[:, :, z] == lab)[y0:y1, x0:x1])
        gfull = load(f"{D}/labelsTs/{case}.nii.gz"); pfull = load(f"{D}/predsTs_{tag}/{case}.nii.gz")
        dsc = dice(pfull == lab, gfull == lab)
        a = ax[row, col]; a.imshow(img, cmap="gray", vmax=np.percentile(img, 99.5)); a.axis("off")
        if gt.any(): a.contour(gt, colors="#1a9850", linewidths=1.6)      # GT green
        if pr.any(): a.contour(pr, colors=color, linewidths=1.6)          # pred
        a.set_title(f"{tlabel}   Dice={dsc:.2f}", fontsize=10, fontweight="bold",
                    color="#1a9850" if dsc > 0.7 else "#d93025")
        if col == 0:
            a.text(-0.07, 0.5, name, transform=a.transAxes, rotation=90, va="center", ha="right",
                   fontsize=11, fontweight="bold", color=color)
fig.suptitle("Acceleration breaks small organs but not large ones (green = ground truth, colored = prediction)\n"
             "the adrenal vanishes by R=8 while the liver stays intact — at near-constant image quality",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0.02, 0, 1, 0.93])
out = "outputs/plots/fragility_sample.png" if os.path.isdir("outputs") else "../outputs/plots/fragility_sample.png"
fig.savefig(out, dpi=140); print(f"wrote {out}  (case {case})")
