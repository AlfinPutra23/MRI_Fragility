"""Segmentation samples on the 2 abdominal datasets (MRISegmentator + AMOS): clean vs accelerated, per-organ fragility
made visible. Rows = clean / R4 / R8, cols = input | ground truth | nnU-Net prediction. Fragile (tail) organs are drawn
opaque; the prediction visibly loses them as k-space is removed. -> outputs/plots/samples_{mriseg,amos}.png"""
import os, sys, glob, importlib, numpy as np, nibabel as nib
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
sys.path.insert(0, "scripts")
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"]})

DSETS = {
    "mriseg": dict(root="nnUNet_raw/Dataset501_MRIfrag", labmod="labels", title="MRISegmentator-Abdomen"),
    "amos":   dict(root="nnUNet_raw/Dataset502_AMOSfrag", labmod="amos_labels", title="AMOS-MRI"),
}
CONDS = ["clean", "R4", "R8"]
PAL = plt.cm.tab20(np.linspace(0, 1, 20))


def load(p): return np.asanyarray(nib.load(p).dataobj)
def norm(x): p = np.percentile(x, 99.0); return np.clip(x / p, 0, 1) if p > 0 else x


def overlay(ax, img, seg, tail):
    ax.imshow(np.rot90(img), cmap="gray")
    rgba = np.zeros((*seg.shape, 4))
    for lab in np.unique(seg):
        if lab == 0: continue
        c = PAL[int(lab) % 20]; m = seg == lab
        rgba[m] = [c[0], c[1], c[2], 0.75 if lab in tail else 0.40]
    ax.imshow(np.rot90(rgba)); ax.axis("off")


for key, cfg in DSETS.items():
    L = importlib.import_module(cfg["labmod"]); ABDO = L.ABDO; TAIL = set(L.TAIL); root = cfg["root"]
    # pick the test case + slice with the most TAIL-organ voxels (so fragility is visible)
    best = (-1, None, None)
    for gp in sorted(glob.glob(f"{root}/labelsTs/*.nii.gz"))[:40]:
        g = load(gp); ps = np.isin(g, list(TAIL)).sum(axis=(0, 1))
        if ps.max() > best[0]: best = (ps.max(), os.path.basename(gp)[:-7], int(ps.argmax()))
    _, case, z = best
    print(f"{key}: case {case} slice {z}", flush=True)
    fig, ax = plt.subplots(len(CONDS), 3, figsize=(9.5, 3.3 * len(CONDS)))
    gt = load(f"{root}/labelsTs/{case}.nii.gz")[:, :, z]
    ys, xs = np.where(np.isin(gt, list(TAIL) + list(ABDO)))
    sy = slice(max(0, ys.min() - 30), ys.max() + 30); sx = slice(max(0, xs.min() - 30), xs.max() + 30)
    for i, cond in enumerate(CONDS):
        img = norm(load(f"{root}/imagesTs_{cond}/{case}_0000.nii.gz")[:, :, z].astype(np.float32))[sy, sx]
        g = gt[sy, sx]; pr = load(f"{root}/predsTs_{cond}/{case}.nii.gz")[:, :, z][sy, sx]
        for jc, (title, seg) in enumerate([("input", None), ("ground truth", g), ("nnU-Net prediction", pr)]):
            a = ax[i, jc]
            if seg is None: a.imshow(np.rot90(img), cmap="gray"); a.axis("off")
            else: overlay(a, img, seg, TAIL)
            if i == 0: a.set_title(title, fontsize=12, fontweight="bold")
            if jc == 0: a.text(-0.05, 0.5, cond, transform=a.transAxes, rotation=90, va="center", ha="center", fontsize=13, fontweight="bold")
    handles = [Patch(facecolor=PAL[int(o) % 20], label=ABDO[o] + (" *" if o in TAIL else "")) for o in ABDO if o in TAIL]
    fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)), frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02), title="fragile (tail) organs *")
    fig.suptitle(f"{cfg['title']}: fragile organs vanish from the prediction as k-space is removed", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 1]); fig.savefig(f"outputs/plots/samples_{key}.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"wrote outputs/plots/samples_{key}.png", flush=True)
