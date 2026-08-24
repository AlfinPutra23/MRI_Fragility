"""Render qualitative knee-segmentation overlays from knee_samples.npz (base-anaconda: has matplotlib).
Per case: rows = clean/R4/R8, cols = [input image, ground truth, our prediction], cropped to the knee, colored by structure."""
import numpy as np, matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

LAB = {1: "patellar", 2: "femoral", 3: "tibial-med", 4: "tibial-lat", 5: "menisc-med", 6: "menisc-lat"}
COLS = ["none", "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#00ced1"]
cmap = ListedColormap(COLS)
S = np.load("outputs/results/knee_samples.npz", allow_pickle=True)["samples"]
conds = ["clean", "R4", "R8"]


def bbox(gt, margin=40):
    ys, xs = np.where(gt > 0)
    if len(ys) == 0:
        return slice(0, gt.shape[0]), slice(0, gt.shape[1])
    y0, y1 = max(0, ys.min() - margin), min(gt.shape[0], ys.max() + margin)
    x0, x1 = max(0, xs.min() - margin), min(gt.shape[1], xs.max() + margin)
    return slice(y0, y1), slice(x0, x1)


for r in S:
    sy, sx = bbox(r["gt"])
    fig, ax = plt.subplots(len(conds), 3, figsize=(9.5, 3.4 * len(conds)))
    for i, cond in enumerate(conds):
        img = r[f"img_{cond}"].astype(np.float32)[sy, sx]
        gt = r["gt"][sy, sx]; pred = r[f"pred_{cond}"][sy, sx]
        for jc, (title, seg) in enumerate([("input", None), ("ground truth", gt), ("our prediction", pred)]):
            a = ax[i, jc]; a.imshow(img, cmap="gray"); a.axis("off")
            if seg is not None:
                a.imshow(np.ma.masked_where(seg == 0, seg), cmap=cmap, vmin=0, vmax=6, alpha=0.6)
            if i == 0:
                a.set_title(title, fontsize=12, fontweight="bold")
            if jc == 0:
                a.text(-0.06, 0.5, cond, transform=a.transAxes, rotation=90, va="center", ha="center", fontsize=13, fontweight="bold")
    handles = [Patch(facecolor=COLS[k], label=LAB[k]) for k in LAB]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"SKM-TEA knee {r['cid']}: segmentation degrades as acceleration removes k-space", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(f"outputs/plots/knee_sample_{r['cid']}.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    print("wrote outputs/plots/knee_sample_" + r["cid"] + ".png")
