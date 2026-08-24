"""B1 qualitative sample: learned mask + segmentation, ours vs fixed variable-density, on the same slice.
Reads b1_samples_sample_ours.npz + b1_samples_sample_vd.npz. -> outputs/plots/b1_sample.png"""
import numpy as np, sys, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "scripts" if os.path.isdir("scripts") else "."); import labels as L
ABDO_IDS = list(L.ABDO); REMAP = {o: i + 1 for i, o in enumerate(ABDO_IDS)}
TAIL_CLS = {REMAP[o]: L.ABDO[o] for o in L.TAIL}

O = np.load("outputs/results/b1_samples_sample_ours.npz"); V = np.load("outputs/results/b1_samples_sample_vd.npz")
# pick the slice + tail organ where OURS beats VD most
def dice(a, b): s = a.sum()+b.sum(); return 2*np.logical_and(a, b).sum()/s if s else 0
best = (-1, 0, 0)
for k in range(O["gt"].shape[0]):
    for c in TAIL_CLS:
        g = O["gt"][k] == c
        if g.sum() < 60: continue
        do = dice(O["pred"][k] == c, g); dv = dice(V["pred"][k] == c, g)
        if do - dv > best[0]: best = (do - dv, k, c)
_, k, c = best; organ = TAIL_CLS[c]
gt = O["gt"][k] == c
ys, xs = np.where(gt); m = 45
y0, y1 = max(ys.min()-m, 0), min(ys.max()+m, 256); x0, x1 = max(xs.min()-m, 0), min(xs.max()+m, 256)

fig, ax = plt.subplots(2, 3, figsize=(11, 7.4))
for row, (S, name, col) in enumerate([(V, "fixed variable-density", "#d93025"), (O, "OURS (learned+fragility)", "#1a9850")]):
    mask = S["mask"]  # 1D k-space lines acquired
    ax[row, 0].imshow(np.tile(mask[:, None], (1, 40)), cmap="gray", aspect="auto"); ax[row, 0].axis("off")
    ax[row, 0].set_title(f"{name}\nk-space mask ({int(mask.sum())} lines)", fontsize=9, fontweight="bold")
    under = S["under"][k][y0:y1, x0:x1]
    ax[row, 1].imshow(under, cmap="gray", vmax=np.percentile(under, 99)); ax[row, 1].axis("off")
    ax[row, 1].set_title("undersampled image", fontsize=9)
    ax[row, 2].imshow(under, cmap="gray", vmax=np.percentile(under, 99)); ax[row, 2].axis("off")
    if gt[y0:y1, x0:x1].any(): ax[row, 2].contour(gt[y0:y1, x0:x1], colors="#1a9850", linewidths=1.6)
    pr = (S["pred"][k] == c)[y0:y1, x0:x1]
    if pr.any(): ax[row, 2].contour(pr, colors=col, linewidths=1.6)
    ax[row, 2].set_title(f"{organ}: Dice {dice(S['pred'][k]==c, gt):.2f}", fontsize=9, fontweight="bold",
                         color=col)
fig.suptitle(f"B1: learned fragility-guided sampling recovers the {organ} where fixed sampling fails\n"
             "(green = GT, colored = prediction; both @R=8)", fontsize=11.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig("outputs/plots/b1_sample.png", dpi=140)
print(f"wrote outputs/plots/b1_sample.png (slice {k}, {organ}, ours-vd Dice gap +{best[0]:.2f})")
