"""Explain the law with images: decompose real organs into SURFACE (boundary, red) vs VOLUME (interior, blue).
Surface-to-volume ratio = red / (red+blue). Thin organ = almost all surface (high SA/V) = fragile.
-> outputs/plots/savv_explainer.png"""
import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion
sys.path.insert(0, "scripts" if os.path.isdir("scripts") else "."); import labels as L
from make_schematics_visual import norm
from paths import PLOTS as P, DATA

def sav(m): return (m ^ binary_erosion(m)).sum() / max(int(m.sum()), 1)

# find the slice with the CLEAREST blob-vs-thin contrast: big liver + a small adrenal cross-section
d = np.load(f"{DATA}/slices/test.npz"); imgs = d["images"].astype(np.float32); labs = d["labels"].astype(np.int64)
best = None
for k in range(0, imgs.shape[0], 2):
    l = labs[k]
    if (l == 5).sum() < 500: continue                      # need a big liver blob
    for aid in (12, 13):
        a = int((l == aid).sum())
        if 45 <= a <= 260:                                 # small adrenal cross-section -> thin -> high SA/V
            gap = sav(l == aid) - sav(l == 5)
            if best is None or gap > best[0]: best = (gap, k, aid)
_, k, aid = best; img, lab = imgs[k], labs[k]
lsav, asav = sav(lab == 5), sav(lab == aid)
# a middle organ present with SA/V between liver and adrenal
mids = [(abs(sav(lab == o) - (lsav + asav) / 2), sav(lab == o), o) for o, nm in L.ABDO.items()
        if o not in (5, aid) and (lab == o).sum() > 220 and lsav < sav(lab == o) < asav]
mids.sort()
picks = [(lsav, 5, "liver")] + ([(mids[0][1], mids[0][2], L.ABDO[mids[0][2]])] if mids else []) + [(asav, aid, L.ABDO[aid])]

def crop(m, pad=14):
    ys, xs = np.where(m); return (max(ys.min()-pad, 0), min(ys.max()+pad, m.shape[0]),
                                  max(xs.min()-pad, 0), min(xs.max()+pad, m.shape[1]))

fig, ax = plt.subplots(3, 3, figsize=(11.5, 11))
for r, (sav, o, nm) in enumerate(picks):
    m = lab == o
    core = binary_erosion(m, iterations=2)      # interior (eroded 2px, for visibility)
    shell = m & ~core                            # boundary band (red)
    y0, y1, x0, x1 = crop(m)
    base = norm(img)[y0:y1, x0:x1]; sh = shell[y0:y1, x0:x1]; co = core[y0:y1, x0:x1]
    # col 0: organ on MRI
    ax[r, 0].imshow(base, cmap="gray"); ax[r, 0].contour(m[y0:y1, x0:x1], colors="#1a9850", linewidths=1.4)
    ax[r, 0].set_ylabel(nm, fontsize=12, fontweight="bold"); ax[r, 0].set_xticks([]); ax[r, 0].set_yticks([])
    if r == 0: ax[r, 0].set_title("the organ (green outline)", fontsize=11, fontweight="bold")
    # col 1: surface (red) vs volume (blue)
    ax[r, 1].imshow(base, cmap="gray")
    ov = np.zeros((*base.shape, 4)); ov[co] = (0.15, 0.4, 0.9, 0.75); ov[sh] = (0.9, 0.15, 0.1, 0.85)
    ax[r, 1].imshow(ov); ax[r, 1].axis("off")
    if r == 0: ax[r, 1].set_title("SURFACE (red) vs VOLUME (blue)", fontsize=11, fontweight="bold")
    # col 2: the SA/V ratio as a stacked bar + verdict
    ax[r, 2].barh([0], [sav], color="#d93025", label="surface"); ax[r, 2].barh([0], [1-sav], left=[sav], color="#2c5fb0", label="volume")
    ax[r, 2].set_xlim(0, 1); ax[r, 2].set_yticks([]); ax[r, 2].set_xlabel("fraction of the organ", fontsize=9)
    verdict = "FRAGILE" if sav > 0.25 else ("robust" if sav < 0.13 else "in between")
    vc = "#d93025" if sav > 0.25 else ("#1a9850" if sav < 0.13 else "#e08214")
    ax[r, 2].text(0.5, 0.55, f"SA/V = {sav:.2f}", ha="center", fontsize=15, fontweight="bold", transform=ax[r, 2].transAxes)
    ax[r, 2].text(0.5, 0.15, f"({int(sav*100)}% is boundary)  →  {verdict}", ha="center", fontsize=10,
                  color=vc, fontweight="bold", transform=ax[r, 2].transAxes)
    if r == 0: ax[r, 2].set_title("surface-to-volume ratio", fontsize=11, fontweight="bold")

fig.suptitle("What 'surface-to-volume ratio (SA/V)' means — and why high SA/V = fragile", fontsize=14, fontweight="bold", y=0.995)
fig.text(0.5, 0.012,
         "A blob (liver) is mostly VOLUME (blue) → low SA/V → robust.   A thin organ (adrenal) is almost all SURFACE (red) → high SA/V.\n"
         "Boundaries are FINE DETAIL = high spatial frequency — exactly what fast MRI (acceleration) throws away.  So high SA/V → fragile.",
         ha="center", fontsize=10.5, style="italic")
fig.tight_layout(rect=[0, 0.045, 1, 0.985]); fig.savefig(f"{P}/savv_explainer.png", dpi=150, bbox_inches="tight")
print(f"wrote outputs/plots/savv_explainer.png  (organs: {[p[2] for p in picks]}, SA/V {[round(p[0],2) for p in picks]})")
