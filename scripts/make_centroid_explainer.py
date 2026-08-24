"""Explain the SPECTRAL-CENTROID law with images (parallel to savv_explainer, same organs):
organ crop -> its k-space (where the energy sits) -> radial spectrum with the CENTROID marked (the balance point).
Low centroid = energy near center (coarse/smooth) = robust;  high centroid = energy spread to high freq = fragile.
-> outputs/plots/centroid_explainer.png"""
import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion
sys.path.insert(0, "scripts" if os.path.isdir("scripts") else "."); import labels as L
from make_schematics_visual import norm
from paths import PLOTS as P, DATA

def sav(m): return (m ^ binary_erosion(m)).sum() / max(int(m.sum()), 1)

# same slice/organs as savv_explainer (clearest liver-vs-adrenal contrast)
d = np.load(f"{DATA}/slices/test.npz"); imgs = d["images"].astype(np.float32); labs = d["labels"].astype(np.int64)
best = None
for k in range(0, imgs.shape[0], 2):
    l = labs[k]
    if (l == 5).sum() < 500: continue
    for aid in (12, 13):
        a = int((l == aid).sum())
        if 45 <= a <= 260:
            gap = sav(l == aid) - sav(l == 5)
            if best is None or gap > best[0]: best = (gap, k, aid)
_, k, aid = best; img, lab = imgs[k], labs[k]
lsav, asav = sav(lab == 5), sav(lab == aid)
mids = [(abs(sav(lab == o) - (lsav + asav) / 2), o) for o, nm in L.ABDO.items()
        if o not in (5, aid) and (lab == o).sum() > 220 and lsav < sav(lab == o) < asav]
mids.sort()
picks = [5] + ([mids[0][1]] if mids else []) + [aid]

def crop(m, pad=10):
    ys, xs = np.where(m); return (max(ys.min()-pad, 0), min(ys.max()+pad, m.shape[0]),
                                  max(xs.min()-pad, 0), min(xs.max()+pad, m.shape[1]))

def spectrum(patch):
    n = patch.shape[0]; c = n // 2
    F = np.fft.fftshift(np.fft.fft2(patch)); PS = np.abs(F) ** 2
    y, x = np.mgrid[:n, :n]; rho = np.sqrt((y - c) ** 2 + (x - c) ** 2) / c
    PS[c, c] = 0
    cen = float((rho * PS).sum() / (PS.sum() + 1e-12))
    rb = np.clip((rho * (c)).astype(int), 0, c - 1)
    prof = np.bincount(rb.ravel(), PS.ravel(), minlength=c) / (np.bincount(rb.ravel(), minlength=c) + 1e-9)
    return np.log1p(np.abs(F)), prof[:c] / (prof[:c].max() + 1e-9), cen

fig, ax = plt.subplots(3, 3, figsize=(12, 11))
R = 8
for r, o in enumerate(picks):
    m = lab == o; y0, y1, x0, x1 = crop(m)
    base = norm(img)[y0:y1, x0:x1]
    patch = (img * m)[y0:y1, x0:x1]; sz = max(patch.shape)
    pp = np.zeros((sz, sz), np.float32); pp[:patch.shape[0], :patch.shape[1]] = patch
    klog, prof, cen = spectrum(pp)
    ax[r, 0].imshow(base, cmap="gray"); ax[r, 0].contour(m[y0:y1, x0:x1], colors="#1a9850", linewidths=1.4)
    ax[r, 0].set_ylabel(L.ABDO[o], fontsize=12, fontweight="bold"); ax[r, 0].set_xticks([]); ax[r, 0].set_yticks([])
    if r == 0: ax[r, 0].set_title("the organ", fontsize=11, fontweight="bold")
    ax[r, 1].imshow(klog, cmap="magma"); ax[r, 1].axis("off")
    if r == 0: ax[r, 1].set_title("its k-space\n(bright center = low freq)", fontsize=10.5, fontweight="bold")
    fx = np.linspace(0, 1, len(prof))
    ax[r, 2].fill_between(fx, prof, color="#6a51a3", alpha=.5)
    ax[r, 2].axvline(cen, color="#d93025", lw=2.5, label=f"centroid = {cen:.2f}")
    ax[r, 2].axvline(1 / R, color="#1a9850", ls="--", lw=1.3, label="R8 keeps ← this")
    verdict = "FRAGILE" if cen > 0.15 else ("robust" if cen < 0.10 else "in between")
    vc = "#d93025" if cen > 0.15 else ("#1a9850" if cen < 0.10 else "#e08214")
    ax[r, 2].text(.55, .78, f"centroid = {cen:.2f}\n→ {verdict}", transform=ax[r, 2].transAxes,
                  fontsize=12, fontweight="bold", color=vc)
    ax[r, 2].set_yticks([]); ax[r, 2].set_xlabel("spatial frequency (Nyquist = 1)", fontsize=9)
    ax[r, 2].legend(loc="upper right", fontsize=7.5)
    if r == 0: ax[r, 2].set_title("energy vs frequency\n(red line = the 'balance point')", fontsize=10.5, fontweight="bold")

fig.suptitle("The spectral-centroid law: where does the organ's signal live in frequency?", fontsize=14, fontweight="bold", y=0.995)
fig.text(0.5, 0.012,
         "Centroid = the energy-weighted AVERAGE frequency (the red 'balance point'). Liver's energy hugs the low-freq center (low centroid → robust);\n"
         "the thin adrenal's energy spreads to HIGH frequency (high centroid) — past the R8 cutoff (green line) → that signal is discarded → fragile.",
         ha="center", fontsize=10.3, style="italic")
fig.tight_layout(rect=[0, 0.045, 1, 0.985]); fig.savefig(f"{P}/centroid_explainer.png", dpi=150, bbox_inches="tight")
print(f"wrote centroid_explainer.png  (organs {[L.ABDO[o] for o in picks]}, centroids computed)")
