"""Visual schematics using REAL MRI images (not drawn boxes):
  1. pipeline_visual.png : a real slice through the pipeline -- clean -> k-space -> undersample -> aliased -> segmented
  2. idea_visual.png     : real liver (blob) vs adrenal (thin) crops, clean vs R8, + their real k-space spectra
"""
import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.colors as mcolors
sys.path.insert(0, "scripts" if os.path.isdir("scripts") else "."); import labels as L
from kspace import vd_cartesian_mask, undersample_slice
from paths import PLOTS as P, DATA

def norm(im, p=99.5):
    v = np.percentile(im, p); return np.clip(im, 0, v) / (v + 1e-8)

def kspace_log(im):
    K = np.fft.fftshift(np.fft.fft2(im)); return np.log1p(np.abs(K))

# organ colors (distinct) for overlay
ORG_COL = {5: "#e6194b", 1: "#3cb44b", 2: "#4363d8", 3: "#42d4f4", 4: "#f58231",
           11: "#ffe119", 12: "#911eb4", 13: "#f032e6", 6: "#bfef45", 7: "#fabed4",
           16: "#469990", 17: "#9A6324", 18: "#800000"}

def overlay(ax, img, lab, organs=None, alpha=.55):
    ax.imshow(img, cmap="gray")
    for o in (organs or L.ABDO):
        m = lab == o
        if m.sum() < 8: continue
        rgba = np.zeros((*m.shape, 4)); c = mcolors.to_rgb(ORG_COL.get(o, "#ff0000"))
        rgba[m] = (*c, alpha); ax.imshow(rgba)
    ax.axis("off")

def pick_slice():
    d = np.load(f"{DATA}/slices/test.npz"); imgs = d["images"].astype(np.float32); labs = d["labels"].astype(np.int64)
    best = (-1, 0)
    for k in range(0, imgs.shape[0], 3):
        l = labs[k]
        if (l == 5).sum() > 400 and max((l == 12).sum(), (l == 13).sum()) > 15:   # liver + an adrenal
            score = sum(1 for o in L.ABDO if (l == o).sum() > 20)
            if score > best[0]: best = (score, k)
    return imgs[best[1]], labs[best[1]]

def bbox(m, pad):
    ys, xs = np.where(m); return (max(ys.min()-pad, 0), min(ys.max()+pad, m.shape[0]),
                                  max(xs.min()-pad, 0), min(xs.max()+pad, m.shape[1]))

# ---------------- 1. PIPELINE (real images) ----------------
def pipeline_visual(img, lab):
    n = img.shape[0]; R = 8
    m = vd_cartesian_mask(n, R); ali = undersample_slice(img, m, pe_axis=0)
    K = kspace_log(img)
    Km = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(img)) * m[:, None]))     # undersampled k-space (log-mag)
    fig, ax = plt.subplots(1, 5, figsize=(17, 4.4))
    ax[0].imshow(norm(img), cmap="gray"); ax[0].set_title("1. Clean MRI", fontweight="bold", fontsize=11)
    ax[1].imshow(K, cmap="magma"); ax[1].set_title("2. k-space (FFT)\nenergy in the center", fontsize=10.5, fontweight="bold")
    ax[2].imshow(Km, cmap="magma"); ax[2].set_title(f"3. Undersample ×{R}\n(keep {int(m.sum())}/{n} lines)", fontsize=10.5, fontweight="bold")
    ax[3].imshow(norm(ali), cmap="gray"); ax[3].set_title(f"4. Aliased image @R{R}\n(blurred / ghosted)", fontsize=10.5, fontweight="bold")
    overlay(ax[4], norm(img), lab); ax[4].set_title("5. Per-organ segmentation\n(→ Dice vs GT = fragility)", fontsize=10.5, fontweight="bold")
    for a in ax: a.axis("off")
    for i in range(4):
        fig.add_artist(FancyArrowPatch((ax[i].get_position().x1, .5), (ax[i+1].get_position().x0, .5),
                       transform=fig.transFigure, arrowstyle="-|>", mutation_scale=18, lw=2, color="#333"))
    fig.suptitle("Pipeline on a real abdominal MRI slice:  acquire → undersample → reconstruct → segment → measure per-organ fragility",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(f"{P}/pipeline_visual.png", dpi=150, bbox_inches="tight")
    print("wrote pipeline_visual.png")

# ---------------- 2. IDEA (real liver vs adrenal) ----------------
def radial_profile(patch):
    F = np.fft.fftshift(np.fft.fft2(patch)); PS = np.abs(F) ** 2; nn = patch.shape[0]; c = nn // 2
    y, x = np.mgrid[:nn, :nn]; rr = np.sqrt((y - c) ** 2 + (x - c) ** 2).astype(int)
    tb = np.bincount(rr.ravel(), PS.ravel()); nb = np.bincount(rr.ravel()); prof = tb / (nb + 1e-9)
    prof[0] = 0; return prof[:c] / (prof[:c].max() + 1e-9)

def idea_visual(img, lab):
    n = img.shape[0]; R = 8; m = vd_cartesian_mask(n, R); ali = undersample_slice(img, m, pe_axis=0)
    adr = 12 if (lab == 12).sum() >= (lab == 13).sum() else 13
    rows = [("liver (blob, low centroid → robust)", 5, "#1a9850"), ("adrenal (thin, high centroid → fragile)", adr, "#d93025")]
    fig, ax = plt.subplots(2, 4, figsize=(15, 7.2))
    for r, (name, o, col) in enumerate(rows):
        y0, y1, x0, x1 = bbox(lab == o, 26)
        cl, al = norm(img)[y0:y1, x0:x1], norm(ali)[y0:y1, x0:x1]
        gt = (lab == o)[y0:y1, x0:x1]
        ax[r, 0].imshow(cl, cmap="gray"); ax[r, 0].contour(gt, colors=col, linewidths=1.6); ax[r, 0].axis("off")
        ax[r, 0].set_title(f"{name}\nclean crop", fontsize=9.5, color=col, fontweight="bold")
        ax[r, 1].imshow(al, cmap="gray"); ax[r, 1].contour(gt, colors=col, linewidths=1.6); ax[r, 1].axis("off")
        ax[r, 1].set_title(f"same crop @R{R}\n(kept-center recon)", fontsize=9.5)
        # real radial spectrum of the organ patch
        patch = (img * (lab == o))[y0:y1, x0:x1]
        sz = max(patch.shape); pp = np.zeros((sz, sz)); pp[:patch.shape[0], :patch.shape[1]] = patch
        prof = radial_profile(pp); fx = np.linspace(0, 1, len(prof))
        ax[r, 2].fill_between(fx, prof, color=col, alpha=.5); ax[r, 2].axvline(1/R, color="k", ls="--", lw=1)
        ax[r, 2].set_title("real k-space energy vs frequency", fontsize=9.5); ax[r, 2].set_xlabel("radial freq (Nyquist=1)")
        ax[r, 2].text(1/R + .02, .8, "R8 keeps ←", fontsize=7.5); ax[r, 2].set_yticks([])
        v = "SURVIVES" if o == 5 else "BREAKS"; vc = "#1a9850" if o == 5 else "#d93025"
        ax[r, 3].axis("off"); ax[r, 3].text(.5, .5, v, ha="center", va="center", fontsize=20, color="white",
                                            fontweight="bold", bbox=dict(boxstyle="round", fc=vc, ec="none"))
    fig.suptitle("Why organs break, on real data: the thin organ's energy lives ABOVE the R8 cutoff → it is discarded\n"
                 "(double jeopardy: acceleration removes it × the network learns high freq last → the spectral-centroid law)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(f"{P}/idea_visual.png", dpi=150, bbox_inches="tight")
    print("wrote idea_visual.png")

if __name__ == "__main__":
    img, lab = pick_slice()
    print(f"picked slice with {sum(1 for o in L.ABDO if (lab==o).sum()>20)} organs")
    pipeline_visual(img, lab); idea_visual(img, lab)
    print("done")
