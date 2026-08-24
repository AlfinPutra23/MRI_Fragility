"""SOTA comparison: the standard way to judge fast MRI is global image quality (SSIM/PSNR). We show it is BLIND to
which ORGAN fails -- one number for the whole image, while per-organ Dice spans 10x -- whereas our anatomy law
(spectral centroid) predicts the per-organ drop a priori. Also: per-organ local SSIM-loss vs drop (post-hoc proxy).
  python metric_vs_law.py --root <Dataset501> [--max_cases 30]
-> outputs/results/metric_vs_law.json , outputs/plots/metric_vs_law.png"""
import os, sys, glob, json, argparse, numpy as np, nibabel as nib
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from scipy.stats import spearmanr
sys.path.insert(0, "scripts" if os.path.isdir("scripts") else "."); import labels as L
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from paths import RESULTS as RES, PLOTS as PLT

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--max_cases", type=int, default=30)
a = ap.parse_args()
Rs = [2, 4, 6, 8]; TAG = {2: "R2", 4: "R4", 6: "R6", 8: "R8"}
frag = json.load(open(f"{RES}/m0_fragility_dice.json"))
cen = {r["organ"]: r["centroid"] for r in json.load(open(f"{RES}/m0_law_v2.json"))["rows"]}

cases = sorted(glob.glob(f"{a.root}/labelsTs/*.nii.gz"))
cases = cases[::max(len(cases) // a.max_cases, 1)]
gm = {r: {"ssim": [], "psnr": []} for r in Rs}
loc8 = {o: [] for o in L.ABDO}                                   # per-organ local SSIM-loss at R8
print(f"metric_vs_law: {len(cases)} cases")
for gp in cases:
    c = os.path.basename(gp)[:-7]
    clean = np.asanyarray(nib.load(f"{a.root}/imagesTs_clean/{c}_0000.nii.gz").dataobj).astype(np.float32)
    dr = float(clean.max() - clean.min()) + 1e-6
    lab = np.asanyarray(nib.load(gp).dataobj).astype(np.int16)
    accs = {}
    for r in Rs:
        acc = np.asanyarray(nib.load(f"{a.root}/imagesTs_{TAG[r]}/{c}_0000.nii.gz").dataobj).astype(np.float32)
        accs[r] = acc
        gm[r]["ssim"].append(ssim(clean, acc, data_range=dr))
        gm[r]["psnr"].append(psnr(clean, acc, data_range=dr))
    a8 = accs[8]
    for o in L.ABDO:
        m = lab == o
        if m.sum() < 60: continue
        sl = tuple(slice(max(p.min()-2, 0), p.max()+3) for p in np.where(m))
        cc, aa = clean[sl], a8[sl]
        if min(cc.shape) >= 7:
            loc8[o].append(1 - ssim(cc, aa, data_range=dr))       # local SSIM-loss (higher = organ region degraded more)

# aggregate
gssim = {r: float(np.mean(gm[r]["ssim"])) for r in Rs}; gpsnr = {r: float(np.mean(gm[r]["psnr"])) for r in Rs}
organs = [nm for o, nm in L.ABDO.items() if nm in frag and nm in cen and loc8[o]]
drop = {nm: frag[nm]["R1"] - frag[nm]["R8"] for nm in organs}
locv = {L.ABDO[o]: float(np.mean(loc8[o])) for o in L.ABDO if L.ABDO[o] in organs and loc8[o]}
y = np.array([drop[nm] for nm in organs])
r_cen = spearmanr([cen[nm] for nm in organs], y).correlation
r_loc = spearmanr([locv[nm] for nm in organs], y).correlation

print(f"\n=== global image quality (the standard metric) ===")
for r in Rs: print(f"  R{r}: SSIM={gssim[r]:.3f}  PSNR={gpsnr[r]:.1f} dB")
print(f"\n=== but per-organ Dice DROP @R8 spans a huge range at that SAME global quality ===")
print(f"  global SSIM @R8 = {gssim[8]:.3f}  |  per-organ drop: min {y.min():.3f} ({organs[int(y.argmin())]}) -> max {y.max():.3f} ({organs[int(y.argmax())]})  ({y.max()/max(y.min(),1e-3):.0f}x)")
print(f"\n=== who predicts WHICH organ fails? ===")
print(f"  global SSIM/PSNR  -> ONE number for all organs -> cannot rank organs (0 per-organ info)")
print(f"  per-organ local SSIM-loss (needs the scan)  Spearman r = {r_loc:+.2f}")
print(f"  spectral centroid (ANATOMY, a priori)        Spearman r = {r_cen:+.2f}")

json.dump(dict(global_ssim=gssim, global_psnr=gpsnr, drop_range=[float(y.min()), float(y.max())],
               r_centroid=float(r_cen), r_local_ssim=float(r_loc)), open(f"{RES}/metric_vs_law.json", "w"), indent=2)

# figure: (A) global metric degrades smoothly; (B) same scans, per-organ Dice spreads
fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.4))
xr = [1] + Rs
ax[0].plot(xr, [1.0] + [gssim[r] for r in Rs], "-o", color="#2c5fb0", lw=2.4, label="global SSIM")
ax[0].set_ylim(0.6, 1.02); ax[0].set_xlabel("acceleration R"); ax[0].set_ylabel("global SSIM (whole image)")
ax[0].set_title("A. Standard metric: ONE number per image\n(looks like a smooth, gentle decline)", fontweight="bold", fontsize=11)
ax[0].set_xticks(xr); ax[0].grid(alpha=.3); ax[0].legend()
ax[0].annotate(f"R8: SSIM {gssim[8]:.2f}\n\"looks {int(gssim[8]*100)}% fine\"", (8, gssim[8]), fontsize=9,
               xytext=(-70, 20), textcoords="offset points", color="#2c5fb0")
for o, nm in L.ABDO.items():
    if nm not in frag: continue
    yv = [frag[nm][f"R{r}"] for r in xr]; tail = o in L.TAIL
    ax[1].plot(xr, yv, "-o", color=("#d93025" if tail else "#2c5fb0"), alpha=(.9 if tail else .45),
               lw=(2.2 if tail else 1.2), ms=(5 if tail else 3))
    if nm in ("adrenal_L", "liver", "gallbladder"): ax[1].annotate(nm, (8, yv[-1]), fontsize=8, xytext=(3, -2), textcoords="offset points")
ax[1].set_xlabel("acceleration R"); ax[1].set_ylabel("per-organ Dice"); ax[1].set_xticks(xr); ax[1].grid(alpha=.3)
ax[1].set_title("B. Same scans, per organ: a 10× spread the metric can't see\n"
                f"(our law predicts this spread: centroid r = {r_cen:+.2f})", fontweight="bold", fontsize=11)
ax[1].plot([], [], "-o", color="#d93025", label="tail (fragile)"); ax[1].plot([], [], "-o", color="#2c5fb0", alpha=.5, label="large (robust)")
ax[1].legend(loc="lower left", fontsize=9)
fig.suptitle("Image quality (SSIM/PSNR) tells you the picture looks worse — NOT which organs became unreliable. Anatomy does.",
             fontsize=12.5, fontweight="bold", y=1.0)
fig.tight_layout(); fig.savefig(f"{PLT}/metric_vs_law.png", dpi=150, bbox_inches="tight")
print(f"\nwrote {RES}/metric_vs_law.json , {PLT}/metric_vs_law.png")
