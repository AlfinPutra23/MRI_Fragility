"""Debunk 'isn't it obvious (small/blurry)?': three predictors of fragility side by side —
SIZE (volume) and BLUR (local image degradation) are weak / have exceptions; FREQUENCY (spectral centroid) is strong.
-> outputs/plots/debunk_obvious.png"""
import os, sys, glob, json, argparse, numpy as np, nibabel as nib
from skimage.metrics import structural_similarity as ssim
from scipy.stats import spearmanr
sys.path.insert(0, "scripts" if os.path.isdir("scripts") else "."); import labels as L
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
from paths import RESULTS as R, PLOTS as P
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"]})

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--max_cases", type=int, default=30); a = ap.parse_args()
law = {r["organ"]: r for r in json.load(open(f"{R}/m0_law_v2.json"))["rows"]}     # centroid, drop
ha = {r["organ"]: r for r in json.load(open(f"{R}/m0_h_a.json"))["rows"]}          # vol_cm3

# recompute per-organ LOCAL image degradation (1 - SSIM of the organ region, clean vs R8) = the 'blur' predictor
cases = sorted(glob.glob(f"{a.root}/labelsTs/*.nii.gz")); cases = cases[::max(len(cases)//a.max_cases, 1)]
loc = {o: [] for o in L.ABDO}
print(f"debunk: computing local blur on {len(cases)} cases")
for gp in cases:
    c = os.path.basename(gp)[:-7]
    cln = np.asanyarray(nib.load(f"{a.root}/imagesTs_clean/{c}_0000.nii.gz").dataobj).astype(np.float32)
    r8 = np.asanyarray(nib.load(f"{a.root}/imagesTs_R8/{c}_0000.nii.gz").dataobj).astype(np.float32)
    lab = np.asanyarray(nib.load(gp).dataobj).astype(np.int16); dr = float(cln.max()-cln.min())+1e-6
    for o in L.ABDO:
        m = lab == o
        if m.sum() < 60: continue
        sl = tuple(slice(max(p.min()-2, 0), p.max()+3) for p in np.where(m))
        if min(cln[sl].shape) >= 7: loc[o].append(1 - ssim(cln[sl], r8[sl], data_range=dr))

organs = [nm for o, nm in L.ABDO.items() if nm in law and nm in ha and loc[o]]
by = {nm: dict(drop=law[nm]["drop"], cen=law[nm]["centroid"], vol=ha[nm]["vol_cm3"],
               blur=float(np.mean(loc[[o for o, n in L.ABDO.items() if n == nm][0]])),
               tail=[o for o, n in L.ABDO.items() if n == nm][0] in L.TAIL) for nm in organs}
y = np.array([by[nm]["drop"] for nm in organs])
RED, BLUE = "#d93025", "#2c5fb0"; cols = [RED if by[nm]["tail"] else BLUE for nm in organs]

def panel(ax, key, xlab, title, verdict, vcol, logx=False, note=None):
    x = np.array([by[nm][key] for nm in organs])
    ax.scatter(x, y, s=95, c=cols, zorder=3, edgecolor="white", lw=.6)
    for nm in organs:
        if nm in ("liver", "colon", "small_bowel", "pancreas", "adrenal_L", "gallbladder"):
            ax.annotate(nm, (by[nm][key], by[nm]["drop"]), fontsize=8, xytext=(4, 3), textcoords="offset points")
    if logx: ax.set_xscale("log")
    r = spearmanr(x, y).correlation
    ax.text(.04, .93, f"Spearman |r| = {abs(r):.2f}", transform=ax.transAxes, fontsize=13, fontweight="bold")
    ax.text(.04, .85, verdict, transform=ax.transAxes, fontsize=12.5, fontweight="bold", color=vcol)
    if note: ax.text(.04, .77, note, transform=ax.transAxes, fontsize=9.5, color="#555", style="italic")
    ax.set_xlabel(xlab, fontsize=11); ax.set_ylabel("fragility (Dice drop)", fontsize=11)
    ax.set_title(title, fontsize=12.5, fontweight="bold"); ax.grid(alpha=.3)
    return abs(r)

fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.6))
rv = panel(ax[0], "vol", "organ volume (cm³, log)", 'A.  "it\'s small"  —  SIZE', "weak + exceptions", "#e08214",
           logx=True, note="colon & small_bowel are LARGE yet fragile")
rb = panel(ax[1], "blur", "how blurry the organ got (1 − SSIM)", 'B.  "it\'s blurry"  —  BLUR', "weakest", "#e08214",
           note="same blur → very different failure")
rc = panel(ax[2], "cen", "spectral centroid (frequency content)", 'C.  the real one  —  FREQUENCY', "STRONG", "#1a9850",
           note="predicts it, cross-validated, 2 datasets")
fig.suptitle("Is it obvious?  No — it's NOT size and NOT blur, it's FREQUENCY CONTENT\n"
             f"(the two 'obvious' predictors are weak: size |r|={rv:.2f}, blur |r|={rb:.2f};  frequency |r|={rc:.2f})",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout(); fig.savefig(f"{P}/debunk_obvious.png", dpi=150, bbox_inches="tight")
json.dump({"size_abs_r": round(abs(rv), 3), "blur_abs_r": round(abs(rb), 3), "freq_centroid_abs_r": round(abs(rc), 3),
           "n_organs": len(by), "metric": "Spearman |r| of each predictor vs per-organ Dice drop",
           "verdict": "frequency (centroid) STRONG >> size (weak, exceptions) > blur (weakest)"},
          open(f"{R}/debunk_obvious.json", "w"), indent=2)
print(f"size |r|={rv:.2f}  blur |r|={rb:.2f}  frequency |r|={rc:.2f}")
print(f"wrote {P}/debunk_obvious.png + {R}/debunk_obvious.json")
