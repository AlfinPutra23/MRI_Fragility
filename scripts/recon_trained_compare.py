"""#2 domain-matched recon comparison (fair W3): all at R8 CS-recon input, tail Dice (>=30-vox, per-case Wilcoxon).
  B clean-trained  @R8cs (predsRECON_R8cs)                 = standard recon-then-segment
  E recon-trained  @R8cs (predsRECONTR_R8cs)               = the STRONGEST recon pipeline (train ON recon distribution)
  D mixedR-trained @R8cs (predsSTACK_..Uniform_s42_R8cs)   = train-for-degradation
Key question: does mixed-R still beat even the domain-matched recon-trained model? -> closes the W3 fairness hole."""
import glob, os, sys, numpy as np, nibabel as nib
from scipy.stats import wilcoxon
sys.path.insert(0, "scripts"); import labels as L

D = "nnUNet_raw/Dataset501_MRIfrag"; LABDIR = f"{D}/labelsTs"; TAIL = list(L.TAIL); MINVOX = 30
CELLS = {"B clean-trained  @R8cs": "predsRECON_R8cs",
         "E recon-trained  @R8cs": "predsRECONTR_R8cs",
         "D mixedR-trained @R8cs": "predsSTACK_nnUNetTrainer_Uniform_s42_R8cs"}


def dice(a, b): s = a.sum() + b.sum(); return 2 * np.logical_and(a, b).sum() / s if s else np.nan


def per_case(pred):
    out = {}
    for gp in sorted(glob.glob(f"{LABDIR}/*.nii.gz")):
        c = os.path.basename(gp); pp = f"{D}/{pred}/{c}"
        if not os.path.exists(pp): continue
        g = np.asanyarray(nib.load(gp).dataobj); p = np.asanyarray(nib.load(pp).dataobj)
        ds = [dice(p == o, g == o) for o in TAIL if (g == o).sum() >= MINVOX]
        if ds: out[c] = float(np.nanmean(ds))
    return out


T = {k: per_case(v) for k, v in CELLS.items() if os.path.isdir(f"{D}/{v}")}
print("=== tail Dice @R8cs (>=30-vox) ===")
for k in CELLS:
    if k in T: print(f"  {k}: {np.mean(list(T[k].values())):.4f}  (n={len(T[k])})")
    else: print(f"  {k}: (not ready)")


def delta(a, b):
    if a not in T or b not in T: return None
    cc = sorted(set(T[a]) & set(T[b])); d = np.array([T[a][c] - T[b][c] for c in cc])
    return d.mean(), wilcoxon([T[a][c] for c in cc], [T[b][c] for c in cc]).pvalue, len(cc)


print("\n=== key deltas ===")
for lab, a, b in [("recon-trained vs clean-trained (does training-on-recon help?)", "E recon-trained  @R8cs", "B clean-trained  @R8cs"),
                  ("mixed-R vs recon-trained (train-for-degradation vs BEST recon)", "D mixedR-trained @R8cs", "E recon-trained  @R8cs")]:
    r = delta(a, b)
    if r: print(f"  {lab}: Δ {r[0]:+.4f}  p={r[1]:.1e}  (n={r[2]})")
    else: print(f"  {lab}: (pending)")
import json; json.dump({k: float(np.mean(list(T[k].values()))) for k in T}, open("outputs/results/recon_trained.json", "w"), indent=2)
print("\nwrote recon_trained.json")
