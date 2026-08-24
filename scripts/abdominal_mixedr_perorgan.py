"""Per-organ abdominal mixed-R breakdown: does mixed-R training rescue exactly the organs the fragility law flags?
Uses the EXISTING nnU-Net predictions (no GPU): predsSW_Uniform (trained @R2) vs predsMIXEDR_Uniform (trained mixed-R),
both on Dataset501 test @R8. -> outputs/results/abdominal_mixedr_perorgan.json"""
import glob, os, json, numpy as np, nibabel as nib, sys
from scipy.stats import wilcoxon, spearmanr
sys.path.insert(0, "scripts"); import labels as L
ROOT = "nnUNet_raw/Dataset501_MRIfrag"; GT = f"{ROOT}/labelsTs"
A_dir = f"{ROOT}/predsSW_nnUNetTrainer_Uniform_s42_R8"
B_dir = f"{ROOT}/predsMIXEDR_nnUNetTrainer_Uniform_s42_R8"


def dice(a, b):
    s = a.sum() + b.sum(); return 2 * np.logical_and(a, b).sum() / s if s else np.nan


per = {o: {"A": [], "B": []} for o in L.ABDO}
for gp in sorted(glob.glob(f"{GT}/*.nii.gz")):
    case = os.path.basename(gp)[:-7]; ap = f"{A_dir}/{case}.nii.gz"; bp = f"{B_dir}/{case}.nii.gz"
    if not (os.path.exists(ap) and os.path.exists(bp)): continue
    gt = np.asanyarray(nib.load(gp).dataobj).astype(np.int16)
    pa = np.asanyarray(nib.load(ap).dataobj).astype(np.int16)
    pb = np.asanyarray(nib.load(bp).dataobj).astype(np.int16)
    for o in L.ABDO:
        if (gt == o).sum() >= 30:
            per[o]["A"].append(dice(pa == o, gt == o)); per[o]["B"].append(dice(pb == o, gt == o))

law = {r["organ"]: r["centroid"] for r in json.load(open("outputs/results/m0_law_v2.json"))["rows"]}
rows = []
for o in L.ABDO:
    nm = L.ABDO[o]; a = np.array(per[o]["A"]); b = np.array(per[o]["B"])
    if len(a) < 3: continue
    rows.append({"organ": nm, "tail": o in L.TAIL, "n": len(a), "centroid": law.get(nm),
                 "dice_R2train": round(float(a.mean()), 4), "dice_mixedR": round(float(b.mean()), 4),
                 "gain": round(float((b - a).mean()), 4),
                 "p": float(wilcoxon(b, a).pvalue) if (b - a).any() else None})
rows.sort(key=lambda r: -r["gain"])
g = [r["gain"] for r in rows]; c = [r["centroid"] for r in rows]
rho = float(spearmanr(c, g).correlation)
out = {"rows": rows, "gain_vs_centroid_spearman": round(rho, 3),
       "tail_mean_gain": round(float(np.mean([r["gain"] for r in rows if r["tail"]])), 4),
       "robust_mean_gain": round(float(np.mean([r["gain"] for r in rows if not r["tail"]])), 4)}
json.dump(out, open("outputs/results/abdominal_mixedr_perorgan.json", "w"), indent=2)
print(f"per-organ mixed-R gain vs centroid: Spearman = {rho:+.3f}")
print(f"tail mean gain {out['tail_mean_gain']:+.4f}  vs robust {out['robust_mean_gain']:+.4f}")
print(f"\n{'organ':13s} {'centroid':>8s} {'R2train':>8s} {'mixedR':>8s} {'gain':>8s}")
for r in rows:
    print(f"{r['organ']:13s} {r['centroid']:8.3f} {r['dice_R2train']:8.3f} {r['dice_mixedR']:8.3f} {r['gain']:+8.4f} {'[FRAG]' if r['tail'] else ''}")
print("\nwrote abdominal_mixedr_perorgan.json")
