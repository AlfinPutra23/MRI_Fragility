"""Compare AMOS mixed-R vs baseline (both nnUNetTrainer_250epochs), on Dataset502 test @R8.
  baseline: Dataset502/predsTs_R8      (trained @R2)     -- already exists
  mixed-R : Dataset502/predsMIXEDR_R8  (trained mixed-R, Dataset506 model)
Per-organ tail Dice + Wilcoxon + gain-vs-centroid, mirroring the abdominal MRISeg mixed-R result.
-> outputs/results/amos_mixedr.json  (+ ..._perorgan)"""
import glob, os, json, numpy as np, nibabel as nib, sys
from scipy.stats import wilcoxon, spearmanr
sys.path.insert(0, "scripts" if os.path.isdir("scripts") else "."); import amos_labels as L
D = "nnUNet_raw/Dataset502_AMOSfrag"; GT = f"{D}/labelsTs"
A_dir = f"{D}/predsTs_R8"; B_dir = f"{D}/predsMIXEDR_R8"


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

law = {r["organ"]: r["centroid"] for r in json.load(open("outputs/results/amos_law_v2.json"))["rows"]}
rows = []
for o in L.ABDO:
    nm = L.ABDO[o]; a = np.array(per[o]["A"]); b = np.array(per[o]["B"])
    if len(a) < 3: continue
    rows.append({"organ": nm, "tail": o in L.TAIL, "n": len(a), "centroid": law.get(nm),
                 "dice_R2train": round(float(a.mean()), 4), "dice_mixedR": round(float(b.mean()), 4),
                 "gain": round(float((b - a).mean()), 4)})
rows.sort(key=lambda r: -r["gain"])

# aggregate tail-mean across cases (paired Wilcoxon over cases)
tail_ids = [o for o in L.ABDO if o in L.TAIL]
cases = sorted(set.intersection(*[set(range(len(per[o]["A"]))) for o in tail_ids if per[o]["A"]])) if tail_ids else []
A_case = [np.nanmean([per[o]["A"][i] for o in tail_ids if i < len(per[o]["A"])]) for i in cases]
B_case = [np.nanmean([per[o]["B"][i] for o in tail_ids if i < len(per[o]["B"])]) for i in cases]
A_case = np.array(A_case); B_case = np.array(B_case); d = B_case - A_case
g = [r["gain"] for r in rows if r["centroid"] is not None]; c = [r["centroid"] for r in rows if r["centroid"] is not None]
out = {"tail": {"n": len(cases), "uniform_R2": round(float(A_case.mean()), 4), "uniform_mixedR": round(float(B_case.mean()), 4),
                "delta": round(float(d.mean()), 4), "wilcoxon_p": float(wilcoxon(B_case, A_case).pvalue) if (d != 0).any() else None,
                "improved_cases": int((d > 0).sum())},
       "gain_vs_centroid_spearman": round(float(spearmanr(c, g).correlation), 3),
       "tail_mean_gain": round(float(np.mean([r["gain"] for r in rows if r["tail"]])), 4),
       "robust_mean_gain": round(float(np.mean([r["gain"] for r in rows if not r["tail"]])), 4),
       "rows": rows,
       "note": "AMOS nnU-Net mixed-R mitigation @R8. THIRD mitigation datapoint (knee +0.134, MRISeg abdomen +0.088)."}
json.dump(out, open("outputs/results/amos_mixedr.json", "w"), indent=2)
t = out["tail"]
print(f"=== AMOS MIXED-R (tail Dice @R8, n={t['n']} cases) ===")
print(f"  uniform @R2      : {t['uniform_R2']}")
print(f"  uniform @mixed-R : {t['uniform_mixedR']}")
print(f"  DELTA = +{t['delta']}  Wilcoxon p={t['wilcoxon_p']}  improved {t['improved_cases']}/{t['n']}")
print(f"  gain vs centroid Spearman = {out['gain_vs_centroid_spearman']}  (tail gain {out['tail_mean_gain']} vs robust {out['robust_mean_gain']})")
print("wrote amos_mixedr.json")
