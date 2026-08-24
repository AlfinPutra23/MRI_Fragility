"""Legitimacy stats (no GPU), from the bulletproofing plan:
 (1) MAX-STATISTIC label-permutation null across ALL candidate frequency predictors — the correct multiplicity control at
     n=13 (kills garden-of-forking-paths: is 'centroid' just the winner of a fishing expedition?).  [VALID — centroid is
     the top predictor, selection-corrected p=0.0015/0.0053.]
 (2) LEAVE-ONE-DATASET-OUT 'predict-then-verify' Spearman.  *** DEPRECATED — DO NOT CITE. ***  Adversarial audit proved
     this is a RANK-INVARIANCE ARTIFACT: with a monotone (linear) source fit scored by rank-based Spearman, the statistic
     equals the TARGET dataset's OWN within-dataset centroid Spearman (0.8545==0.8545 exactly), carrying no cross-dataset
     information. The honest cross-dataset test is the CALIBRATED one (source fit -> target drop VALUES, Pearson R2=0.67/
     0.74) in scripts/reconcile_stats.py -> reconcile_stats.json. Use that, not the Spearman below.
-> outputs/results/legit_stats.json"""
import json, numpy as np
from scipy import stats
np.random.seed(0)
PREDS = ["centroid", "hf8_img", "sav", "contrast", "fdim"]     # competing frequency/shape descriptors present in *_law_v2.json

def load(f):
    rows = json.load(open(f"outputs/results/{f}"))["rows"]
    drop = np.array([r["drop"] for r in rows])
    X = {p: np.array([r.get(p, np.nan) for r in rows]) for p in PREDS}
    X = {p: v for p, v in X.items() if not np.isnan(v).any()}
    return drop, X

def maxstat(drop, X, nperm=20000):
    obs = {p: abs(stats.spearmanr(X[p], drop)[0]) for p in X}
    maxnull = np.empty(nperm)
    for i in range(nperm):
        d = np.random.permutation(drop)
        maxnull[i] = max(abs(stats.spearmanr(X[p], d)[0]) for p in X)      # max |r| over ALL predictors each permutation
    return {p: {"abs_spearman": round(obs[p], 3), "p_selection_corrected": round(float((maxnull >= obs[p]).mean()), 4)}
            for p in sorted(X, key=lambda k: -obs[k])}

def lodo(fitf, predf):                                       # fit centroid->drop on A, predict B's ordering
    dA, XA = load(fitf); dB, XB = load(predf)
    b = np.polyfit(XA["centroid"], dA, 1)
    r, p = stats.spearmanr(np.polyval(b, XB["centroid"]), dB)
    return {"spearman_pred_vs_actual": round(float(r), 3), "p": round(float(p), 4), "n": len(dB)}

out = {}
for name, f in [("MRISegmentator", "m0_law_v2.json"), ("AMOS", "amos_law_v2.json")]:
    drop, X = load(f); out[name] = {"n": len(drop), "maxstat_permutation": maxstat(drop, X)}
out["LODO_predict_then_verify"] = {"fit_MRISeg_predict_AMOS": lodo("m0_law_v2.json", "amos_law_v2.json"),
                                   "fit_AMOS_predict_MRISeg": lodo("amos_law_v2.json", "m0_law_v2.json")}
json.dump(out, open("outputs/results/legit_stats.json", "w"), indent=2)
for name in ["MRISegmentator", "AMOS"]:
    c = out[name]["maxstat_permutation"]["centroid"]
    win = list(out[name]["maxstat_permutation"])[0]
    print(f"{name}: centroid |r|={c['abs_spearman']} selection-corrected p={c['p_selection_corrected']}  (top predictor: {win})")
lo = out["LODO_predict_then_verify"]
print(f"LODO predict-then-verify: MRISeg->AMOS Spearman={lo['fit_MRISeg_predict_AMOS']['spearman_pred_vs_actual']} (p={lo['fit_MRISeg_predict_AMOS']['p']}); AMOS->MRISeg={lo['fit_AMOS_predict_MRISeg']['spearman_pred_vs_actual']} (p={lo['fit_AMOS_predict_MRISeg']['p']})")
print("wrote legit_stats.json")
