"""GPU-free hardening stats (bulletproofing plan #3, #8), run while the GPU is busy with the make-or-break:
 (1) DOMINANCE (LMG / Shapley-R2) decomposition + VIF -> centroid's UNIQUE variance share vs SA:V, contrast, texture,
     on BOTH datasets (kills 'centroid is just size/contrast re-described').
 (2) MATCHED-PAIRS dissociation -> organ pairs matched on SA:V (the size/shape proxy) but differing in centroid: does
     the HIGHER-centroid member drop more? A clean within-size spectral dissociation.
 (3) HONEST-n stack -> centroid Spearman with organ-level bootstrap CI, leave-one-organ-out range (which single organ
     carries it?), and a full permutation p. States the inferential n honestly.
-> outputs/results/harden_stats.json"""
import json, itertools, math, numpy as np
from scipy import stats
np.random.seed(0)
PREDS = ["centroid", "sav", "contrast", "hf8_img", "fdim"]


def load(f):
    rows = json.load(open(f"outputs/results/{f}"))["rows"]
    y = np.array([r["drop"] for r in rows], float)
    X = {p: np.array([r[p] for r in rows], float) for p in PREDS}
    return [r["organ"] for r in rows], y, X


def z(v): return (v - v.mean()) / (v.std() + 1e-12)


def r2(y, cols):                                             # OLS R^2 of y on standardized predictors (+intercept)
    if not cols: return 0.0
    A = np.column_stack([np.ones_like(y)] + [z(c) for c in cols])
    beta = np.linalg.lstsq(A, y, rcond=None)[0]; ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - A @ beta) ** 2).sum() / ss) if ss > 0 else 0.0


def lmg(y, X):                                               # LMG = avg marginal R^2 increment over ALL orderings (= Shapley for R^2)
    ps = list(X); share = {p: 0.0 for p in ps}
    for order in itertools.permutations(ps):
        used = []
        for p in order:
            base = r2(y, [X[q] for q in used]); used.append(p); share[p] += r2(y, [X[q] for q in used]) - base
    n = math.factorial(len(ps)); return {p: share[p] / n for p in ps}


def vif(X):
    ps = list(X)
    return {p: round(1 / (1 - r2(X[p], [X[q] for q in ps if q != p]) + 1e-12), 2) for p in ps}


def matched_pairs(names, y, X):                              # pairs matched on SA:V, differing in centroid
    sav, cen = X["sav"], X["centroid"]
    for tol in (0.01, 0.02, 0.03, 0.05, 0.08):               # tighten as far as still yields >=4 pairs
        pairs = []
        for i, j in itertools.combinations(range(len(y)), 2):
            if abs(sav[i] - sav[j]) <= tol and abs(cen[i] - cen[j]) > 0.03:
                hi, lo = (i, j) if cen[i] > cen[j] else (j, i)
                pairs.append([names[hi], names[lo], round(float(y[hi] - y[lo]), 3)])   # >0 => higher-centroid drops MORE
        if len(pairs) >= 4: return {"sav_tol": tol, "pairs": pairs}
    return {"sav_tol": tol, "pairs": pairs}


def honest_n(names, y, X, nboot=10000):
    cen = X["centroid"]; obs = stats.spearmanr(cen, y)[0]
    rs = []
    for _ in range(nboot):
        idx = np.random.randint(0, len(y), len(y))
        if len(set(idx)) > 2:
            r = stats.spearmanr(cen[idx], y[idx])[0]
            if np.isfinite(r): rs.append(r)
    rs = np.array(rs)
    loo = [[names[k], round(float(stats.spearmanr(np.delete(cen, k), np.delete(y, k))[0]), 3)] for k in range(len(y))]
    perm = np.array([abs(stats.spearmanr(cen, np.random.permutation(y))[0]) for _ in range(10000)])
    return {"spearman": round(float(obs), 3),
            "boot_CI95": [round(float(np.percentile(rs, 2.5)), 3), round(float(np.percentile(rs, 97.5)), 3)],
            "loo_spearman_range": [round(min(v for _, v in loo), 3), round(max(v for _, v in loo), 3)],
            "loo_weakest_when_dropping": min(loo, key=lambda t: t[1]),
            "perm_p": round(float((perm >= abs(obs)).mean()), 4)}


out = {}
for name, f in [("MRISegmentator", "m0_law_v2.json"), ("AMOS", "amos_law_v2.json")]:
    names, y, X = load(f); Lm = lmg(y, X); tot = sum(Lm.values())
    out[name] = {"n": len(y), "lmg_R2": {p: round(Lm[p], 3) for p in PREDS},
                 "lmg_share_pct": ({p: round(100 * Lm[p] / tot, 1) for p in PREDS} if tot > 0 else None),
                 "total_R2": round(tot, 3), "vif": vif(X),
                 "matched_pairs": matched_pairs(names, y, X), "honest_n": honest_n(names, y, X)}
json.dump(out, open("outputs/results/harden_stats.json", "w"), indent=2)
for name in out:
    o = out[name]; mp = o["matched_pairs"]["pairs"]; npos = sum(1 for *_, d in mp if d > 0); sh = o["lmg_share_pct"]
    print(f"\n{name} (n={o['n']}):")
    print(f"  LMG unique-R2 share: centroid {sh['centroid']}% | SA:V {sh['sav']}% | contrast {sh['contrast']}% | hf8 {sh['hf8_img']}% | fdim {sh['fdim']}%   (model R2={o['total_R2']})")
    print(f"  VIF centroid={o['vif']['centroid']}  (collinearity, <5 = fine)")
    print(f"  matched pairs (|ΔSA:V|≤{o['matched_pairs']['sav_tol']}, |Δcentroid|>0.03): {npos}/{len(mp)} higher-centroid drops MORE")
    h = o["honest_n"]; print(f"  honest-n: Spearman {h['spearman']} boot95 {h['boot_CI95']} | LOO range {h['loo_spearman_range']} (weakest dropping {h['loo_weakest_when_dropping']}) | perm p={h['perm_p']}")
print("\nwrote harden_stats.json")
