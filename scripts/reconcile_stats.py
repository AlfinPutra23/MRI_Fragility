"""Verify the adversarial panel's recomputations MYSELF before correcting any published claim (trust neither the panel
nor my old scripts blindly). Deterministic, no GPU.
 (1) LODO 'predict-then-verify' — is it a rank-invariance ARTIFACT (== within-dataset Spearman)? + the honest CALIBRATED
     out-of-sample test (predict target drop VALUES with the SOURCE fit; Pearson R2 / RMSE).
 (2) partial corr centroid~drop | SA:V (and | SA:V+contrast+fdim), BOTH Spearman and Pearson, permutation p
     — reconciles debunk_partial (r=0.636,p=0.02) vs panel (p=0.18).
 (3) true SEMIPARTIAL (unique) R2 of centroid over the other 4 predictors.
 (4) freq-PC1 dominance on the non-singular blocks {freq-PC1, SA:V, contrast}.
 (5) collinearity r(centroid,hf8_img), r(centroid,SA:V).
-> outputs/results/reconcile_stats.json"""
import json, itertools, math, numpy as np
from scipy import stats
np.random.seed(0)


def load(f):
    rows = json.load(open(f"outputs/results/{f}"))["rows"]
    return {k: np.array([r[k] for r in rows], float) for k in ["drop", "centroid", "sav", "contrast", "hf8_img", "fdim"]}


def z(v): return (v - v.mean()) / (v.std() + 1e-12)


def _resid(v, Z, rankit):
    cols = [z(stats.rankdata(c) if rankit else c) for c in Z]
    A = np.column_stack([np.ones_like(v)] + cols)
    return v - A @ np.linalg.lstsq(A, v, rcond=None)[0]


def partial(x, y, Z, spearman=True):
    xx = stats.rankdata(x) if spearman else x; yy = stats.rankdata(y) if spearman else y
    return float(stats.pearsonr(_resid(xx, Z, spearman), _resid(yy, Z, spearman))[0])


def perm_p(x, y, Z, obs, spearman, n=5000):
    return round(float(np.mean([abs(partial(x, np.random.permutation(y), Z, spearman)) >= abs(obs) for _ in range(n)])), 4)


def r2(y, cols):
    if not cols: return 0.0
    A = np.column_stack([np.ones_like(y)] + [z(c) for c in cols]); b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(1 - ((y - A @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def lmg(y, X):
    ps = list(X); sh = {p: 0. for p in ps}
    for o in itertools.permutations(ps):
        u = []
        for p in o: base = r2(y, [X[q] for q in u]); u.append(p); sh[p] += r2(y, [X[q] for q in u]) - base
    return {p: sh[p] / math.factorial(len(ps)) for p in ps}


D = {"MRISeg": load("m0_law_v2.json"), "AMOS": load("amos_law_v2.json")}
out = {}

# (1) LODO artifact + calibrated out-of-sample
def lodo(src, tgt):
    a, b = D[src], D[tgt]; slope, inter = np.polyfit(a["centroid"], a["drop"], 1); pred = slope * b["centroid"] + inter
    within = stats.spearmanr(b["centroid"], b["drop"])[0]; sp = stats.spearmanr(pred, b["drop"])[0]
    r2cal = 1 - ((b["drop"] - pred) ** 2).sum() / ((b["drop"] - b["drop"].mean()) ** 2).sum()
    return {"spearman_pred_vs_actual": round(float(sp), 4), "within_target_centroid_spearman": round(float(within), 4),
            "IS_RANK_ARTIFACT": bool(abs(sp - within) < 1e-6), "calibrated_R2_source_fit": round(float(r2cal), 3),
            "rmse": round(float(np.sqrt(((b["drop"] - pred) ** 2).mean())), 4)}
out["LODO_check"] = {"fit_MRISeg_predict_AMOS": lodo("MRISeg", "AMOS"), "fit_AMOS_predict_MRISeg": lodo("AMOS", "MRISeg")}

# (2) partial over size
for n, d in D.items():
    r = {}
    for lbl, Z in [("ctrl_SAV", [d["sav"]]), ("ctrl_SAV_contrast_fdim", [d["sav"], d["contrast"], d["fdim"]])]:
        ps = partial(d["centroid"], d["drop"], Z, True); pp = partial(d["centroid"], d["drop"], Z, False)
        r[lbl] = {"spearman": round(ps, 3), "perm_p_spearman": perm_p(d["centroid"], d["drop"], Z, ps, True),
                  "pearson": round(pp, 3), "perm_p_pearson": perm_p(d["centroid"], d["drop"], Z, pp, False)}
    r["reverse_SAV_given_centroid_spearman"] = round(partial(d["sav"], d["drop"], [d["centroid"]], True), 3)
    out.setdefault("partial_over_size", {})[n] = r

# (3) semipartial unique R2 of centroid over other 4
for n, d in D.items():
    o = [d["sav"], d["contrast"], d["hf8_img"], d["fdim"]]
    out.setdefault("semipartial_unique_R2", {})[n] = {"centroid_unique": round(r2(d["drop"], o + [d["centroid"]]) - r2(d["drop"], o), 3),
                                                       "full5_R2": round(r2(d["drop"], o + [d["centroid"]]), 3)}

# (4) freq-PC1 dominance
for n, d in D.items():
    F = np.column_stack([z(d["centroid"]), z(d["hf8_img"]), z(d["fdim"])]); Fc = F - F.mean(0)
    U, S, Vt = np.linalg.svd(Fc, full_matrices=False); pc1 = F @ Vt[0]
    X = {"freqPC1": pc1, "sav": d["sav"], "contrast": d["contrast"]}; Lm = lmg(d["drop"], X); tot = sum(Lm.values())
    out.setdefault("freqPC1_dominance", {})[n] = {"pc1_var_expl": round(float(S[0] ** 2 / (S ** 2).sum()), 3),
                                                  "shares_pct": {k: round(100 * Lm[k] / tot, 1) for k in X}, "total_R2": round(tot, 3)}

# (5) collinearity
for n, d in D.items():
    out.setdefault("collinearity", {})[n] = {"r_centroid_hf8": round(float(stats.spearmanr(d["centroid"], d["hf8_img"])[0]), 3),
                                             "r_centroid_sav": round(float(stats.spearmanr(d["centroid"], d["sav"])[0]), 3)}

json.dump(out, open("outputs/results/reconcile_stats.json", "w"), indent=2)
print("=== (1) LODO ===")
for k, v in out["LODO_check"].items(): print(f"  {k}: Spearman {v['spearman_pred_vs_actual']} vs within {v['within_target_centroid_spearman']} -> ARTIFACT={v['IS_RANK_ARTIFACT']} | calibrated R2={v['calibrated_R2_source_fit']}")
print("=== (2) partial centroid~drop | size ===")
for n in D:
    for lbl, r in out["partial_over_size"][n].items():
        if isinstance(r, dict): print(f"  {n} {lbl}: Spearman {r['spearman']} (p={r['perm_p_spearman']}) | Pearson {r['pearson']} (p={r['perm_p_pearson']})")
    print(f"  {n} reverse SA:V|centroid = {out['partial_over_size'][n]['reverse_SAV_given_centroid_spearman']}")
print("=== (3) semipartial unique R2 ===")
for n in D: print(f"  {n}: centroid unique R2 = {out['semipartial_unique_R2'][n]['centroid_unique']} (full-5 R2 {out['semipartial_unique_R2'][n]['full5_R2']})")
print("=== (4) freq-PC1 dominance {freqPC1, SA:V, contrast} ===")
for n in D: print(f"  {n}: PC1 var {out['freqPC1_dominance'][n]['pc1_var_expl']} | shares {out['freqPC1_dominance'][n]['shares_pct']} (R2 {out['freqPC1_dominance'][n]['total_R2']})")
print("=== (5) collinearity ===")
for n in D: print(f"  {n}: r(centroid,hf8)={out['collinearity'][n]['r_centroid_hf8']} r(centroid,SA:V)={out['collinearity'][n]['r_centroid_sav']}")
print("\nwrote reconcile_stats.json")
