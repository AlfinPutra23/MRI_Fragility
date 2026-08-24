"""SERIALIZE the W1 whole-body dissociation stats (previously only prose in PAPER_DRAFT.md -> unauditable + drift-prone).
Computes, for BOTH TotalSeg reconstructions (full-res and fast), the partial rank-correlation of centroid~drop CONTROLLING
for baseline difficulty (clean dice_R1), with a permutation p; plus the gradual-only subset, new-anatomy-only subset, and a
difficulty-matched-pairs sign test. This is what backs 'centroid dissociates from difficulty on the whole body' (W1).
  python totalseg_w1_partial.py  -> outputs/results/totalseg_w1_partial.json"""
import json, itertools, numpy as np
from scipy.stats import spearmanr, rankdata, binomtest
np.random.seed(0)

GRADUAL_MAX_DROP = 0.25   # structures with drop>0.25 collapse for non-spectral (peripheral-MSK) reasons
MATCH_TOL = 0.05          # |dice_R1_i - dice_R1_j| <= tol => "difficulty-matched" pair
N_PERM = 20000


def partial_spearman(x, y, ctrl):
    """Spearman partial correlation of x,y controlling ctrl: rank, regress each on [1,rank(ctrl)], correlate residuals."""
    xr, yr, cr = rankdata(x), rankdata(y), rankdata(ctrl)
    A = np.column_stack([np.ones_like(cr), cr])
    rx = xr - A @ np.linalg.lstsq(A, xr, rcond=None)[0]
    ry = yr - A @ np.linalg.lstsq(A, yr, rcond=None)[0]
    return float(spearmanr(rx, ry).correlation)


def perm_p(x, y, ctrl, obs, n=N_PERM):
    """one-sided permutation p: permute x, recompute partial, P(perm >= obs)."""
    x = np.asarray(x); ge = 1
    for _ in range(n):
        if partial_spearman(np.random.permutation(x), y, ctrl) >= obs: ge += 1
    return ge / (n + 1)


def matched_pairs(cen, drop, dif, tol=MATCH_TOL):
    """among difficulty-matched pairs, does the higher-centroid structure have the higher drop? sign test."""
    conc = tot = 0
    for i, j in itertools.combinations(range(len(cen)), 2):
        if abs(dif[i] - dif[j]) <= tol and cen[i] != cen[j] and drop[i] != drop[j]:
            tot += 1
            if (cen[i] - cen[j]) * (drop[i] - drop[j]) > 0: conc += 1
    p = binomtest(conc, tot, 0.5, alternative="greater").pvalue if tot else None
    return conc, tot, (float(p) if p is not None else None)


def analyze(path):
    d = json.load(open(path)); rows = d["rows"]
    cen = np.array([r["centroid"] for r in rows]); drop = np.array([r["drop"] for r in rows])
    dif = np.array([r["dice_R1"] for r in rows]); newa = np.array([r["new_anatomy"] for r in rows])
    def block(mask, label):
        c, dr, di = cen[mask], drop[mask], dif[mask]
        if len(c) < 5: return {"n": int(mask.sum()), "note": "too few structures"}
        ps = partial_spearman(c, dr, di)
        return {"n": int(mask.sum()),
                "marginal_centroid_drop_spearman": round(float(spearmanr(c, dr).correlation), 3),
                "difficulty_drop_spearman": round(float(spearmanr(di, dr).correlation), 3),
                "partial_centroid_drop_given_difficulty": round(ps, 3),
                "perm_p": round(perm_p(c, dr, di, ps), 4)}
    allm = np.ones(len(rows), bool)
    grad = drop <= GRADUAL_MAX_DROP
    conc, tot, mp = matched_pairs(cen, drop, dif)
    return {"n_structures": len(rows),
            "all": block(allm, "all"),
            "gradual_only": {**block(grad, "gradual"), "gradual_max_drop": GRADUAL_MAX_DROP},
            "new_anatomy_only": block(newa, "new_anatomy"),
            "matched_pairs": {"tol": MATCH_TOL, "concordant": conc, "total": tot,
                              "frac": round(conc / tot, 3) if tot else None, "sign_test_p": mp}}


out = {"description": "W1 whole-body dissociation: does spectral centroid predict Dice-drop OVER baseline difficulty "
                      "(clean dice_R1)? Serialized from the TotalSeg-MRI R1-vs-R8 rows so the paper's W1 numbers are "
                      "auditable. NOTE preprocessing-sensitive: strong on full-res, attenuates on fast recon.",
       "fullres": analyze("outputs/results/totalseg_law_fullres.json"),
       "fast": analyze("outputs/results/totalseg_law.json")}
json.dump(out, open("outputs/results/totalseg_w1_partial.json", "w"), indent=2)
print("=== W1 whole-body dissociation (SERIALIZED) ===")
for res in ["fullres", "fast"]:
    a = out[res]["all"]; g = out[res]["gradual_only"]; na = out[res]["new_anatomy_only"]; m = out[res]["matched_pairs"]
    print(f"\n[{res}]  (n={out[res]['n_structures']})")
    print(f"  ALL      : marginal {a['marginal_centroid_drop_spearman']} | difficulty {a['difficulty_drop_spearman']} | "
          f"PARTIAL {a['partial_centroid_drop_given_difficulty']} (perm p={a['perm_p']})")
    print(f"  gradual  : partial {g['partial_centroid_drop_given_difficulty']} (perm p={g['perm_p']}, n={g['n']})")
    print(f"  new-anat : partial {na['partial_centroid_drop_given_difficulty']} (perm p={na['perm_p']}, n={na['n']})")
    print(f"  matched  : {m['concordant']}/{m['total']} concordant ({m['frac']}), sign-test p={m['sign_test_p']}")
print("\nwrote outputs/results/totalseg_w1_partial.json")
