"""Per-structure x R fragility MATRIX with effect sizes + multiple-comparison correction (the stats layer the AC asked
for; no re-training). Uses the 5 cross-validation folds (m0 + fold1-4) of per-organ Dice-vs-R. For each organ: mean+/-sd
Dice per R across folds, the R1->R8 drop, Cohen's d, paired-t p, Holm & BH corrected p. Pre-registered fragile set = tail
organs (labels.TAIL). -> outputs/results/perstruct_matrix.json"""
import json, numpy as np
from scipy import stats
import sys; sys.path.insert(0, "scripts")
import labels as L

FOLDS = ["m0", "fold1", "fold2", "fold3", "fold4"]
RS = ["R1", "R2", "R4", "R6", "R8"]
TAIL = set(L.ABDO[o] for o in L.TAIL)
data = [json.load(open(f"outputs/results/{f}_fragility_dice.json")) for f in FOLDS]
organs = [o for o in data[0] if all(o in d for d in data)]

rows = {}; pv = []
for o in organs:
    perR = {R: np.array([d[o][R] for d in data]) for R in RS}
    r1, r8 = perR["R1"], perR["R8"]; drop = r1 - r8
    d_eff = float(drop.mean() / (drop.std(ddof=1) + 1e-9))
    t, p = stats.ttest_rel(r1, r8)
    rows[o] = {**{R: [round(float(perR[R].mean()), 3), round(float(perR[R].std(ddof=1)), 3)] for R in RS},
               "drop_mean": round(float(drop.mean()), 3), "drop_sd": round(float(drop.std(ddof=1)), 3),
               "cohens_d": round(d_eff, 2), "p": float(p), "fragile": o in TAIL}
    pv.append(p)

# Holm-Bonferroni + Benjamini-Hochberg across organs
ps = np.array(pv); m = len(ps); order = np.argsort(ps)
holm = np.ones(m); bh = np.ones(m); running = 0.0
for rank, i in enumerate(order):
    holm[i] = max(running, min(1.0, ps[i] * (m - rank))); running = holm[i]     # monotone-enforced Holm
srt = np.argsort(ps)
bh_raw = np.array([min(1.0, ps[srt[r]] * m / (r + 1)) for r in range(m)])
for r in range(m - 2, -1, -1): bh_raw[r] = min(bh_raw[r], bh_raw[r + 1])         # monotone BH
for r, i in enumerate(srt): bh[i] = bh_raw[r]
for k, o in enumerate(organs):
    rows[o]["p_holm"] = round(float(holm[k]), 4); rows[o]["p_bh"] = round(float(bh[k]), 4)

frag = [o for o in organs if rows[o]["fragile"]]; rob = [o for o in organs if not rows[o]["fragile"]]
out = {"folds": FOLDS, "pre_registered_fragile_set": sorted(frag), "R_levels": RS,
       "fragile_mean_drop": round(float(np.mean([rows[o]["drop_mean"] for o in frag])), 3),
       "robust_mean_drop": round(float(np.mean([rows[o]["drop_mean"] for o in rob])), 3),
       "n_significant_holm_05": int(sum(rows[o]["p_holm"] < 0.05 for o in organs)),
       "organs": dict(sorted(rows.items(), key=lambda kv: -kv[1]["drop_mean"]))}
json.dump(out, open("outputs/results/perstruct_matrix.json", "w"), indent=2)
print(f"{m} organs, 5 folds | fragile mean-drop {out['fragile_mean_drop']} vs robust {out['robust_mean_drop']} | {out['n_significant_holm_05']}/{m} organs Holm-significant drop")
print("worst 3:", [(o, rows[o]["drop_mean"], rows[o]["cohens_d"]) for o in list(out["organs"])[:3]])
print("wrote perstruct_matrix.json")
