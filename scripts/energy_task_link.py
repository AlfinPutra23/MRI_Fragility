"""W2 (escape the Parseval tautology): show that removed k-space energy predicts the DOWNSTREAM TASK loss (Dice
degradation), not merely image error (which is true by Parseval identity, r~0.978). Uses REAL knee k-space per-structure
(skmtea_law_v2.json): Elost_R = fraction of energy removed at acceleration R; degr_R = Dice drop from R1 at R.
Pooled AND within-R (fixed acceleration) energy->Dice — the within-R link is the non-trivial part (not just 'more R =
more loss').  -> outputs/results/energy_task_link.json"""
import json, numpy as np
from scipy.stats import spearmanr, pearsonr

d = json.load(open("outputs/results/skmtea_law_v2.json"))
rows = d["rows"]; Rs = [2, 4, 6, 8]
E, G = [], []
within = {}
for R in Rs:
    e = [r[f"Elost_R{R}"] for r in rows]; g = [r[f"degr_R{R}"] for r in rows]
    within[f"R{R}"] = {"spearman": round(float(spearmanr(e, g).correlation), 3),
                       "pearson": round(float(pearsonr(e, g)[0]), 3), "n": len(e)}
    E += e; G += g
out = {"n_structures": len(rows), "n_points_pooled": len(E),
       "pooled_energy_to_dicedrop": {"spearman": round(float(spearmanr(E, G).correlation), 3),
                                     "pearson": round(float(pearsonr(E, G)[0]), 3)},
       "within_R_energy_to_dicedrop": within,
       "note": "energy->image-error is r~0.978 by Parseval (trivial); this is energy->TASK(Dice). within-R at R8 "
               "isolates the non-trivial link at a FIXED acceleration."}
json.dump(out, open("outputs/results/energy_task_link.json", "w"), indent=2)
print("=== W2 energy -> TASK(Dice), REAL knee k-space ===")
print(f"  pooled ({len(E)} pts): Spearman {out['pooled_energy_to_dicedrop']['spearman']}, Pearson {out['pooled_energy_to_dicedrop']['pearson']}")
for R in Rs: print(f"  within {f'R{R}':>3}: Spearman {within[f'R{R}']['spearman']}")
print("wrote energy_task_link.json")
