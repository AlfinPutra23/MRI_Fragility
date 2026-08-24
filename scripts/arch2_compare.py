"""2nd-architecture check: does a ResU-Net (different family) show the SAME per-organ fragility ordering as nnU-Net?
If yes -> fragility is a DATA property, not an nnU-Net artifact. Compares clean->R8 Dice-drop per organ (arch2 vs the
nnU-Net m0 benchmark) + re-tests the centroid law on the 2nd arch. Only organs the 2D ResU-Net actually learns
(clean Dice > 0.2) are compared (a downsampled 2D proxy can't segment the tiniest organs -- stated honestly).
-> outputs/results/arch2_compare.json"""
import json, numpy as np
from scipy.stats import spearmanr, pearsonr

rows = json.load(open("outputs/results/m0_law_v2.json"))["rows"]
nn = {r["organ"]: {"drop": r["drop"], "centroid": r["centroid"]} for r in rows}
a2j = json.load(open("outputs/results/arch2_fragility.json"))
a2, per = a2j["dice_drop_clean_to_R8"], a2j["per_organ_dice"]

organs = [o for o in nn if o in a2 and per[o]["clean"] > 0.2]      # organs the 2nd arch actually learned
nn_drop = np.array([nn[o]["drop"] for o in organs])
a2_drop = np.array([a2[o] for o in organs])
cent = np.array([nn[o]["centroid"] for o in organs])
out = {"n_organs": len(organs), "organs": organs,
       "ordering_nnunet_vs_resunet_spearman": float(spearmanr(nn_drop, a2_drop).correlation) if len(organs) > 2 else None,
       "ordering_pearson": float(pearsonr(nn_drop, a2_drop)[0]) if len(organs) > 2 else None,
       "centroid_law_resunet_pearson": float(pearsonr(cent, a2_drop)[0]) if len(organs) > 2 else None,
       "per_organ": {o: {"nnunet_drop": float(nn[o]["drop"]), "resunet_drop": float(a2[o]),
                         "resunet_clean_dice": float(per[o]["clean"])} for o in organs}}
json.dump(out, open("outputs/results/arch2_compare.json", "w"), indent=2)
print(json.dumps({k: v for k, v in out.items() if k != "per_organ"}, indent=2))
