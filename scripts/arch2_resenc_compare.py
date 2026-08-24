"""CLEAN 3D 2nd-architecture test: does nnU-Net ResEnc (ResidualEncoderUNet) show the SAME per-organ fragility ordering
as the plain nnU-Net (PlainConvUNet)? Same 3D pipeline, same eval, different network -> if the ordering matches,
fragility is architecture-independent (not an nnU-Net artifact). Reads resenc_fragility_dice.json + m0_law_v2 rows.
-> outputs/results/arch2_resenc_compare.json"""
import json, numpy as np
from scipy.stats import spearmanr, pearsonr

rows = json.load(open("outputs/results/m0_law_v2.json"))["rows"]
nn = {r["organ"]: {"drop": r["drop"], "centroid": r["centroid"]} for r in rows}
re = json.load(open("outputs/results/resenc_fragility_dice.json"))
organs = [o for o in nn if o in re]
nn_drop = np.array([nn[o]["drop"] for o in organs])
re_drop = np.array([re[o]["R1"] - re[o]["R8"] for o in organs])
cent = np.array([nn[o]["centroid"] for o in organs])
out = {"n_organs": len(organs),
       "ordering_nnunet_vs_resenc_spearman": float(spearmanr(nn_drop, re_drop).correlation),
       "ordering_pearson": float(pearsonr(nn_drop, re_drop)[0]),
       "centroid_law_resenc_pearson": float(pearsonr(cent, re_drop)[0]),
       "per_organ": {o: {"nnunet_drop": float(nn[o]["drop"]), "resenc_drop": float(re[o]["R1"] - re[o]["R8"]),
                         "centroid": float(nn[o]["centroid"])} for o in organs}}
json.dump(out, open("outputs/results/arch2_resenc_compare.json", "w"), indent=2)
print(json.dumps({k: v for k, v in out.items() if k != "per_organ"}, indent=2))
print("\nper-organ (nnU-Net drop -> ResEnc drop):")
for o in sorted(organs, key=lambda o: -nn[o]["drop"]):
    print(f"  {o:14} {nn[o]['drop']:+.3f} -> {re[o]['R1']-re[o]['R8']:+.3f}")
