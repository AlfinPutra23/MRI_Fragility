"""Aggregate the LAW v2 (spectral centroid) + R* prediction ACROSS folds -> mean +- std = error bars on the
upgraded law. Reads {m0,fold1..4}_law_v2.json (single-feature Spearman r) + {..}_rstar.json (centroid->R* r).
-> outputs/results/law_multifold.json"""
import json, os, numpy as np
from paths import RESULTS as R

prefixes = [p for p in ["m0", "fold1", "fold2", "fold3", "fold4"] if os.path.exists(f"{R}/{p}_law_v2.json")]
cen_r, sav_r, hf_r, rstar_r = [], [], [], []
for p in prefixes:
    s = json.load(open(f"{R}/{p}_law_v2.json"))["single"]
    cen_r.append(s["centroid"]["r"]); sav_r.append(s["sav"]["r"]); hf_r.append(s.get("hf8_img", {}).get("r", np.nan))
    rp = f"{R}/{p}_rstar.json"
    if os.path.exists(rp): rstar_r.append(json.load(open(rp))["law_r"])

def ms(x):
    x = [v for v in x if v == v]
    return (float(np.mean(x)), float(np.std(x))) if x else (float("nan"), float("nan"))

print(f"\n=== LAW v2 across {len(prefixes)} folds: {prefixes} ===")
print(f"  centroid -> drop   Spearman r = {ms(cen_r)[0]:+.2f} +- {ms(cen_r)[1]:.2f}")
print(f"  SA/V     -> drop   Spearman r = {ms(sav_r)[0]:+.2f} +- {ms(sav_r)[1]:.2f}   (old law, for comparison)")
print(f"  centroid -> R*     Spearman r = {ms(rstar_r)[0]:+.2f} +- {ms(rstar_r)[1]:.2f}   (safe-limit prediction)")
json.dump(dict(prefixes=prefixes, n_folds=len(prefixes),
               centroid_drop_r=ms(cen_r), sav_drop_r=ms(sav_r), hf8_drop_r=ms(hf_r), centroid_rstar_r=ms(rstar_r)),
          open(f"{R}/law_multifold.json", "w"), indent=2)
print(f"wrote {R}/law_multifold.json")
