"""The REAL debunk the AC demanded: is the spectral-centroid law 'just size'? Compute the PARTIAL correlation of
centroid vs per-organ Dice-drop CONTROLLING for volume / surface-area-to-volume / contrast, plus a nested-model R^2 gain
and a size<->centroid dissociation. Marginal ranking (centroid 0.86 > size 0.66) was NOT enough (size 0.66 is not weak).
-> outputs/results/debunk_partial.json"""
import json, numpy as np
from scipy import stats

def merge(lawf, haf):
    law = json.load(open(f"outputs/results/{lawf}"))["rows"]
    ha = {r["organ"]: r for r in json.load(open(f"outputs/results/{haf}"))["rows"]}
    D = []
    for r in law:
        h = ha.get(r["organ"], {})
        vol = h.get("vol_cm3", r.get("vol_cm3", np.nan))
        D.append(dict(organ=r["organ"], drop=r["drop"], centroid=r["centroid"],
                      vol=vol, sav=r.get("sav", h.get("sav", np.nan)), contrast=r.get("contrast", h.get("contrast", np.nan))))
    return [d for d in D if not any(np.isnan([d["drop"], d["centroid"], d["vol"], d["sav"], d["contrast"]]))]

def partial_r(x, y, Z):                        # Pearson r(x,y | Z): correlate residuals after regressing out Z
    Z1 = np.column_stack([np.ones(len(x))] + [np.asarray(z, float) for z in Z])
    rx = x - Z1 @ np.linalg.lstsq(Z1, x, rcond=None)[0]
    ry = y - Z1 @ np.linalg.lstsq(Z1, y, rcond=None)[0]
    r, p = stats.pearsonr(rx, ry); return round(float(r), 3), float(p)

def r2(y, X):                                   # OLS R^2 of y ~ [1, X...]
    A = np.column_stack([np.ones(len(y))] + [np.asarray(x, float) for x in X])
    b = np.linalg.lstsq(A, y, rcond=None)[0]; yh = A @ b
    ss = ((y - yh) ** 2).sum(); return 1 - ss / ((y - y.mean()) ** 2).sum()

def analyze(name, D):
    org = [d["organ"] for d in D]
    drop = np.array([d["drop"] for d in D]); cen = np.array([d["centroid"] for d in D])
    vol = np.array([d["vol"] for d in D]); sav = np.array([d["sav"] for d in D]); con = np.array([d["contrast"] for d in D])
    lv = np.log(vol + 1e-6)
    out = {"n": len(D),
           "marginal_spearman": {"centroid": round(float(stats.spearmanr(cen, drop)[0]), 3),
                                  "log_volume": round(float(stats.spearmanr(lv, drop)[0]), 3),
                                  "sav": round(float(stats.spearmanr(sav, drop)[0]), 3),
                                  "contrast": round(float(stats.spearmanr(con, drop)[0]), 3)},
           "partial_centroid_vs_drop": {
               "| log_vol": partial_r(cen, drop, [lv]),
               "| log_vol,sav": partial_r(cen, drop, [lv, sav]),
               "| log_vol,sav,contrast": partial_r(cen, drop, [lv, sav, con])},
           "nested_R2": {"size+contrast": round(float(r2(drop, [lv, con])), 3),
                         "size+contrast+centroid": round(float(r2(drop, [lv, con, cen])), 3)}}
    out["nested_R2"]["delta_from_centroid"] = round(out["nested_R2"]["size+contrast+centroid"] - out["nested_R2"]["size+contrast"], 3)
    # dissociation: organs where size-rank and centroid-rank disagree most
    rc = stats.rankdata(cen); rv = stats.rankdata(-vol)                 # high centroid vs SMALL volume (both -> fragile per each theory)
    diss = sorted(zip(org, rc - rv, cen, vol, drop), key=lambda t: -abs(t[1]))[:3]
    out["dissociation_top3"] = [{"organ": o, "centroid": round(c, 3), "vol_cm3": round(v, 1), "drop": round(dp, 3),
                                 "note": "high-centroid but large" if d > 0 else "low-centroid but small"} for o, d, c, v, dp in diss]
    return out

res = {"MRISegmentator": analyze("m0", merge("m0_law_v2.json", "m0_h_a.json")),
       "AMOS": analyze("amos", merge("amos_law_v2.json", "amos_h_a.json"))}
json.dump(res, open("outputs/results/debunk_partial.json", "w"), indent=2)
for ds, o in res.items():
    pc = o["partial_centroid_vs_drop"]["| log_vol,sav,contrast"]
    print(f"{ds}: partial r(centroid, drop | size,shape,contrast) = {pc[0]} (p={pc[1]:.3g}) | nested Delta R^2 from centroid = {o['nested_R2']['delta_from_centroid']}")
print("wrote debunk_partial.json  ->  centroid survives size control iff partial r stays high & significant")
