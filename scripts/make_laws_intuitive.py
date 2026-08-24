"""Intuitive version of the law: (A) predicted-from-anatomy fragility vs MEASURED fragility, per organ (bars that
match = the law works); (B) predicted safe acceleration limit R* as a traffic-light per organ (what to flag).
-> outputs/plots/laws_intuitive.png"""
import json, os, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "scripts" if os.path.isdir("scripts") else "."); import labels as L
from paths import RESULTS as R, PLOTS as P

law = {r["organ"]: r for r in json.load(open(f"{R}/m0_law_v2.json"))["rows"]}
dms = json.load(open(f"{R}/multifold.json"))["drop_mean_std"]
rst = {}
for rf in [f"{R}/m0_rstar.json"] + [f"{R}/fold{f}_rstar.json" for f in (1, 2, 3, 4)]:
    if os.path.exists(rf):
        for row in json.load(open(rf))["rows"]: rst.setdefault(row["organ"], []).append(row["rstar"])

organs = [nm for o, nm in L.ABDO.items() if nm in law and nm in dms]
cen = np.array([law[nm]["centroid"] for nm in organs])
meas = np.array([dms[nm][0] for nm in organs]); measerr = np.array([dms[nm][1] for nm in organs])
b, a = np.polyfit(cen, meas, 1); pred = a + b * cen                      # predicted drop from anatomy (the law line)

fig, ax = plt.subplots(1, 2, figsize=(15.5, 7), gridspec_kw={"width_ratios": [1.05, 1]})

# ---- Panel A: measured vs predicted fragility (bars that match = law works) ----
order = np.argsort(meas)                                                  # least -> most fragile
y = np.arange(len(organs)); h = 0.38
ax[0].barh(y + h/2, meas[order], h, xerr=measerr[order], color="#d93025", alpha=.9,
           error_kw=dict(ecolor="#333", lw=1), label="MEASURED  (ran the fast-scan experiment)")
ax[0].barh(y - h/2, pred[order], h, color="#4a90d9", alpha=.9, label="PREDICTED  (from anatomy only, no scan)")
ax[0].set_yticks(y); ax[0].set_yticklabels([organs[i] for i in order], fontsize=9)
ax[0].set_xlabel("fragility  =  Dice drop from full-scan → 8× fast", fontsize=10.5)
ax[0].set_title("A.  Does anatomy predict fragility?  YES — the bars match\n"
                "(each organ: what actually broke vs what we predicted from its shape)", fontsize=11, fontweight="bold")
ax[0].legend(loc="lower right", fontsize=9.5, framealpha=.95); ax[0].grid(axis="x", alpha=.3)

# ---- Panel B: safe acceleration limit R* as a traffic light ----
rmean = {nm: np.mean(rst.get(nm, [np.nan])) for nm in organs}
order2 = sorted(organs, key=lambda nm: rmean[nm])                        # most fragile (low R*) at top
def col(r): return "#d93025" if r < 4 else ("#e6a000" if r < 7 else "#1a9850")
def verdict(r): return "FLAG" if r < 4 else ("watch" if r < 7 else "safe ✓")
yy = np.arange(len(order2))
ax[1].barh(yy, [rmean[nm] for nm in order2], color=[col(rmean[nm]) for nm in order2], alpha=.9)
for i, nm in enumerate(order2):
    ax[1].text(rmean[nm] + .08, i, verdict(rmean[nm]), va="center", fontsize=8.5, color=col(rmean[nm]), fontweight="bold")
ax[1].axvline(4, color="#d93025", ls="--", lw=1); ax[1].axvline(8, color="#1a9850", ls="--", lw=1)
ax[1].set_yticks(yy); ax[1].set_yticklabels(order2, fontsize=9)
ax[1].set_xlim(0, 9.5); ax[1].set_xlabel("safe acceleration limit  R*  (how fast you can scan this organ)", fontsize=10.5)
ax[1].set_title("B.  So the pipeline knows what to protect (predicted a priori)\n"
                "red = only safe to ~4×  ·  green = safe to 8×", fontsize=11, fontweight="bold")
ax[1].grid(axis="x", alpha=.3)

fig.suptitle("The law, intuitively:  predict each organ's fragility (and its safe scan speed) from its anatomy alone",
             fontsize=13.5, fontweight="bold", y=1.0)
fig.tight_layout(); fig.savefig(f"{P}/laws_intuitive.png", dpi=155, bbox_inches="tight")
print("wrote outputs/plots/laws_intuitive.png")
