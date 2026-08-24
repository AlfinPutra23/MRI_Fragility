"""The three laws, under 5-fold cross-validation (error bars = across-fold std):
  A. SA/V -> fragility        B. spectral centroid -> fragility        C. centroid -> safe-limit R*
-> outputs/plots/laws_crossval.png"""
import json, os, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "scripts" if os.path.isdir("scripts") else "."); import labels as L
from paths import RESULTS as R, PLOTS as P

RED, BLUE = "#d93025", "#2c5fb0"
law = {r["organ"]: r for r in json.load(open(f"{R}/m0_law_v2.json"))["rows"]}      # sav, centroid (fold-independent)
dms = json.load(open(f"{R}/multifold.json"))["drop_mean_std"]                       # drop mean±std across folds
lm = json.load(open(f"{R}/law_multifold.json"))
# R* mean±std per organ across the 5 folds
rst = {}
for rf in [f"{R}/m0_rstar.json"] + [f"{R}/fold{f}_rstar.json" for f in (1, 2, 3, 4)]:
    if os.path.exists(rf):
        for row in json.load(open(rf))["rows"]: rst.setdefault(row["organ"], []).append(row["rstar"])

organs = [nm for o, nm in L.ABDO.items() if nm in law and nm in dms]
tailset = {L.ABDO[o] for o in L.TAIL}
cols = [RED if nm in tailset else BLUE for nm in organs]

def panel(ax, xkey, ykind, rkey, xlab, ylab, title):
    x, y, ye = [], [], []
    for nm in organs:
        x.append(law[nm][xkey])
        if ykind == "drop": m, s = dms[nm]
        else: v = rst.get(nm, [np.nan]); m, s = np.mean(v), np.std(v)
        y.append(m); ye.append(s)
    x, y, ye = np.array(x), np.array(y), np.array(ye)
    ax.errorbar(x, y, yerr=ye, fmt="o", ms=8, ecolor="#888", elinewidth=1.3, capsize=3,
                mfc="none", zorder=2, linestyle="none", color="#555")
    ax.scatter(x, y, s=70, c=cols, zorder=3)
    for nm, xi, yi in zip(organs, x, y):
        if nm in ("adrenal_R", "gallbladder", "colon", "esophagus", "liver", "pancreas", "kidney_L"):
            ax.annotate(nm, (xi, yi), fontsize=7.5, xytext=(4, 3), textcoords="offset points")
    b, a = np.polyfit(x, y, 1); xx = np.linspace(x.min(), x.max(), 40)
    ax.plot(xx, a + b * xx, "--", color="#333", lw=1.5)
    rm, rs = lm[rkey]
    ax.text(.05, .90, f"Spearman r = {rm:+.2f} ± {rs:.2f}", transform=ax.transAxes,
            fontsize=12, fontweight="bold", color="#111")
    ax.text(.05, .82, "(5-fold cross-validated)", transform=ax.transAxes, fontsize=8.5, color="#555")
    ax.set_xlabel(xlab, fontsize=10.5); ax.set_ylabel(ylab, fontsize=10.5)
    ax.set_title(title, fontsize=11.5, fontweight="bold"); ax.grid(alpha=.3)

fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.4))
panel(ax[0], "sav", "drop", "sav_drop_r", "surface-to-volume ratio (SA/V)", "R1→R8 Dice drop",
      "A.  SA/V → fragility")
panel(ax[1], "centroid", "drop", "centroid_drop_r", "spectral centroid", "R1→R8 Dice drop",
      "B.  spectral centroid → fragility")
panel(ax[2], "centroid", "rstar", "centroid_rstar_r", "spectral centroid", "safe acceleration limit R*",
      "C.  centroid → safe limit R*")
fig.suptitle("The three laws hold under 5-fold cross-validation  (red = tail organs; error bars = across-fold std)",
             fontsize=13.5, fontweight="bold", y=1.01)
fig.tight_layout(); fig.savefig(f"{P}/laws_crossval.png", dpi=155, bbox_inches="tight")
print("wrote outputs/plots/laws_crossval.png")
