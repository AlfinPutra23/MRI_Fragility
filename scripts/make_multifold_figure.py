"""Multifold cross-validation figure (the real 'fold 1-4' deliverable):
  A. per-organ R1->R8 Dice drop, mean +- std across folds (tail organs highlighted)
  B. the SA/V -> fragility LAW with across-fold error bars + Spearman r (mean +- std)
Reads outputs/results/multifold.json (from multifold_aggregate.py) + m0_h_a.json (SA/V).
Renders with whatever folds exist -> single fold = no error bars, N folds = cross-validated.
-> outputs/plots/multifold_law.png"""
import json, os, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
sys.path.insert(0, "scripts" if os.path.isdir("scripts") else "."); import labels as L

MF = "outputs/results/multifold.json"
if not os.path.exists(MF):
    sys.exit("multifold.json missing -> run scripts/multifold_aggregate.py first (it needs the per-fold fragility jsons).")
mf = json.load(open(MF)); folds = mf["folds"]; dms = mf["drop_mean_std"]            # organ -> [mean, std]
try:
    sav = {r["organ"]: r["sav"] for r in json.load(open("outputs/results/m0_h_a.json"))["rows"]}
except Exception:
    sav = {}
TAIL = set(L.ABDO[o] for o in L.TAIL)

organs = sorted([o for o in dms if o in sav], key=lambda o: dms[o][0])
means = np.array([dms[o][0] for o in organs]); stds = np.array([dms[o][1] for o in organs])
xs = np.array([sav[o] for o in organs]); cols = ["#d93025" if o in TAIL else "#4285f4" for o in organs]

fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))
# --- A: per-organ drop mean +- std ---
ax[0].barh(range(len(organs)), means, xerr=(stds if stds.any() else None), color=cols, alpha=.85,
           error_kw=dict(ecolor="#222", lw=1.2, capsize=2))
ax[0].set_yticks(range(len(organs))); ax[0].set_yticklabels(organs, fontsize=8.5)
ax[0].set_xlabel("R1→R8 Dice drop (mean ± std across folds)"); ax[0].axvline(0, color="k", lw=.6)
ax[0].set_title(f"A. Fragility is consistent across {len(folds)} fold(s)\n(red = tail organs)", fontweight="bold", fontsize=10.5)
# --- B: the SA/V law with error bars ---
ax[1].errorbar(xs, means, yerr=(stds if stds.any() else None), fmt="o", color="#1a9850",
               ecolor="#999", capsize=3, ms=7)
for o, x, y in zip(organs, xs, means):
    ax[1].annotate(o, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
if len(xs) > 2:
    b, a = np.polyfit(xs, means, 1); xx = np.linspace(xs.min(), xs.max(), 50)
    ax[1].plot(xx, a + b * xx, "--", color="#d93025", lw=1.5)
    r = mf.get("law_r_mean", spearmanr(xs, means).correlation)
    lab = f"Spearman r = {r:+.2f}" + (f" ± {mf['law_r_std']:.2f}" if mf.get("law_r_std") else "")
    ax[1].text(.04, .92, f"{lab}  (across {len(folds)} fold(s))", transform=ax[1].transAxes,
               fontsize=10.5, fontweight="bold")
ax[1].set_xlabel("surface-to-volume ratio (SA/V)"); ax[1].set_ylabel("R1→R8 Dice drop")
ax[1].set_title("B. Anatomy predicts fragility (the law)\n(error bars = across-fold std)", fontweight="bold", fontsize=10.5)

tag = "SINGLE FOLD (0) — error bars appear once folds 1-4 retrain" if len(folds) < 2 else f"{len(folds)}-fold cross-validated"
fig.suptitle(f"Multi-fold fragility benchmark — {tag}", fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig("outputs/plots/multifold_law.png", dpi=145)
print(f"wrote outputs/plots/multifold_law.png  [{tag}; folds={folds}, {len(organs)} organs]")
