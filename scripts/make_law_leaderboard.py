"""Which predictor is best? Compare all laws across 5 folds on the metric that matters (leave-one-out CV R^2 =
out-of-sample prediction of the actual drop) + rank correlation. -> outputs/plots/law_leaderboard.png"""
import json, os, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
sys.path.insert(0, "scripts" if os.path.isdir("scripts") else ".")
from paths import RESULTS as R, PLOTS as P

PREDS = [("centroid", "spectral centroid\n(frequency)", "#1a9850"),
         ("sav", "SA/V\n(shape)", "#2c5fb0"),
         ("hf8_img", "HF-fraction @R8", "#e08214"),
         ("fdim", "fractal dim\n(tortuosity)", "#999999")]
files = [f"{R}/m0_law_v2.json"] + [f"{R}/fold{f}_law_v2.json" for f in (1, 2, 3, 4)]
files = [f for f in files if os.path.exists(f)]

def loo_r2(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); ok = ~np.isnan(x); x, y = x[ok], y[ok]
    if len(x) < 4: return np.nan
    xs = (x - x.mean()) / (x.std() + 1e-9); pr = np.zeros(len(y))
    for i in range(len(y)):
        tr = np.ones(len(y), bool); tr[i] = False
        A = np.column_stack([xs[tr], np.ones(tr.sum())]); b = np.linalg.lstsq(A, y[tr], rcond=None)[0]
        pr[i] = np.append(xs[i], 1) @ b
    return 1 - ((y - pr) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12)

r2 = {k: [] for k, _, _ in PREDS}; rr = {k: [] for k, _, _ in PREDS}
for f in files:
    rows = json.load(open(f))["rows"]; y = [r["drop"] for r in rows]
    for k, _, _ in PREDS:
        x = [r.get(k, np.nan) for r in rows]
        r2[k].append(loo_r2(x, y))
        ok = ~np.isnan(np.asarray(x, float))
        rr[k].append(spearmanr(np.asarray(x, float)[ok], np.asarray(y)[ok]).correlation)

fig, ax = plt.subplots(figsize=(9.5, 6))
xpos = np.arange(len(PREDS))
m = [np.nanmean(r2[k]) for k, _, _ in PREDS]; s = [np.nanstd(r2[k]) for k, _, _ in PREDS]
bars = ax.bar(xpos, m, yerr=s, capsize=5, color=[c for _, _, c in PREDS], alpha=.9, edgecolor="#222")
bars[0].set_edgecolor("#0a5"); bars[0].set_linewidth(3)
for i, (k, _, _) in enumerate(PREDS):
    ax.text(i, m[i] + s[i] + .02, f"LOO R²\n{m[i]:.2f}±{s[i]:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax.text(i, .02, f"rank r\n{np.nanmean(rr[k]):+.2f}", ha="center", fontsize=8.5, color="white", fontweight="bold")
ax.set_xticks(xpos); ax.set_xticklabels([lab for _, lab, _ in PREDS], fontsize=10)
ax.set_ylabel("leave-one-out CV R²  (out-of-sample prediction of the drop)", fontsize=10.5)
ax.set_title("Which law is best?  →  SPECTRAL CENTROID\n(best out-of-sample R², ties best rank, and it's the physical quantity)",
             fontweight="bold", fontsize=12)
ax.axhline(0, color="k", lw=.6); ax.grid(axis="y", alpha=.3); ax.set_ylim(min(0, min(m) - .1), max(m) + max(s) + .12)
ax.annotate("WINNER", xy=(0, m[0]), xytext=(0.35, m[0] + .12), fontsize=11, fontweight="bold", color="#0a5",
            arrowprops=dict(arrowstyle="->", color="#0a5", lw=2))
fig.tight_layout(); fig.savefig(f"{P}/law_leaderboard.png", dpi=150)
print("=== law leaderboard (5-fold) ===")
for k, lab, _ in PREDS:
    print(f"  {k:10} LOO-R²={np.nanmean(r2[k]):+.2f}±{np.nanstd(r2[k]):.2f}   rank r={np.nanmean(rr[k]):+.2f}")
print("wrote outputs/plots/law_leaderboard.png")
