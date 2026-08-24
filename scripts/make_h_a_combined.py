"""Headline figure: surface-to-volume ratio predicts fragility on BOTH datasets.
Reads m0_h_a.json (MRISegmentator) + amos_h_a.json (AMOS22-MRI). -> outputs/plots/h_a_combined.png"""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from paths import RESULTS as R, PLOTS as P

panels = [("MRISegmentator-Abdomen (n=13 organs)", f"{R}/m0_h_a.json"),
          ("AMOS22-MRI (n=11 organs)", f"{R}/amos_h_a.json")]
fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
for ax, (title, path) in zip(axes, panels):
    rows = json.load(open(path))["rows"]
    x = np.array([r["sav"] for r in rows]); y = np.array([r["drop"] for r in rows])
    r_s = spearmanr(x, y).correlation
    for rr in rows:
        ax.scatter(rr["sav"], rr["drop"], s=95, color="#d93025" if rr["tail"] else "#2c5fb0", zorder=3,
                   edgecolor="white", linewidth=0.7)
        ax.annotate(rr["organ"], (rr["sav"], rr["drop"]), fontsize=7.5, xytext=(4, 3), textcoords="offset points")
    # trend line
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50); ax.plot(xs, a + b * xs, "--", color="#555", lw=1.3, zorder=2)
    ax.set_title(f"{title}\nSpearman r = {r_s:+.2f}", fontsize=11, fontweight="bold")
    ax.set_xlabel("surface-to-volume ratio  (a priori, from anatomy)"); ax.grid(alpha=0.3)
axes[0].set_ylabel("fragility:  Dice drop  R1→R8")
from matplotlib.lines import Line2D
axes[1].legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor="#d93025", markersize=9, label="small / tail organ"),
                        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2c5fb0", markersize=9, label="large organ")],
               loc="lower right", fontsize=9)
fig.suptitle("Fragility is predictable from anatomy: surface-to-volume ratio forecasts per-organ collapse on two datasets",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f"{P}/h_a_combined.png", dpi=145)
print(f"wrote {P}/h_a_combined.png")
