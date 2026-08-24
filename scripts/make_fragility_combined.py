"""Combined two-dataset fragility curves: per-organ Dice vs R on MRISegmentator + AMOS side by side.
-> outputs/plots/fragility_combined.png"""
import json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from paths import RESULTS as R, PLOTS as P

TAIL = {"gallbladder", "esophagus", "pancreas", "adrenal_R", "adrenal_L", "duodenum"}
Rs = [1, 2, 4, 6, 8]
panels = [("MRISegmentator-Abdomen", f"{R}/m0_fragility_dice.json"),
          ("AMOS22-MRI", f"{R}/amos_fragility_dice.json")]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), sharey=True)
for ax, (title, path) in zip(axes, panels):
    d = json.load(open(path))
    tail_seen = large_seen = False
    for nm, curve in d.items():
        ys = [curve[f"R{r}"] for r in Rs]
        is_tail = nm in TAIL
        ax.plot(Rs, ys, "-o", color="#d93025" if is_tail else "#2c5fb0",
                lw=2.4 if is_tail else 1.3, alpha=0.95 if is_tail else 0.5, ms=5,
                label=("small / tail organ" if (is_tail and not tail_seen) else
                       ("large organ" if (not is_tail and not large_seen) else None)))
        # annotate the most fragile
        if nm in ("adrenal_R", "adrenal_L", "liver"):
            ax.annotate(nm, (8, ys[-1]), fontsize=8, xytext=(4, -2), textcoords="offset points")
        tail_seen = tail_seen or is_tail; large_seen = large_seen or (not is_tail)
    ax.set_title(title, fontsize=11.5, fontweight="bold")
    ax.set_xlabel("acceleration factor  R"); ax.grid(alpha=0.3); ax.set_ylim(0.2, 1.0)
    ax.legend(fontsize=9, loc="lower left")
axes[0].set_ylabel("per-organ Dice")
fig.suptitle("Small organs break first on both datasets: per-organ Dice collapses with acceleration (red = small/tail)",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(f"{P}/fragility_combined.png", dpi=145)
print(f"wrote {P}/fragility_combined.png")
