"""New site visualizations (matplotlib): (1) loop-closer — mixed-R per-organ gain vs centroid; (2) method landscape —
acquisition-side interventions vs baseline (task-driven LOUPE wins, fragility-prior masks lose). CVD-distinguishable
colors + direct value labels (color is never the sole encoding). -> site/assets/{loop_closer,method_landscape}.png"""
import json, glob, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

plt.rcParams.update({"font.size": 12, "axes.spmines.top": False} if False else {"font.size": 12})
FRAG, ROB = "#E1701A", "#2E6FDB"          # fragile / robust — orange vs blue (CVD-safe pair)
WIN, LOSE = "#0E8A5F", "#C4362B"          # win / lose — teal vs red, always with value labels
INK, MUTE, GRID = "#12151a", "#5b6572", "#e3e7ec"


def style(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTE, length=0); ax.grid(True, color=GRID, lw=0.8, alpha=0.7); ax.set_axisbelow(True)


# ---------- FIG 1: loop-closer ----------
d = json.load(open("outputs/results/abdominal_mixedr_perorgan.json"))["rows"]
cen = np.array([r["centroid"] for r in d]); gain = np.array([r["gain"] for r in d])
frag = np.array([r["tail"] for r in d]); names = [r["organ"] for r in d]
rho = spearmanr(cen, gain).correlation
fig, ax = plt.subplots(figsize=(7.6, 5.2), dpi=140)
style(ax)
for xi, yi, fi in zip(cen, gain, frag):
    ax.scatter(xi, yi, s=150, c=FRAG if fi else ROB, edgecolor="white", lw=1.6, zorder=3)
b = np.polyfit(cen, gain, 1); xs = np.linspace(cen.min(), cen.max(), 50)
ax.plot(xs, np.polyval(b, xs), color=MUTE, lw=2, ls="--", zorder=2)
for lab in ["adrenal_R", "gallbladder", "liver", "kidney_L"]:
    i = names.index(lab)
    ax.annotate(lab.replace("_", " "), (cen[i], gain[i]), textcoords="offset points",
                xytext=(8, 6), fontsize=10, color=INK)
ax.scatter([], [], s=110, c=FRAG, edgecolor="white", label="fragile organ")
ax.scatter([], [], s=110, c=ROB, edgecolor="white", label="robust organ")
ax.legend(frameon=False, loc="upper left", fontsize=11)
ax.set_xlabel("spectral centroid  (mean radial k-space frequency)", color=INK)
ax.set_ylabel("mixed-R Dice gain @ R8", color=INK)
ax.set_title(f"Mixed-R rescues exactly the organs the law flags\nper-organ gain vs centroid: Spearman ρ = {rho:.2f}",
             color=INK, fontsize=13.5, loc="left", pad=12)
fig.tight_layout(); fig.savefig("site/assets/loop_closer.png", bbox_inches="tight", facecolor="white")
print("wrote site/assets/loop_closer.png")


# ---------- FIG 2: method landscape (acquisition side) ----------
def tail(pat):
    fs = sorted(glob.glob(f"outputs/results/{pat}"))
    return np.mean([json.load(open(f))["tail"] for f in fs]) if fs else None

base = tail("b1_acq_vd_s*.json")
arms = [("LOUPE  (learned, task-driven)", tail("b1_acq_loupe_s*.json")),
        ("fgLOUPE  λ0.05  (frag prior)", tail("b1_fgloupe_l0.05_s*.json")),
        ("frag-mask B  (hand prior)", tail("b1_acq_fragB_s*.json")),
        ("frag-mask A  (hand prior)", tail("b1_acq_fragA_s*.json")),
        ("fgLOUPE  λ0.15  (frag prior)", tail("b1_fgloupe_l0.15_s*.json"))]
arms = [(n, v - base) for n, v in arms if v is not None]
arms.sort(key=lambda t: t[1])
fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=140)
style(ax); ax.grid(axis="y", alpha=0)
ys = np.arange(len(arms)); vals = [a[1] for a in arms]
bars = ax.barh(ys, vals, color=[WIN if v > 0 else LOSE for v in vals], height=0.62, zorder=3)
ax.axvline(0, color=MUTE, lw=1.4)
ax.set_yticks(ys); ax.set_yticklabels([a[0] for a in arms], fontsize=11, color=INK)
for y, v in zip(ys, vals):
    ax.annotate(f"{v:+.3f}", (v, y), xytext=(6 if v > 0 else -6, 0), textcoords="offset points",
                va="center", ha="left" if v > 0 else "right", fontsize=11, color=WIN if v > 0 else LOSE, fontweight="bold")
ax.set_xlabel("Δ tail Dice @ R8  vs variable-density baseline", color=INK)
ax.set_title("Task-driven sampling wins; every fragility-prior mask loses\n(the audit finding that motivates target-conditioned LOUPE)",
             color=INK, fontsize=13.5, loc="left", pad=12)
ax.set_xlim(min(vals) * 1.35, max(vals) * 1.5)
fig.tight_layout(); fig.savefig("site/assets/method_landscape.png", bbox_inches="tight", facecolor="white")
print("wrote site/assets/method_landscape.png")
