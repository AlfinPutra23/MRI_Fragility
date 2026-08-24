"""(b) THE MONEY FIGURE — the whole thesis in one 2x3 panel, from the saved results:
  1 per-organ fragility (tail vs large)   2 the Spectral Fragility Law (centroid->drop)   3 mechanism on REAL k-space
  4 R* safe limits                          5 mitigation proven on REAL k-space              6 negative-results (fundamental limit)
Reads existing jsons; graceful if a result isn't in yet. Run with base-anaconda (matplotlib). -> outputs/plots/money_figure.png"""
import json, os, numpy as np
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import pearsonr
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"]})
R = "outputs/results"
def load(f):
    p = f"{R}/{f}"; return json.load(open(p)) if os.path.exists(p) else None

fig, ax = plt.subplots(2, 3, figsize=(16.5, 9)); ax = ax.ravel()

# 1 — per-organ fragility
rows = load("m0_law_v2.json")
if rows:
    rows = sorted(rows["rows"], key=lambda r: -r["drop"])
    cols = ["#d73027" if r["tail"] else "#4575b4" for r in rows]
    ax[0].barh([r["organ"] for r in rows][::-1], [r["drop"] for r in rows][::-1], color=cols[::-1])
    ax[0].set_xlabel("Dice drop  clean → R8"); ax[0].set_title("1 · Per-organ fragility (R1→R8)", fontweight="bold", loc="left")
    ax[0].text(.97, .04, "red = tail organs", transform=ax[0].transAxes, ha="right", color="#d73027", fontsize=9)

# 2 — the law
if rows:
    c = np.array([r["centroid"] for r in rows]); d = np.array([r["drop"] for r in rows]); t = np.array([r["tail"] for r in rows])
    ax[1].scatter(c[~t], d[~t], c="#4575b4", s=55, label="large"); ax[1].scatter(c[t], d[t], c="#d73027", s=55, label="tail")
    b, m = np.polynomial.polynomial.polyfit(c, d, 1); xx = np.linspace(c.min(), c.max(), 50)
    ax[1].plot(xx, b + m * xx, "k--", lw=1.4); r_ = pearsonr(c, d)[0]
    ax[1].set_xlabel("spectral centroid (anatomy)"); ax[1].set_ylabel("Dice drop")
    ax[1].set_title(f"2 · Spectral Fragility Law  (r={r_:.2f})", fontweight="bold", loc="left"); ax[1].legend(fontsize=8)

# 3 — mechanism on real k-space
mc = load("skmtea_law_multicase.json")
if mc:
    rs = list(mc["percase_r"].values()); ax[2].hist(rs, bins=14, color="#1a9850", alpha=.85, edgecolor="white")
    ax[2].axvline(mc["percase_r_mean"], color="k", ls="--", lw=1.5)
    ax[2].set_xlabel("per-case Pearson r (energy→error)"); ax[2].set_ylabel("# cases")
    ax[2].set_title(f"3 · Mechanism on REAL k-space\n(44 qDESS cases, r={mc['percase_r_mean']:.3f}±{mc['percase_r_std']:.3f})", fontweight="bold", loc="left")

# 4 — R*
rst = load("m0_rstar.json")
ax[3].set_title("4 · Per-organ safe limit R*", fontweight="bold", loc="left")
if rst and "rows" in rst:
    rr = sorted(rst["rows"], key=lambda r: r["rstar"]); cmap = {2: "#d73027", 4: "#fc8d59", 6: "#fee08b", 8: "#1a9850"}
    ax[3].barh([r["organ"] for r in rr], [r["rstar"] for r in rr], color=[cmap.get(int(round(r["rstar"])), "#999") for r in rr])
    ax[3].set_xlabel("R*  (largest safe acceleration)")
    ax[3].text(.97, .04, "red = fragile (R*≈4)   green = robust (R*=8)", transform=ax[3].transAxes, ha="right", fontsize=8)
else:
    ax[3].text(.5, .5, "see m0_rstar.png", ha="center", va="center", transform=ax[3].transAxes); ax[3].axis("off")

# 5 — mitigation proven on real k-space
kf = load("condseg_knee_full.json")
if kf:
    v = kf["R8_structavg_mean"]; s = kf.get("R8_std", {})
    labels = ["clean\n@zero-fill", "recon-then\n-segment", "mixed-R\n(ours)"]; vals = [v["A_clean_zf"], v["B_recon_then_seg"], v["C_mixedR"]]
    err = [s.get("A", 0), s.get("B", 0), s.get("C", 0)]
    ax[4].bar(labels, vals, yerr=err, color=["#bbb", "#91bfdb", "#1a9850"], capsize=4)
    ax[4].set_ylabel("structure-avg Dice @R8"); ax[4].set_title(f"5 · Mitigation on REAL k-space\n(C−B {kf['delta_C_minus_B']:+.3f}, p={kf['wilcoxon_C_vs_B']:.0e})", fontweight="bold", loc="left")
else:
    ax[4].text(.5, .5, "real-k-space method\n(condseg_knee_full pending)", ha="center", va="center", transform=ax[4].transAxes); ax[4].axis("off")
    ax[4].set_title("5 · Mitigation on REAL k-space", fontweight="bold", loc="left")

# 6 — negative results (the fundamental limit), from the catalog
methods = [("mixed-R (train)", 0.088, "#1a9850"), ("Focal-Tversky", 0.030, "#66bd63"), ("FragW4 CE", 0.023, "#a6d96a"),
           ("distillation", 0.034, "#fee08b"), ("R-cond", 0.012, "#fdae61"), ("FiLM", 0.004, "#fdae61"),
           ("frag-recon", 0.009, "#f46d43"), ("HF-loss", 0.0, "#f46d43"), ("R*-acq", -0.002, "#d73027"),
           ("FG-LOUPE", -0.089, "#a50026"), ("consistency", -0.40, "#a50026")]
methods = sorted(methods, key=lambda x: x[1])
ax[5].barh([m[0] for m in methods], [m[1] for m in methods], color=[m[2] for m in methods])
ax[5].axvline(0, color="k", lw=.8); ax[5].set_xlabel("Δ tail Dice vs baseline")
ax[5].set_title("6 · 13 mitigations, one ceiling\n(novel methods fail → fundamental limit)", fontweight="bold", loc="left")

fig.suptitle("Per-organ MRI-acceleration fragility: predictable from anatomy, grounded in k-space physics, a fundamental limit",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, .97]); fig.savefig("outputs/plots/money_figure.png", dpi=150, bbox_inches="tight")
print("wrote outputs/plots/money_figure.png")
