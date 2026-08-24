"""FG-TDR ablation results figure: 4 arms on real knee k-space (R8), mean±std + per-seed points + paired-Wilcoxon
verdict. Honest read: the fragility prior helps (FG-TDR > task-adapted, ***), but mixed-R still wins.
-> outputs/plots/fgtdr_results.png"""
import json, re, numpy as np
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"]})

d = json.load(open("outputs/results/fgtdr.json"))
arms = ["task_adapted", "recon_then_seg", "FGTDR", "mixedR"]
labs = ["Task-adapted recon\n(no fragility prior)", "Recon-then-\nsegment", "FG-TDR\n(ours)", "mixed-R\n(train on blur)"]
COL = {"task_adapted": "#aeb9c8", "recon_then_seg": "#7fa8d0", "FGTDR": "#c0392b", "mixedR": "#2ca25f"}
mean = np.array([d["mean"][a] for a in arms]); std = np.array([d["std"][a] for a in arms])

seeds = {}
for ln in open("outputs/logs/fgtdr.log"):
    m = re.search(r"seed(\d): recon.seg ([\d.]+) \| mixedR ([\d.]+) \| task ([\d.]+) \| FG-TDR ([\d.]+)", ln)
    if m: seeds[int(m.group(1))] = dict(recon_then_seg=float(m[2]), mixedR=float(m[3]), task_adapted=float(m[4]), FGTDR=float(m[5]))
def stars(p): return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "n.s."

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.6), gridspec_kw={"width_ratios": [1.35, 1]})

# ---- Panel A: the four arms ----
x = np.arange(4)
axA.bar(x, mean, yerr=std, color=[COL[a] for a in arms], edgecolor="#222", lw=1.2,
        error_kw=dict(ecolor="#555", lw=1.3, capsize=5), width=.66, zorder=2)
for i, a in enumerate(arms):                                   # per-seed dots (reproducibility)
    ys = [seeds[s][a] for s in sorted(seeds)]
    axA.scatter([x[i]] * len(ys), ys, color="#1a1f2b", s=34, zorder=4, edgecolor="w", lw=.8)
for i, a in enumerate(arms):
    axA.text(x[i], mean[i] + std[i] + .012, f"{mean[i]:.3f}", ha="center", fontsize=11, fontweight="bold")
axA.set_xticks(x); axA.set_xticklabels(labs, fontsize=10)
axA.set_ylabel("Structure-averaged Dice @ R8", fontsize=11.5)
axA.set_ylim(0.55, 0.92); axA.set_axisbelow(True); axA.grid(axis="y", alpha=.25)
axA.spines[["top", "right"]].set_visible(False)
# significance brackets (paired Wilcoxon, per-case n=28)
def bracket(i, j, y, txt, col="#333"):
    axA.plot([x[i], x[i], x[j], x[j]], [y, y + .008, y + .008, y], lw=1.3, color=col)
    axA.text((x[i] + x[j]) / 2, y + .012, txt, ha="center", fontsize=10.5, fontweight="bold", color=col)
pv = d["FGTDR_vs_task_adapted"]["p"]; bracket(0, 2, 0.845, f"prior helps  {stars(pv)}", "#c0392b")
pv = d["FGTDR_vs_mixedR"]["p"]; bracket(2, 3, 0.885, f"mixed-R wins  {stars(pv)}", "#2ca25f")
axA.set_title("Four arms on real knee k-space  (n = 28 cases, 2 seeds)", fontsize=12.5, fontweight="bold")

# ---- Panel B: paired deltas from FG-TDR's view ----
comps = [("vs. task-adapted\n(does the prior help?)", d["FGTDR_vs_task_adapted"]),
         ("vs. recon-then-seg", d["FGTDR_vs_recon_then_seg"]),
         ("vs. mixed-R\n(the strong baseline)", d["FGTDR_vs_mixedR"])]
yy = np.arange(len(comps))[::-1]
for y, (name, c) in zip(yy, comps):
    dv = c["delta"]; col = "#2ca25f" if dv > 0 else "#c0392b"
    axB.barh(y, dv, color=col, edgecolor="#222", lw=1.1, height=.55, zorder=2)
    off = .001 if dv > 0 else -.001
    axB.text(dv + off * 6, y, f"{dv:+.3f}  {stars(c['p'])}", va="center",
             ha="left" if dv > 0 else "right", fontsize=11, fontweight="bold", color=col)
    axB.text(-0.031, y, name, va="center", ha="left", fontsize=9.6, color="#333")
axB.axvline(0, color="#222", lw=1.4)
axB.set_xlim(-0.033, 0.033); axB.set_ylim(-0.6, len(comps) - .4)
axB.set_yticks([]); axB.set_xlabel("Δ Dice  (FG-TDR − baseline)", fontsize=11)
axB.spines[["top", "right", "left"]].set_visible(False); axB.set_axisbelow(True); axB.grid(axis="x", alpha=.25)
axB.text(0.026, -0.5, "better →", color="#2ca25f", fontsize=9, fontweight="bold", ha="center")
axB.text(-0.026, -0.5, "← worse", color="#c0392b", fontsize=9, fontweight="bold", ha="center")
axB.set_title("Paired comparison (Wilcoxon)", fontsize=12.5, fontweight="bold")

fig.suptitle("FG-TDR: the fragility prior gives a real, reproducible gain over task-driven recon — but mixed-R still wins",
             fontsize=13.5, fontweight="bold", y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig("outputs/plots/fgtdr_results.png", dpi=150, bbox_inches="tight")
print("wrote outputs/plots/fgtdr_results.png")
print("means:", {a: round(d['mean'][a], 3) for a in arms})
