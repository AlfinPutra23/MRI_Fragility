"""UPDATED architecture / overview figure — reflects the CURRENT paper (mechanism on real k-space + mixed-R mitigation +
fundamental-limit finding), replacing the stale pipeline_architecture.png (which showed FG-LOUPE 'learned mask' + SA/V).
-> outputs/plots/architecture_v2.png"""
import numpy as np, matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"]})

fig, ax = plt.subplots(figsize=(16.5, 9.5)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def box(x, y, w, h, text, fc, ec="#333", fs=10.5, bold=False, tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", color=tc, wrap=True)


def arrow(x1, y1, x2, y2, col="#333", lw=2.2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18, lw=lw, color=col))


BLU, GRN, ORA, RED, PUR, GRY = "#3a6bb0", "#2ca25f", "#e6a02c", "#d94f43", "#8c6bb1", "#dddddd"
W, H = 14.5, 9; yF = 82   # FORWARD row

# --- FORWARD: acquire -> segment -> measure fragility ---
ax.text(2, 95, "① MEASURE fragility  (acquire → segment → sweep R)", fontsize=13, fontweight="bold", color="#333")
fwd = [("Clean\nMRI", GRY, "black"), ("k-space\n𝓕", BLU, "white"), ("Undersample ×R\n(variable-density)", BLU, "white"),
       ("Aliased\nimage", BLU, "white"), ("Segment\nnnU-Net", GRN, "white"), ("per-organ\nDice", GRN, "white")]
xs = np.linspace(2, 2 + 5 * 15.5, 6)
for i, (t, fc, tc) in enumerate(fwd):
    box(xs[i], yF, W, H, t, fc, fs=10.5, tc=tc, bold=(i in (0,)))
    if i: arrow(xs[i - 1] + W, yF + H / 2, xs[i], yF + H / 2)
box(xs[-1], yF - 13, W, H, "FRAGILITY =\nDice drop\nR1→R8", ORA, fs=10, bold=True)
arrow(xs[-1] + W / 2, yF, xs[-1] + W / 2, yF - 13 + H)

# --- PREDICT: anatomy -> law -> R* (a priori, no scan) ---
yP = 55
ax.text(2, 68, "② PREDICT it from anatomy  (a priori — no acceleration experiment)", fontsize=13, fontweight="bold", color=RED)
pred = [("Anatomy\n(organ shape)", "#f2d0cc"), ("spectral\ncentroid", "#eeb4ae"), ("LAW: centroid→drop\nr=0.84, 2 datasets, CV", "#e79087"),
        ("R*  safe limit\n(derived tool)", "#dd6f63"), ("FLAG fragile organs\nadrenals, gallbladder", RED)]
xsp = np.linspace(2, 2 + 4 * 19, 5); Wp = 17
for i, (t, fc) in enumerate(pred):
    box(xsp[i], yP, Wp, H, t, fc, fs=9.8, tc="white" if i >= 3 else "black", bold=(i in (2, 3)))
    if i: arrow(xsp[i - 1] + Wp, yP + H / 2, xsp[i], yP + H / 2, col=RED)
arrow(xs[0] + W / 2, yF, xsp[0] + Wp / 2, yP + H, col=RED, lw=1.8)  # clean MRI -> anatomy

# --- WHY: mechanism (Parseval, REAL k-space) ---
box(2, 30, 44, 15, "③ WHY — Double Jeopardy\n\nk-space removes high freq  ×  nets learn high freq last (spectral bias)\n= Parseval: |energy removed| = |error created|\nvalidated on REAL multicoil k-space:  r = 0.978 ± 0.010  (44 cases)", PUR, fs=10.5, tc="white", bold=False)

# --- FIX + LIMIT: mitigation + fundamental limit ---
box(54, 30, 44, 15, "④ FIX + its LIMIT\n\nmixed-R training (train for the degradation)\nrecovers fragile organs on REAL k-space: +0.134 vs recon (p=2e-12)\nBUT 13 methods (acq/recon/train/cond) → one ceiling\n= a FUNDAMENTAL information limit → know your R*", "#2c8f5a", fs=10.5, tc="white")
arrow(xsp[-1] + Wp / 2, yP, 76, 45, col="#2c8f5a", lw=1.8)  # flag -> fix

# banner
box(2, 8, 96, 12, "Contribution:  the PREDICTION framework (anatomy → centroid → R*) + the real-k-space MECHANISM + the proven mixed-R MITIGATION + the fundamental-limit finding\n(NOT a new sampler/reconstructor/loss — those all fail, which is the point)", "#f7f7f7", fs=11, bold=True)

fig.suptitle("Per-organ MRI-acceleration fragility — updated architecture", fontsize=15, fontweight="bold", y=0.99)
fig.savefig("outputs/plots/architecture_v2.png", dpi=150, bbox_inches="tight"); print("wrote outputs/plots/architecture_v2.png")
