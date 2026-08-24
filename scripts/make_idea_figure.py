"""Conceptual 'our idea' schematic: joint sampling+recon+seg with anatomy-prior gradient rebalancing.
Pulls real M0/M1 numbers for the mini fragility inset. -> outputs/plots/idea_overview.png"""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from paths import RESULTS as R, PLOTS as P

frag = json.load(open(f"{R}/m0_fragility_dice.json"))
ks = json.load(open(f"{R}/m0_kspace_metrics.json"))
Rs = [1, 2, 4, 6, 8]
adr = [frag["adrenal_L"][f"R{r}"] for r in Rs]
liv = [frag["liver"][f"R{r}"] for r in Rs]
ssim = [1.0] + [ks[str(r)]["ssim"] for r in (2, 4, 6, 8)]

fig = plt.figure(figsize=(15, 8.6))
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

BLUE, RED, GREEN, GREY, PRIOR = "#2c5fb0", "#d93025", "#1a9850", "#444", "#7b3fa0"


def box(x, y, w, h, text, fc="#eef2fb", ec=BLUE, fs=10.5, bold=True):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                fc=fc, ec=ec, lw=1.6))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", color="#111")


def arrow(x0, y0, x1, y1, lw=2.2, color=GREY, style="-|>", ms=16, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms,
                                 lw=lw, color=color, linestyle=ls, shrinkA=2, shrinkB=2))


# ---- title ----
ax.text(0.5, 0.965, "Anatomy-prior gradient rebalancing for acceleration-robust multi-organ MRI segmentation",
        ha="center", fontsize=15, fontweight="bold")
ax.text(0.5, 0.928, "As MRI acceleration R↑, small organs collapse long before SSIM/PSNR notice — because the "
        "task-loss gradient is dominated by large organs. A frozen anatomy prior rebalances it.",
        ha="center", fontsize=10.3, color="#333", style="italic")

# ---- top pipeline strip (jointly trained) ----
yp = 0.82
xs = [0.10, 0.30, 0.50, 0.70, 0.905]
labels = ["k-space  y\n(undersampled)", "learned mask\n$M_\\Omega$  (LOUPE /\nPGA-DPS-style)",
          "reconstruction\n$R_\\theta$ (E2E-VarNet)", "segmentation\n$S_\\phi$ (nnU-Net)",
          "per-organ loss\n$L=\\Sigma\\, w_o\\,$DiceCE$(o)$"]
fcs = ["#f1f1f1", "#fbe7e7", "#eef2fb", "#eef2fb", "#eafaf0"]
ecs = [GREY, RED, BLUE, BLUE, GREEN]
for x, lb, fc, ec in zip(xs, labels, fcs, ecs):
    box(x, yp, 0.155, 0.10, lb, fc=fc, ec=ec, fs=9.3)
for i in range(len(xs) - 1):
    arrow(xs[i] + 0.078, yp, xs[i+1] - 0.078, yp, lw=2.4)
ax.annotate("", xy=(0.10, yp - 0.075), xytext=(0.905, yp - 0.075),
            arrowprops=dict(arrowstyle="<->", color=PRIOR, lw=1.4, ls="--"))
ax.text(0.5, yp - 0.105, "trained end-to-end  →  the mask is optimized for segmentation, not just image fidelity",
        ha="center", fontsize=9.2, color=PRIOR, fontweight="bold")

ax.plot([0.02, 0.98], [0.67, 0.67], color="#ccc", lw=1)

# ================= three explanatory panels =================
# ---- (1) THE PROBLEM: mini fragility inset (real data) ----
ax.text(0.17, 0.625, "①  THE PROBLEM  (measured)", ha="center", fontsize=11.5, fontweight="bold", color="#111")
iax = fig.add_axes([0.045, 0.295, 0.25, 0.25])
iax.plot(Rs, liv, "-o", color=BLUE, lw=2.2, label="liver Dice")
iax.plot(Rs, adr, "-o", color=RED, lw=2.2, label="adrenal Dice")
iax.set_ylim(0.3, 1.02); iax.set_xlabel("acceleration R", fontsize=8.5); iax.set_ylabel("Dice", fontsize=8.5)
iax.tick_params(labelsize=7.5)
i2 = iax.twinx(); i2.plot(Rs, ssim, "--s", color=GREEN, lw=1.8); i2.set_ylim(0.3, 1.02)
i2.set_ylabel("SSIM", color=GREEN, fontsize=8.5); i2.tick_params(labelsize=7.5, colors=GREEN)
iax.legend(fontsize=7.5, loc="lower left")
iax.set_title("adrenal −31% · liver −2% · SSIM flat", fontsize=8.5, fontweight="bold")
ax.text(0.17, 0.125, "Small organs crater under acceleration;\nimage metrics stay high (metric-blindness).",
        ha="center", fontsize=9.3, color="#333")

# ---- (2) THE MECHANISM: imbalanced gradient ----
cx = 0.50
ax.text(cx, 0.625, "②  THE MECHANISM  (measured)", ha="center", fontsize=11.5, fontweight="bold", color="#111")
box(cx, 0.20, 0.20, 0.075, "organ-agnostic loss\n$w_o=1$", fc="#f1f1f1", ec=GREY, fs=9.3)
box(cx - 0.085, 0.49, 0.12, 0.07, "LIVER\n(big)", fc="#eef2fb", ec=BLUE, fs=9)
box(cx + 0.085, 0.49, 0.12, 0.07, "adrenal\n(tiny)", fc="#fbe7e7", ec=RED, fs=8.5)
arrow(cx - 0.03, 0.238, cx - 0.085, 0.452, lw=11, color=BLUE)      # thick -> liver
arrow(cx + 0.03, 0.238, cx + 0.085, 0.452, lw=1.2, color=RED)      # thin  -> adrenal
ax.text(cx - 0.16, 0.37, "×44", color=BLUE, fontsize=12, fontweight="bold")
ax.text(cx + 0.115, 0.34, "×1", color=RED, fontsize=10, fontweight="bold")
ax.text(cx, 0.115, "Liver dominates the CE gradient (44×, sustained)\n→ the fragile organ starves.",
        ha="center", fontsize=9.3, color="#333")

# ---- (3) OUR IDEA: rebalanced ----
dx = 0.83
ax.text(dx, 0.625, "③  OUR IDEA  (the fix)", ha="center", fontsize=11.5, fontweight="bold", color=PRIOR)
box(dx, 0.565, 0.21, 0.062, "frozen anatomy prior\n(TotalSeg-MRI)", fc="#f3ecfb", ec=PRIOR, fs=8.8)
box(dx, 0.20, 0.22, 0.075, "rebalanced loss\n$w_o \\propto$ fragility", fc="#f3ecfb", ec=PRIOR, fs=9.3)
arrow(dx, 0.534, dx, 0.243, lw=2, color=PRIOR, ls="--")
ax.text(dx + 0.135, 0.39, "per-organ\nweights $w_o$", color=PRIOR, fontsize=8.6, ha="center", fontweight="bold")
box(dx - 0.085, 0.49, 0.12, 0.07, "LIVER", fc="#eef2fb", ec=BLUE, fs=9)
box(dx + 0.085, 0.49, 0.12, 0.07, "adrenal", fc="#fbe7e7", ec=RED, fs=8.5)
arrow(dx - 0.03, 0.238, dx - 0.085, 0.452, lw=4.5, color=BLUE)     # balanced
arrow(dx + 0.03, 0.238, dx + 0.085, 0.452, lw=6.5, color=RED)      # boosted adrenal
ax.text(dx - 0.16, 0.37, "×1", color=BLUE, fontsize=10, fontweight="bold")
ax.text(dx + 0.115, 0.36, "×4", color=RED, fontsize=12, fontweight="bold")
ax.text(dx, 0.115, "Up-weight the fragile organs' CE gradient\n→ recover small organs at high R.",
        ha="center", fontsize=9.3, color=PRIOR, fontweight="bold")

ax.text(0.5, 0.03, "Contribution: (1) first per-organ fragility benchmark for abdominal MRI acceleration  ·  "
        "(2) falsifiable gradient-imbalance mechanism  ·  (3) fragility-rebalanced task loss.",
        ha="center", fontsize=9.6, color="#111", fontweight="bold")

fig.savefig(f"{P}/idea_overview.png", dpi=145)
print(f"wrote {P}/idea_overview.png")
