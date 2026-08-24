"""Visual schematics (pictures, not words) for the paper/talk:
  1. pipeline_architecture.png : the forward pipeline (acquire->segment->measure) + the PREDICT (law) branch
  2. idea_schematic.png        : WHY organs break -- shape -> k-space frequency -> acceleration -> the law
  3. pareto_frontier.png       : acceleration<->accuracy Pareto per organ, with the safe-limit R*
"""
import json, os, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Circle, Rectangle
sys.path.insert(0, "scripts" if os.path.isdir("scripts") else "."); import labels as L
from paths import RESULTS as R, PLOTS as P

BLUE, GREEN, ORANGE, RED, GREY = "#2c5fb0", "#1a9850", "#e08214", "#d93025", "#dddddd"

def box(ax, x, y, w, h, text, fc, fs=9.5, tc="black", ec="#444"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.015",
                                fc=fc, ec=ec, lw=1.4, alpha=.95))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, zorder=5)

def arr(ax, x0, y0, x1, y1, color="#333", lw=1.8, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=15, lw=lw, color=color))


# ============================ 1. PIPELINE / ARCHITECTURE ============================
def pipeline():
    fig, ax = plt.subplots(figsize=(14.5, 7.2)); ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis("off")
    ax.text(50, 50, "Per-organ fragility under k-space acceleration: pipeline", ha="center",
            fontsize=15, fontweight="bold")
    # --- forward path (top) ---
    ax.text(2, 45.5, "FORWARD  (acquire → segment → measure)", fontsize=10.5, fontweight="bold", color="#333")
    y = 36; w, h = 15, 6.5; xs = [1, 17.5, 34, 50.5, 67, 84]
    steps = [("Clean MRI\nimage", GREY), ("k-space\n(FFT)", BLUE),
             ("Undersample ×R\n(learned / fixed mask)", BLUE), ("Aliased image\n(+ optional recon-Net)", BLUE),
             ("Segmentation\nU-Net / nnU-Net", GREEN), ("Per-organ\nprediction", GREEN)]
    for (t, c), x in zip(steps, xs):
        box(ax, x, y, w, h, t, c)
    for i in range(len(xs) - 1):
        arr(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)
    # measure box
    box(ax, 84, 24, 15, 6.5, "Per-organ Dice\nvs ground truth", ORANGE)
    arr(ax, 91.5, y, 91.5, 30.5)
    box(ax, 60, 24, 20, 6.5, "FRAGILITY = Dice drop\nas R: 1 → 2 → 4 → 6 → 8", ORANGE, fs=9)
    arr(ax, 84, 27.2, 80, 27.2)
    ax.text(70, 21.5, "sweep acceleration R", fontsize=8.5, style="italic", ha="center", color="#a05a00")

    # --- predict / law branch (bottom) ---
    ax.text(2, 17.5, "PREDICT  (the law — no acceleration experiment needed)", fontsize=10.5, fontweight="bold", color=RED)
    yb = 8
    lb = [("Anatomy\n(GT organ)", GREY), ("Shape + spectral\ncentroid (SA/V, HF)", "#f4c9c2"),
          ("LAW:  centroid → fragility\n(Spearman r ≈ 0.86, 2 datasets)", "#f4a79b"),
          ("Predict safe limit R*\nper organ (r ≈ −0.86)", "#ef8b7c"), ("FLAG fragile organs\n(adrenals, gallbladder…)", RED)]
    xb = [1, 17.5, 37, 61, 82]; wb = [14, 17, 22, 20, 17]
    for (t, c), x, ww in zip(lb, xb, wb):
        box(ax, x, yb, ww, 6.8, t, c, fs=8.8, tc=("white" if c == RED else "black"))
    for i in range(len(xb) - 1):
        arr(ax, xb[i] + wb[i], yb + 3.4, xb[i + 1], yb + 3.4, color=RED)
    # link anatomy of forward -> predict
    arr(ax, 8.5, y, 8, yb + 6.8, color="#999", lw=1.3, style="-|>")
    ax.text(11.5, 21, "same organ,\npredicted a priori", fontsize=8, color="#777", style="italic")
    fig.tight_layout(); fig.savefig(f"{P}/pipeline_architecture.png", dpi=150, bbox_inches="tight")
    print("wrote pipeline_architecture.png")


# ============================ 2. IDEA SCHEMATIC (the WHY) ============================
def idea():
    fig, ax = plt.subplots(figsize=(13.5, 6.4)); ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")
    ax.text(50, 47.5, "Why some organs break: high-frequency content is what acceleration destroys",
            ha="center", fontsize=14, fontweight="bold")
    # two organs
    def organ_row(cy, name, shape, sav, freq_hi, verdict, vcol):
        # shape glyph
        if shape == "blob":
            ax.add_patch(Circle((9, cy), 4.2, fc=BLUE, ec="#123", lw=1.5))
        else:
            ax.add_patch(Ellipse((9, cy), 2.2, 8.6, fc=RED, ec="#311", lw=1.5))
        ax.text(9, cy - 7.2, name, ha="center", fontsize=10, fontweight="bold")
        ax.text(20, cy, f"SA/V {sav}\nspectral centroid {'HIGH' if freq_hi else 'low'}", fontsize=9, va="center")
        # k-space energy strip: center=low freq, edges=high freq
        x0, ww = 34, 30
        for i in range(60):
            fx = i / 59.0
            # blob: energy concentrated center; thin: energy spread to edges
            e = np.exp(-((fx - .5) ** 2) / (2 * (0.08 if not freq_hi else 0.32) ** 2))
            ax.add_patch(Rectangle((x0 + fx * ww, cy - 3), ww / 60, 6, fc=str(1 - min(e, 1) * 0.9), ec="none"))
        ax.add_patch(Rectangle((x0, cy - 3), ww, 6, fc="none", ec="#333", lw=1.2))
        # acceleration keeps center band only
        ax.add_patch(Rectangle((x0 + ww * .5 - ww * .0625, cy - 3), ww * .125, 6, fc="none", ec=GREEN, lw=2.2))
        ax.text(x0 + ww / 2, cy + 4.2, "kept @R8 (center)", fontsize=7.5, ha="center", color=GREEN)
        ax.text(x0 + 2, cy - 4.6, "high-freq (dropped)", fontsize=7, color="#b00")
        ax.text(x0 + ww - 2, cy - 4.6, "high-freq (dropped)", fontsize=7, color="#b00", ha="right")
        arr(ax, x0 + ww + 1, cy, 70, cy, color="#555")
        box(ax, 71, cy - 3, 26, 6, verdict, vcol, fs=9.2, tc="white")
    organ_row(35, "liver (blob)", "blob", "low", False, "energy is in the kept center\n→ SURVIVES (Dice 0.99→0.97)", GREEN)
    organ_row(16, "adrenal (thin)", "thin", "high", True, "energy was in the dropped edges\n→ BREAKS (Dice 0.64→0.44)", RED)
    ax.text(50, 3.0, "Double jeopardy: acceleration removes the high frequencies  ×  networks learn high frequencies "
            "LAST (spectral bias)\n⇒ the law: an organ's spectral centroid predicts its fragility",
            ha="center", fontsize=9.6, style="italic", color="#333")
    fig.tight_layout(); fig.savefig(f"{P}/idea_schematic.png", dpi=150, bbox_inches="tight")
    print("wrote idea_schematic.png")


# ============================ 3. PARETO: acceleration vs accuracy ============================
def pareto():
    frag = json.load(open(f"{R}/m0_fragility_dice.json"))
    rstar = {r["organ"]: r for r in json.load(open(f"{R}/m0_rstar.json"))["rows"]}
    Rs = [1, 2, 4, 6, 8]
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    for o, nm in L.ABDO.items():
        if nm not in frag: continue
        y = [frag[nm][f"R{r}"] for r in Rs]; tail = o in L.TAIL
        ax.plot(Rs, y, "-o", color=(RED if tail else BLUE), alpha=(.9 if tail else .5),
                lw=(2.2 if tail else 1.3), ms=(5 if tail else 3), zorder=(3 if tail else 2))
        if nm in ("adrenal_L", "gallbladder", "esophagus", "liver", "pancreas", "kidney_L"):
            ax.annotate(nm, (8, y[-1]), fontsize=8, xytext=(4, -1), textcoords="offset points",
                        color=(RED if tail else BLUE))
        # R* marker
        if nm in rstar and rstar[nm]["rstar"] < 8:
            rs = rstar[nm]["rstar"]
            ax.scatter([rs], [np.interp(rs, Rs, y)], marker="v", s=70,
                       color=(RED if tail else BLUE), edgecolor="k", zorder=5)
    ax.axhspan(0, 0, color="none")
    ax.plot([], [], "-o", color=RED, label="tail organs (fragile)")
    ax.plot([], [], "-o", color=BLUE, alpha=.5, label="large organs (robust)")
    ax.scatter([], [], marker="v", color="grey", edgecolor="k", label="safe limit R* (Dice drop > 0.05)")
    ax.set_xlabel("acceleration factor  R  (faster scan →)", fontsize=11)
    ax.set_ylabel("segmentation Dice", fontsize=11); ax.set_xticks(Rs)
    ax.set_title("Speed–accuracy Pareto: tail organs fall off early;\nsafe limit R* is predictable from anatomy",
                 fontweight="bold", fontsize=12)
    ax.grid(alpha=.3); ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{P}/pareto_frontier.png", dpi=150)
    print("wrote pareto_frontier.png")


if __name__ == "__main__":
    pipeline(); idea(); pareto()
    print("all schematics written to outputs/plots/")
