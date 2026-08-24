"""LAW -> ATTENTION: predict each organ's SAFE ACCELERATION LIMIT R* from anatomy, so the pipeline knows which
organs need protection/flagging WITHOUT running the acceleration experiment.

R*(organ) = the largest acceleration R at which its Dice stays within `tol` of the unaccelerated (R1) Dice.
Low R* = fragile = "needs attention". We show the LAW (spectral centroid) predicts R*: high-frequency organs
have low R*. -> per-organ reliability table + outputs/plots/{prefix}_rstar.png

  python predict_rstar.py --prefix m0            # MRISeg
  python predict_rstar.py --prefix amos --labels_module amos_labels
"""
import json, os, sys, argparse, importlib, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
sys.path.insert(0, "scripts" if os.path.isdir("scripts") else "."); import labels  # noqa
from paths import RESULTS as RES, PLOTS as PLT

RS = [1, 2, 4, 6, 8]

def rstar(curve, tol):
    """largest R where Dice stays within tol of R1 Dice (linear interp in R); RS[-1] if it never drops below."""
    thr = curve[0] - tol
    for i in range(1, len(RS)):
        if curve[i] < thr:
            (x0, x1), (y0, y1) = (RS[i-1], RS[i]), (curve[i-1], curve[i])
            return float(x0 + (x1 - x0) * (y0 - thr) / (y0 - y1 + 1e-9))
    return float(RS[-1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="m0"); ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--labels_module", default="labels"); a = ap.parse_args()
    L = importlib.import_module(a.labels_module)
    frag = json.load(open(f"{RES}/{a.prefix}_fragility_dice.json"))
    cen = {r["organ"]: r["centroid"] for r in json.load(open(f"{RES}/{a.prefix}_law_v2.json"))["rows"]}

    rows = []
    for o, nm in L.ABDO.items():
        if nm in frag and nm in cen and all(f"R{r}" in frag[nm] for r in RS):
            curve = [frag[nm][f"R{r}"] for r in RS]
            rows.append(dict(organ=nm, tail=o in L.TAIL, rstar=rstar(curve, a.tol), centroid=cen[nm]))
    rows.sort(key=lambda r: r["rstar"])
    xs = np.array([r["centroid"] for r in rows]); ys = np.array([r["rstar"] for r in rows])
    sr = spearmanr(xs, ys).correlation

    print(f"\n=== {a.prefix}: safe acceleration limit R* per organ (tol={a.tol} Dice) ===")
    print(f"{'organ':14}{'tail':5}{'R*':>6}{'centroid':>10}   verdict@R8")
    for r in rows:
        v = "OK" if r["rstar"] >= 8 else ("FLAG (needs attention)" if r["rstar"] < 4 else "watch")
        print(f"{r['organ']:14}{'*' if r['tail'] else '':5}{r['rstar']:6.1f}{r['centroid']:10.3f}   {v}")
    print(f"\nLAW predicts the safe limit: centroid vs R*  Spearman r = {sr:+.2f}  (n={len(rows)})")
    json.dump(dict(tol=a.tol, law_r=float(sr), rows=rows), open(f"{RES}/{a.prefix}_rstar.json", "w"), indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    cols = ["#d93025" if r["tail"] else "#2c5fb0" for r in rows]
    ax[0].barh(range(len(rows)), [r["rstar"] for r in rows], color=cols, alpha=.85)
    ax[0].set_yticks(range(len(rows))); ax[0].set_yticklabels([r["organ"] for r in rows], fontsize=8.5)
    ax[0].axvline(4, color="k", ls="--", lw=.8); ax[0].set_xlabel("safe acceleration limit  R*")
    ax[0].set_title("A. Which organs need attention (low R* = flag)\n(red = tail; dashed = R4)", fontweight="bold", fontsize=10.5)
    ax[1].scatter(xs, ys, s=95, color=cols, zorder=3)
    for r in rows: ax[1].annotate(r["organ"], (r["centroid"], r["rstar"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax[1].set_xlabel("spectral centroid (a priori, from anatomy)"); ax[1].set_ylabel("safe limit R*")
    ax[1].set_title(f"B. The law predicts the safe limit\nSpearman r = {sr:+.2f}", fontweight="bold", fontsize=10.5)
    ax[1].grid(alpha=.3)
    fig.suptitle(f"Law -> attention: predicting each organ's safe acceleration limit ({a.prefix})", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .95]); fig.savefig(f"{PLT}/{a.prefix}_rstar.png", dpi=145)
    print(f"wrote {RES}/{a.prefix}_rstar.json , {PLT}/{a.prefix}_rstar.png")


if __name__ == "__main__":
    main()
