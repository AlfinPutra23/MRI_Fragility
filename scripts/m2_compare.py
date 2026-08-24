"""M2-entry de-risk readout: does the FRAGILITY-WEIGHTED loss recover tail Dice under acceleration?

Computes the weighted model's per-organ Dice vs R (from predsW_{tag}) and diffs it against the M0 uniform
baseline (outputs/results/m0_fragility_dice.json). Verdict focuses on tail organs at high R.

  python m2_compare.py --root <Dataset501_MRIfrag> --R 1 2 4 6 8
"""
import os, glob, json, argparse
import numpy as np
import nibabel as nib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from paths import RESULTS as OUT_R, PLOTS as OUT_P
import labels as L


def dice(a, b):
    s = a.sum() + b.sum()
    return 2.0 * np.logical_and(a, b).sum() / s if s else np.nan


def tag_of(R):
    return "clean" if R <= 1 else f"R{int(R)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--preds_tpl", default="predsW_{tag}")
    ap.add_argument("--R", type=float, nargs="+", default=[1, 2, 4, 6, 8])
    ap.add_argument("--uniform_json", default=f"{OUT_R}/m0_fragility_dice.json")
    args = ap.parse_args()
    organs = L.ABDO
    gts = sorted(glob.glob(f"{args.root}/labelsTs/*.nii.gz"))

    # weighted model per-organ Dice at each R
    wmean = {R: {} for R in args.R}
    for R in args.R:
        pdir = f"{args.root}/{args.preds_tpl.format(tag=tag_of(R))}"
        acc = {k: [] for k in organs}
        for gp in gts:
            case = os.path.basename(gp)[:-7]
            pp = f"{pdir}/{case}.nii.gz"
            if not os.path.exists(pp):
                continue
            gt = np.asanyarray(nib.load(gp).dataobj).astype(np.int16)
            pr = np.asanyarray(nib.load(pp).dataobj).astype(np.int16)
            for k in organs:
                if (gt == k).sum() >= 30:
                    acc[k].append(dice(pr == k, gt == k))
        for k in organs:
            wmean[R][k] = float(np.nanmean(acc[k])) if acc[k] else np.nan

    uni = json.load(open(args.uniform_json))   # {name: {"R1":d,...}}

    print(f"\n=== fragility-weighted vs uniform (M0) — Dice @ each R ===")
    print(f"{'organ':12} {'tail':5} " + " ".join(f"u/wR{int(r)}" for r in args.R))
    rows = []
    for k, nm in organs.items():
        u = [uni.get(nm, {}).get(f"R{int(r)}", np.nan) for r in args.R]
        w = [wmean[r][k] for r in args.R]
        rows.append((k, nm, u, w))
        cells = " ".join(f"{uu:.2f}/{ww:.2f}" for uu, ww in zip(u, w))
        print(f"{nm:12} {'*' if k in L.TAIL else ' ':5} {cells}")

    Rmax = max(args.R)
    iR = args.R.index(Rmax)
    d_tail = np.nanmean([rows_w[iR] - rows_u[iR] for k, nm, rows_u, rows_w in rows if k in L.TAIL])
    d_large = np.nanmean([rows_w[iR] - rows_u[iR] for k, nm, rows_u, rows_w in rows if k not in L.TAIL])
    print(f"\nΔDice @R{int(Rmax)} (weighted - uniform):  TAIL {d_tail:+.3f}   LARGE {d_large:+.3f}")
    # a real tail recovery must clear run-to-run/seed noise (~0.01 for small organs) AND be preferential to tail
    holds = (d_tail >= 0.02) and (d_tail > 1.5 * d_large)
    if holds:
        print("=> LOSS HELPS: fragility-weighting preferentially recovers tail organs @high R -> build full M2")
    else:
        print(f"=> NULL: tail gain {d_tail:+.3f} is within seed noise and not preferential (large {d_large:+.3f}) "
              f"-> loss-reweighting does NOT fix it; bottleneck is structural / acquisition-side (H2 favored)")

    json.dump({nm: dict(tail=k in L.TAIL,
                        uniform={f"R{int(r)}": u for r, u in zip(args.R, U)},
                        weighted={f"R{int(r)}": w for r, w in zip(args.R, W)})
               for k, nm, U, W in rows} | {"_dTail_R8": float(d_tail), "_dLarge_R8": float(d_large),
                                           "_holds": bool(holds)},
              open(f"{OUT_R}/m2_loss_derisk.json", "w"), indent=2)
    print(f"wrote {OUT_R}/m2_loss_derisk.json")

    # figure: tail mean Dice vs R, uniform vs weighted
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for grp, ids, col in [("tail", L.TAIL, "#d93025"), ("large", [k for k in organs if k not in L.TAIL], "#5b8def")]:
        u = [np.nanmean([uni.get(organs[k], {}).get(f"R{int(r)}", np.nan) for k in ids]) for r in args.R]
        w = [np.nanmean([wmean[r][k] for k in ids]) for r in args.R]
        ax.plot(args.R, u, "--o", color=col, alpha=0.6, label=f"{grp} uniform")
        ax.plot(args.R, w, "-o", color=col, lw=2.5, label=f"{grp} weighted")
    ax.set_xlabel("acceleration R"); ax.set_ylabel("mean Dice")
    ax.set_title(f"M2 loss de-risk: fragility-weighted vs uniform\nΔtail@R{int(Rmax)} = {d_tail:+.3f}",
                 fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT_P}/m2_loss_derisk.png", dpi=140)
    print(f"wrote {OUT_P}/m2_loss_derisk.png")


if __name__ == "__main__":
    main()
