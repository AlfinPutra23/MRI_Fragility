"""λ dose-response: tail Dice @R8 vs weighting strength, each variant vs the seed-matched Uniform baseline.
Plots the dose-response curve (does stronger weighting keep helping, or saturate/over-segment?)."""
import glob, os, json, argparse, numpy as np, nibabel as nib, sys
from scipy.stats import wilcoxon
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "scripts" if os.path.isdir("scripts") else "."); import labels as L

VARIANTS = [("uniform", 1, "predsSW_nnUNetTrainer_Uniform_s42"),
            ("x2", 2, "predsSW_nnUNetTrainer_FragW2_s42"),
            ("x4", 4, "predsSW_nnUNetTrainer_FragW4_s42"),
            ("x6", 6, "predsSW_nnUNetTrainer_FragW6_s42"),
            ("x8", 8, "predsSW_nnUNetTrainer_FragW8_s42")]


def dice(a, b):
    s = a.sum() + b.sum(); return 2 * np.logical_and(a, b).sum() / s if s else np.nan


def per_case_tail(pref, root, tag="R8"):
    out = {}
    for gp in sorted(glob.glob(f"{root}/labelsTs/*.nii.gz")):
        c = os.path.basename(gp)[:-7]; pp = f"{root}/{pref}_{tag}/{c}.nii.gz"
        if not os.path.exists(pp): continue
        g = np.asanyarray(nib.load(gp).dataobj).astype(np.int16); pr = np.asanyarray(nib.load(pp).dataobj).astype(np.int16)
        ds = [dice(pr == o, g == o) for o in L.TAIL if (g == o).sum() >= 30]
        if ds: out[c] = np.nanmean(ds)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); args = ap.parse_args()
    base = per_case_tail(VARIANTS[0][2], args.root)
    print(f"\n=== λ dose-response: tail Dice @R8 vs uniform (n≈{len(base)} cases) ===")
    print(f"{'λ(max w)':10} {'tail Dice':>10} {'Δ vs uniform':>13} {'p':>9} {'improved/worsened':>18}")
    print(f"{'uniform':10} {np.mean(list(base.values())):10.3f} {'—':>13}")
    rows = [(1, np.mean(list(base.values())), 0.0, None, None, None)]
    for name, lam, pref in VARIANTS[1:]:
        w = per_case_tail(pref, args.root)
        if not w:
            print(f"{name:10} (not predicted yet)"); continue
        cc = sorted(set(base) & set(w)); d = np.array([w[c] - base[c] for c in cc])
        p = wilcoxon([base[c] for c in cc], [w[c] for c in cc]).pvalue
        print(f"{name:10} {np.mean([w[c] for c in cc]):10.3f} {d.mean():+13.4f} {p:9.1e} "
              f"{int((d>0).sum())}/{int((d<0).sum()):>8}")
        rows.append((lam, np.mean([w[c] for c in cc]), float(d.mean()), float(p), int((d>0).sum()), int((d<0).sum())))
    json.dump(rows, open(f"outputs/results/m2_lambda.json" if os.path.isdir("outputs") else "../outputs/results/m2_lambda.json", "w"), indent=2)

    lam = [r[0] for r in rows]; dd = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(lam, dd, "-o", color="#d93025", lw=2.5, ms=8)
    for r in rows[1:]:
        ax.annotate(f"{r[2]:+.3f}", (r[0], r[2]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax.axhline(0, color="k", lw=0.8, ls=":"); ax.set_xlabel("weighting strength (max per-organ CE weight)")
    ax.set_ylabel("Δ tail Dice @R8 (vs uniform)"); ax.set_title("λ dose-response: how strong should the fragility weight be?",
                  fontweight="bold"); ax.grid(alpha=0.3)
    out = "outputs/plots/m2_lambda.png" if os.path.isdir("outputs") else "../outputs/plots/m2_lambda.png"
    fig.tight_layout(); fig.savefig(out, dpi=140); print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
