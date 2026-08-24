"""Seed-controlled readout of the loss-weighting sweep.

Computes per-organ tail Dice @R8 for each variant and, crucially:
  - SEED FLOOR  = per-case |Uniform_s42 - M0uniform(no-seed)|  (same loss, different seed)
  - WEIGHTING   = per-case (variant - Uniform_s42)             (different loss, SAME seed)
If a variant's tail gain (same-seed) exceeds the seed floor, the weighting genuinely helps.
"""
import os, glob, json, argparse
import numpy as np
import nibabel as nib
from paths import RESULTS as OUT_R, PLOTS as OUT_P
import labels as L

VARIANTS = {  # label -> pred-dir prefix
    "M0uniform(no-seed)": "predsTs",
    "Uniform_s42":        "predsSW_nnUNetTrainer_Uniform_s42",
    "FragW4_s42":         "predsSW_nnUNetTrainer_FragW4_s42",
    "FragW2_s42":         "predsSW_nnUNetTrainer_FragW2_s42",
    "FragTopK_s42":       "predsSW_nnUNetTrainer_FragTopK_s42",
}


def dice(a, b):
    s = a.sum() + b.sum()
    return 2.0 * np.logical_and(a, b).sum() / s if s else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tag", default="R8")
    args = ap.parse_args()
    gts = sorted(glob.glob(f"{args.root}/labelsTs/*.nii.gz"))
    tail = sorted(L.TAIL)

    # per-case per-organ Dice for each variant: D[var][organ] = {case: dice}
    Dv = {v: {o: {} for o in L.ABDO} for v in VARIANTS}
    for v, pref in VARIANTS.items():
        pdir = f"{args.root}/{pref}_{args.tag}" if pref.startswith("predsSW") else f"{args.root}/{pref}_{args.tag}"
        for gp in gts:
            c = os.path.basename(gp)[:-7]
            pp = f"{pdir}/{c}.nii.gz"
            if not os.path.exists(pp):
                continue
            g = np.asanyarray(nib.load(gp).dataobj).astype(np.int16)
            p = np.asanyarray(nib.load(pp).dataobj).astype(np.int16)
            for o in L.ABDO:
                if (g == o).sum() >= 30:
                    Dv[v][o][c] = dice(p == o, g == o)

    def tail_mean(v):
        return float(np.nanmean([np.nanmean(list(Dv[v][o].values())) for o in tail]))

    def paired_delta(va, vb, organs):
        """per-case (va - vb) over organs, matched cases."""
        d = []
        for o in organs:
            common = set(Dv[va][o]) & set(Dv[vb][o])
            d += [Dv[va][o][c] - Dv[vb][o][c] for c in common]
        d = np.array(d)
        return d.mean(), d.std(), int((d > 0.02).sum()), int((d < -0.02).sum()), len(d)

    print(f"\n=== loss-weighting sweep @ {args.tag} (tail organs) ===")
    print(f"{'variant':22} {'tail Dice':>10}")
    for v in VARIANTS:
        print(f"{v:22} {tail_mean(v):10.3f}")

    # seed floor: Uniform_s42 vs M0uniform (same loss, different seed)
    sf_m, sf_s, _, _, sf_n = paired_delta("Uniform_s42", "M0uniform(no-seed)", tail)
    print(f"\nSEED FLOOR  |Uniform_s42 - M0uniform|  (same loss, diff seed):  "
          f"mean Δ={sf_m:+.3f}, per-case std={sf_s:.3f}  (n={sf_n})")

    print(f"\n=== weighting effect (SAME seed: variant - Uniform_s42) on tail @ {args.tag} ===")
    print(f"{'variant':16} {'meanΔ':>8} {'std':>7} {'improved':>9} {'worsened':>9}  verdict")
    out = {"seed_floor_std": sf_s, "seed_floor_mean": sf_m, "tail_dice": {v: tail_mean(v) for v in VARIANTS}, "variants": {}}
    for v in ("FragW4_s42", "FragW2_s42", "FragTopK_s42"):
        m, s, imp, wor, n = paired_delta(v, "Uniform_s42", tail)
        beats = (m > 0.01) and (m > sf_s)            # net gain that exceeds the seed noise floor
        verdict = "BEATS seed noise -> real gain" if beats else "within seed noise / wash"
        print(f"{v:16} {m:+8.3f} {s:7.3f} {imp:9d} {wor:9d}  {verdict}")
        out["variants"][v] = dict(mean_delta=m, std=s, improved=imp, worsened=wor, beats_seed=bool(beats))
    best = max(out["variants"], key=lambda k: out["variants"][k]["mean_delta"])
    print(f"\nBEST: {best} (Δtail={out['variants'][best]['mean_delta']:+.3f}); "
          f"any beats seed noise: {any(d['beats_seed'] for d in out['variants'].values())}")
    json.dump(out, open(f"{OUT_R}/m2_sweep.json", "w"), indent=2)
    print(f"wrote {OUT_R}/m2_sweep.json")


if __name__ == "__main__":
    main()
