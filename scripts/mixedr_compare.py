"""Mixed-R upgrade readout (2x2 ablation): training distribution {R2 vs mixed-R} x loss {uniform vs fragW4},
tail Dice @R8 on Dataset501's test set. Does mixed-R training + weighting beat the R2-trained +2.2?"""
import glob, os, json, argparse, numpy as np, nibabel as nib, sys
from scipy.stats import wilcoxon
sys.path.insert(0, "scripts" if os.path.isdir("scripts") else "."); import labels as L

MODELS = {  # label -> pred-dir prefix (all evaluated on Dataset501 labelsTs)
    "uniform-R2":     "predsSW_nnUNetTrainer_Uniform_s42",
    "fragW4-R2":      "predsSW_nnUNetTrainer_FragW4_s42",
    "uniform-mixedR": "predsMIXEDR_nnUNetTrainer_Uniform_s42",
    "fragW4-mixedR":  "predsMIXEDR_nnUNetTrainer_FragW4_s42",
}


def dice(a, b):
    s = a.sum() + b.sum(); return 2 * np.logical_and(a, b).sum() / s if s else np.nan


def per_case_tail(root, pref, tag="R8"):
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
    T = {k: per_case_tail(args.root, p) for k, p in MODELS.items() if os.path.isdir(f"{args.root}/{p}_R8")}
    print(f"\n=== tail Dice @R8 (2x2: train-dist x loss) ===")
    for k in MODELS:
        if k in T: print(f"  {k:16} {np.mean(list(T[k].values())):.3f}  (n={len(T[k])})")
        else: print(f"  {k:16} (not ready)")

    def delta(a, b):
        if a not in T or b not in T: return None
        cc = sorted(set(T[a]) & set(T[b])); d = np.array([T[a][c] - T[b][c] for c in cc])
        return d.mean(), wilcoxon([T[a][c] for c in cc], [T[b][c] for c in cc]).pvalue, int((d>0).sum()), int((d<0).sum())

    print("\n=== key deltas (tail Dice @R8) ===")
    for lab, a, b in [("method @R2 (known +0.022)", "fragW4-R2", "uniform-R2"),
                      ("method @mixed-R", "fragW4-mixedR", "uniform-mixedR"),
                      ("mixed-R training alone (uniform)", "uniform-mixedR", "uniform-R2"),
                      ("BEST vs R2-uniform baseline", "fragW4-mixedR", "uniform-R2")]:
        r = delta(a, b)
        if r: print(f"  {lab:34} Δ {r[0]:+.4f}  p={r[1]:.1e}  ({r[2]}/{r[3]})")
        else: print(f"  {lab:34} (pending)")
    json.dump({k: float(np.mean(list(v.values()))) for k, v in T.items()},
              open("outputs/results/m2_mixedr.json" if os.path.isdir("outputs") else "../outputs/results/m2_mixedr.json", "w"), indent=2)


if __name__ == "__main__":
    main()
