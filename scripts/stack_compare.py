"""STACK 2x2 (audit-corrected, 2026-07-08): does a TRAINING fix (mixed-R augmentation, Dataset504 model) beat the
INPUT fix (CS reconstruction), and do they STACK?  Everything evaluated on ONE test set (Dataset501 labelsTs) with
ONE consistent tail-Dice definition, per-case Wilcoxon.

Why this supersedes the earlier numbers (the apples-to-oranges the audit caught):
  - run_recon_baseline.sh: tail Dice with NO size filter + empty->1.0  (0.6195 / 0.6526)
  - mixedr_compare.py:     tail Dice with >=30-vox filter + nan         (0.650 / 0.738), and on imagesTs_R8 (!=R8zf)
Here all four cells use the >=30-vox/nan definition on the SAME inputs (R8zf, R8cs), so deltas are honest.

Cells (all pred dirs under Dataset501_MRIfrag, evaluated vs labelsTs):
  A clean-trained  @ R8zf  (no fix)          predsRECON_R8zf
  B clean-trained  @ R8cs  (recon only)      predsRECON_R8cs
  C mixedR-trained @ R8zf  (aug only)        predsSTACK_nnUNetTrainer_Uniform_s42_R8zf
  D mixedR-trained @ R8cs  (STACK)           predsSTACK_nnUNetTrainer_Uniform_s42_R8cs
"""
import glob, os, sys, argparse, numpy as np, nibabel as nib
from scipy.stats import wilcoxon
sys.path.insert(0, "scripts" if os.path.isdir("scripts") else "."); import labels as L

D = "nnUNet_raw/Dataset501_MRIfrag"; LABDIR = f"{D}/labelsTs"
TAIL = list(L.TAIL); LARGE = [o for o in L.ABDO if o not in L.TAIL]
MINVOX = 30

CELLS = {  # short key -> (label, pred-dir)
    "A": ("clean  @R8zf  (no fix)",     "predsRECON_R8zf"),
    "B": ("clean  @R8cs  (recon only)", "predsRECON_R8cs"),
    "C": ("mixedR @R8zf  (aug only)",   "predsSTACK_nnUNetTrainer_Uniform_s42_R8zf"),
    "D": ("mixedR @R8cs  (STACK)",      "predsSTACK_nnUNetTrainer_Uniform_s42_R8cs"),
}


def dice(a, b):
    s = a.sum() + b.sum()
    return 2 * np.logical_and(a, b).sum() / s if s else np.nan


def per_case(pred, organs):
    """per-case mean Dice over `organs` present (>=MINVOX voxels) in the GT."""
    out = {}
    for gp in sorted(glob.glob(f"{LABDIR}/*.nii.gz")):
        c = os.path.basename(gp); pp = f"{D}/{pred}/{c}"
        if not os.path.exists(pp):
            continue
        g = np.asanyarray(nib.load(gp).dataobj); p = np.asanyarray(nib.load(pp).dataobj)
        ds = [dice(p == o, g == o) for o in organs if (g == o).sum() >= MINVOX]
        if ds:
            out[c] = float(np.nanmean(ds))
    return out


def per_organ(pred, organs):
    acc = {o: [] for o in organs}
    for gp in sorted(glob.glob(f"{LABDIR}/*.nii.gz")):
        c = os.path.basename(gp); pp = f"{D}/{pred}/{c}"
        if not os.path.exists(pp):
            continue
        g = np.asanyarray(nib.load(gp).dataobj); p = np.asanyarray(nib.load(pp).dataobj)
        for o in organs:
            if (g == o).sum() >= MINVOX:
                acc[o].append(dice(p == o, g == o))
    return {o: (float(np.nanmean(v)) if v else float("nan")) for o, v in acc.items()}


def delta(T, a, b):
    """paired per-case a-b with Wilcoxon; returns (mean, p, n_up, n_dn, n)."""
    if a not in T or b not in T:
        return None
    cc = sorted(set(T[a]) & set(T[b]))
    if not cc:
        return None
    d = np.array([T[a][c] - T[b][c] for c in cc])
    try:
        p = wilcoxon([T[a][c] for c in cc], [T[b][c] for c in cc]).pvalue
    except ValueError:
        p = float("nan")
    return d.mean(), p, int((d > 0).sum()), int((d < 0).sum()), len(cc)


def main():
    global MINVOX
    ap = argparse.ArgumentParser()
    ap.add_argument("--minvox", type=int, default=MINVOX)
    args = ap.parse_args()
    MINVOX = args.minvox

    ready = {k: v for k, (lab, v) in CELLS.items() if os.path.isdir(f"{D}/{v}")}
    tail = {k: per_case(ready[k], TAIL) for k in ready}
    large = {k: per_case(ready[k], LARGE) for k in ready}

    print(f"\n================= STACK 2x2  (tail Dice @R8, >= {MINVOX}-vox filter, n per cell) =================")
    print(f"{'':22}{'TAIL Dice':>12}{'LARGE Dice':>12}{'n':>6}")
    for k, (lab, v) in CELLS.items():
        if k in tail:
            print(f"  {k} {lab:20}{np.mean(list(tail[k].values())):>10.4f}"
                  f"{np.mean(list(large[k].values())):>12.4f}{len(tail[k]):>6}")
        else:
            print(f"  {k} {lab:20}{'(not ready)':>12}")

    print("\n================= key deltas (per-case Wilcoxon) =================")
    checks = [
        ("recon fix on clean model     B-A", "B", "A"),
        ("mixed-R aug on zero-filled    C-A", "C", "A"),
        ("mixed-R aug on recon          D-B", "D", "B"),
        ("STACK vs recon-only           D-B", "D", "B"),
        ("STACK vs aug-only             D-C", "D", "C"),
        ("STACK vs no-fix (total)       D-A", "D", "A"),
    ]
    for lab, a, b in checks:
        r = delta(tail, a, b)
        if r:
            star = "  *sig*" if (r[1] == r[1] and r[1] < 0.05) else ""
            print(f"  {lab:34} Δ {r[0]:+.4f}  p={r[1]:.1e}  ({r[2]}↑/{r[3]}↓, n={r[4]}){star}")
        else:
            print(f"  {lab:34} (pending)")

    # interaction: does recon still add once you train on mixed-R?
    dBA = delta(tail, "B", "A"); dDC = delta(tail, "D", "C")
    if dBA and dDC:
        print(f"\n  INTERACTION  recon-gain(clean)={dBA[0]:+.4f}  recon-gain(mixedR)={dDC[0]:+.4f}"
              f"  -> {'fixes COMPOSE' if dDC[0] > 0.005 else 'recon REDUNDANT once mixed-R (interfere/subsume)'}")

    # per-organ tail for the two most informative cells
    for k in ("B", "D"):
        if k in ready:
            po = per_organ(ready[k], TAIL)
            print(f"\n  per-organ tail [{k} {CELLS[k][0]}]: "
                  + "  ".join(f"{L.ABDO[o]}={po[o]:.3f}" for o in TAIL))

    # machine-readable
    out = {k: {"tail": float(np.mean(list(tail[k].values()))),
               "large": float(np.mean(list(large[k].values()))), "n": len(tail[k])} for k in tail}
    os.makedirs("outputs/results", exist_ok=True)
    import json
    json.dump(out, open("outputs/results/stack_2x2.json", "w"), indent=2)
    print("\n  wrote outputs/results/stack_2x2.json")


if __name__ == "__main__":
    main()
