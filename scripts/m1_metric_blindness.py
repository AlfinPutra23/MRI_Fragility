"""M1 part (a): metric-blindness dissociation — does image quality predict per-organ segmentability?

For every test case x R, compute the per-case image SSIM (clean vs R-undersampled) and the per-organ
Dice (M0 preds vs GT). Then per organ measure the rank-correlation between image SSIM and Dice:
  - rho_pooled : Spearman over all (case,R) points.
  - rho_withinR: mean over R of the within-R Spearman (removes the global acceleration trend ->
                 the clean test of "at matched R, does a sharper image give a better mask?").

Metric-blindness HOLDS if small 'tail' organs have rho_withinR ~ 0 (image quality carries no info
about whether you can segment them) while large organs track (rho_withinR clearly positive).

Run (after M0 predict):  python m1_metric_blindness.py --root <Dataset501_MRIfrag> --R 2 4 6 8
"""
import os, glob, json, argparse
import numpy as np
import nibabel as nib
from skimage.metrics import structural_similarity as ssim
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from paths import PLOTS as OUT_P, RESULTS as OUT_R
import labels as L


def dice(a, b):
    s = a.sum() + b.sum()
    return 2.0 * np.logical_and(a, b).sum() / s if s else np.nan


def case_ssim(clean, rec, n_central=30):
    """mean per-slice SSIM over up to n_central central axial slices (matches kspace sim)."""
    zc = clean.shape[2] // 2
    z0, z1 = max(zc - n_central // 2, 0), min(zc + n_central // 2, clean.shape[2])
    vals = []
    for z in range(z0, z1):
        ref = clean[:, :, z]
        if ref.max() <= 0:
            continue
        vals.append(ssim(ref, rec[:, :, z], data_range=ref.max()))
    return float(np.mean(vals)) if vals else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset<ID>_MRIfrag dir")
    ap.add_argument("--preds_tpl", default="predsTs_{tag}")
    ap.add_argument("--R", type=float, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--labels_module", default="labels", help="label map module (labels | amos_labels)")
    ap.add_argument("--out_prefix", default="m1", help="output prefix (m1 | amos_m1)")
    args = ap.parse_args()
    import importlib
    global L
    L = importlib.import_module(args.labels_module)
    organs = L.ABDO
    gts = sorted(glob.glob(f"{args.root}/labelsTs/*.nii.gz"))
    print(f"{len(gts)} test cases x R={args.R}")

    # points[o] = list of (ssim, dice) over (case,R);  byR[o][R] = list within that R
    points = {o: [] for o in organs}
    byR = {o: {R: [] for R in args.R} for o in organs}
    for R in args.R:
        tag = f"R{int(R)}"
        pdir = f"{args.root}/{args.preds_tpl.format(tag=tag)}"
        idir = f"{args.root}/imagesTs_{tag}"
        cdir = f"{args.root}/imagesTs_clean"
        nseen = 0
        for gp in gts:
            case = os.path.basename(gp)[:-7]
            pp, cp, rp = f"{pdir}/{case}.nii.gz", f"{cdir}/{case}_0000.nii.gz", f"{idir}/{case}_0000.nii.gz"
            if not (os.path.exists(pp) and os.path.exists(cp) and os.path.exists(rp)):
                continue
            clean = np.asanyarray(nib.load(cp).dataobj).astype(np.float32)
            rec = np.asanyarray(nib.load(rp).dataobj).astype(np.float32)
            s = case_ssim(clean, rec)
            gt = np.asanyarray(nib.load(gp).dataobj).astype(np.int16)
            pr = np.asanyarray(nib.load(pp).dataobj).astype(np.int16)
            for o in organs:
                if (gt == o).sum() >= 30:
                    d = dice(pr == o, gt == o)
                    points[o].append((s, d)); byR[o][R].append((s, d))
            nseen += 1
        print(f"  R={int(R)}: {nseen} cases scored")

    rows = []
    for o, nm in organs.items():
        pts = np.array(points[o]) if points[o] else np.empty((0, 2))
        rho_p = spearmanr(pts[:, 0], pts[:, 1]).correlation if len(pts) >= 3 else np.nan
        wr = []
        for R in args.R:
            a = np.array(byR[o][R])
            if len(a) >= 3 and np.ptp(a[:, 1]) > 0:
                wr.append(spearmanr(a[:, 0], a[:, 1]).correlation)
        rho_w = float(np.nanmean(wr)) if wr else np.nan
        rows.append(dict(id=o, organ=nm, tail=o in L.TAIL, n=len(pts),
                         rho_pooled=float(rho_p), rho_withinR=rho_w))

    rows.sort(key=lambda r: (r["tail"], -(r["rho_withinR"] if not np.isnan(r["rho_withinR"]) else -9)))
    print(f"\n=== image-SSIM vs per-organ Dice rank-correlation ===")
    print(f"{'organ':12} {'tail':5} {'n':>5} {'rho_pooled':>11} {'rho_withinR':>12}")
    for r in rows:
        print(f"{r['organ']:12} {'*' if r['tail'] else ' ':5} {r['n']:5d} "
              f"{r['rho_pooled']:11.3f} {r['rho_withinR']:12.3f}")

    tail_w = np.nanmean([r["rho_withinR"] for r in rows if r["tail"]])
    large_w = np.nanmean([r["rho_withinR"] for r in rows if not r["tail"]])
    print(f"\nmean within-R rho:  TAIL {tail_w:+.3f}   vs   LARGE {large_w:+.3f}")
    holds = (abs(tail_w) < 0.2) and (large_w > tail_w + 0.15)
    print("=> METRIC-BLINDNESS HOLDS: SSIM predicts large-organ Dice but NOT small-organ Dice"
          if holds else "=> WEAK dissociation: SSIM tracks Dice similarly across organ sizes")

    os.makedirs(OUT_R, exist_ok=True)
    json.dump(dict(rows=rows, tail_withinR=float(tail_w), large_withinR=float(large_w),
                   holds=bool(holds)), open(f"{OUT_R}/{args.out_prefix}_metric_blindness.json", "w"), indent=2)
    print(f"wrote {OUT_R}/{args.out_prefix}_metric_blindness.json")

    # figure: per-organ within-R rho (tail red), with the dissociation called out
    rr = [r for r in rows if not np.isnan(r["rho_withinR"])]
    x = np.arange(len(rr))
    cols = ["#d93025" if r["tail"] else "#5b8def" for r in rr]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x, [r["rho_withinR"] for r in rr], color=cols)
    ax.axhline(0, color="k", lw=0.8); ax.axhline(0.2, color="gray", ls="--", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"{'*' if r['tail'] else ''}{r['organ']}" for r in rr],
                                         rotation=45, ha="right")
    ax.set_ylabel("within-R Spearman ρ(image SSIM, organ Dice)")
    ax.set_title("Metric-blindness: image quality predicts LARGE-organ Dice (blue) "
                 "but not small 'tail' organs (red)\n"
                 f"mean within-R ρ:  tail {tail_w:+.2f}  vs  large {large_w:+.2f}",
                 fontsize=10, fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#5b8def", label="large organ"),
                       Patch(color="#d93025", label="small / tail organ")])
    fig.tight_layout(); fig.savefig(f"{OUT_P}/m1_metric_blindness.png", dpi=140)
    print(f"wrote {OUT_P}/m1_metric_blindness.png")


if __name__ == "__main__":
    main()
