"""M0 step 3: per-organ Dice-vs-R fragility curve (the kill/continue gate) + metric-blindness panel.

After nnUNetv2_predict on each imagesTs_R{R} -> predsTs_{tag}/, run this.
Computes per-organ Dice(pred, GT) at each R, the fragility ORDERING (collapse vs organ volume),
and overlays the SSIM/PSNR-vs-R curve (image metrics) to show metric-blindness.

  python fragility_eval.py --root <Dataset501_MRIfrag dir> --preds_glob "predsTs_{tag}" \
         --R 1 2 4 6 8 --kspace_metrics ../outputs/results/m0_kspace_metrics.json
"""
import os, glob, json, argparse
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paths import PLOTS as OUT_P, RESULTS as OUT_R
import labels as L


def dice(a, b):
    s = a.sum() + b.sum()
    return 2.0 * np.logical_and(a, b).sum() / s if s else np.nan


def tag_of(R):
    return "clean" if R <= 1 else f"R{int(R)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset<ID>_MRIfrag dir (has labelsTs/)")
    ap.add_argument("--preds_tpl", default="predsTs_{tag}", help="pred folder template under root")
    ap.add_argument("--R", type=float, nargs="+", default=[1, 2, 4, 6, 8])
    ap.add_argument("--kspace_metrics", default=f"{OUT_R}/m0_kspace_metrics.json")
    ap.add_argument("--labels_module", default="labels", help="label map module (labels | amos_labels)")
    ap.add_argument("--out_prefix", default="m0", help="output filename prefix (m0 | amos)")
    args = ap.parse_args()
    import importlib
    global L
    L = importlib.import_module(args.labels_module)

    gts = sorted(glob.glob(f"{args.root}/labelsTs/*.nii.gz"))
    organs = L.ABDO
    # per-organ Dice at each R: dice_RO[R][organ] = list over cases
    res = {R: {k: [] for k in organs} for R in args.R}
    for R in args.R:
        pdir = f"{args.root}/{args.preds_tpl.format(tag=tag_of(R))}"
        for gp in gts:
            case = os.path.basename(gp)[:-7]
            pp = f"{pdir}/{case}.nii.gz"
            if not os.path.exists(pp):
                continue
            gt = np.asanyarray(nib.load(gp).dataobj).astype(np.int16)
            pr = np.asanyarray(nib.load(pp).dataobj).astype(np.int16)
            for k in organs:
                if (gt == k).sum() >= 30:
                    res[R][k].append(dice(pr == k, gt == k))
    mean = {R: {k: float(np.nanmean(res[R][k])) if res[R][k] else np.nan for k in organs} for R in args.R}

    # ---- print + save table ----
    print(f"\n=== per-organ Dice vs R  (n GT cases = {len(gts)}) ===")
    hdr = "organ        tail " + " ".join(f"R={int(r) if r>1 else 1:>5}" for r in args.R) + "   drop(1->max)"
    print(hdr)
    rows = []
    Rmax = max(args.R)
    for k, nm in organs.items():
        ds = [mean[R][k] for R in args.R]
        drop = (mean[args.R[0]][k] - mean[Rmax][k])
        rows.append((k, nm, ds, drop))
    for k, nm, ds, drop in sorted(rows, key=lambda r: -r[3]):
        star = "*" if k in L.TAIL else " "
        print(f"{nm:12} {star:4} " + " ".join(f"{d:6.3f}" for d in ds) + f"   {drop:+.3f}")
    json.dump({nm: {f"R{int(R)}": mean[R][k] for R in args.R} for k, nm in organs.items()},
              open(f"{OUT_R}/{args.out_prefix}_fragility_dice.json", "w"), indent=2)
    print(f"\nwrote {OUT_R}/{args.out_prefix}_fragility_dice.json")

    # ---- fragility de-risk verdict: do small organs drop more? ----
    tail_drop = np.nanmean([drop for k, nm, ds, drop in rows if k in L.TAIL])
    large_drop = np.nanmean([drop for k, nm, ds, drop in rows if k not in L.TAIL])
    print(f"\nmean Dice drop (R1->R{int(Rmax)}):  TAIL {tail_drop:+.3f}   vs   LARGE {large_drop:+.3f}")
    print("=> PREMISE HOLDS (small organs break first)" if tail_drop > large_drop + 0.03
          else "=> WEAK: small organs do NOT collapse faster -> reconsider")

    # ---- figure: fragility curves + metric-blindness ----
    try:
        km = json.load(open(args.kspace_metrics)) if os.path.exists(args.kspace_metrics) else None
    except (json.JSONDecodeError, OSError):
        km = None   # e.g. /dev/null or missing metrics -> just skip the SSIM overlay
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.4))
    for k, nm in organs.items():
        ds = [mean[R][k] for R in args.R]
        c = "#d93025" if k in L.TAIL else "#5b8def"
        axL.plot(args.R, ds, "-o", color=c, lw=2 if k in L.TAIL else 1.2,
                 alpha=0.95 if k in L.TAIL else 0.5, label=nm if k in L.TAIL else None)
    axL.set_xlabel("acceleration R"); axL.set_ylabel("per-organ Dice")
    axL.set_title("Fragility: small 'tail' organs (red) collapse first", fontweight="bold")
    axL.legend(fontsize=8, title="tail organs", ncol=2); axL.grid(alpha=0.3)

    axR.plot(args.R, [np.nanmean([mean[R][k] for k in L.TAIL]) for R in args.R],
             "-o", color="#d93025", lw=2.5, label="mean tail Dice")
    axR.plot(args.R, [np.nanmean([mean[R][k] for k in organs if k not in L.TAIL]) for R in args.R],
             "-o", color="#5b8def", lw=2.5, label="mean large-organ Dice")
    axR.set_xlabel("acceleration R"); axR.set_ylabel("Dice", color="#333")
    if km:
        ax2 = axR.twinx()
        # km is keyed by integer-R ("2","4","6","8"); clean (R<=1) is fully sampled -> SSIM=1.0
        ssim_at = lambda R: 1.0 if R <= 1 else (km[str(int(R))]["ssim"] if str(int(R)) in km else np.nan)
        ax2.plot(args.R, [ssim_at(R) for R in args.R],
                 "--s", color="#1a9850", label="image SSIM")
        ax2.set_ylabel("image SSIM", color="#1a9850"); ax2.set_ylim(0, 1)
    axR.set_title("Metric-blindness: Dice craters while image SSIM stays high", fontweight="bold")
    axR.legend(loc="lower left", fontsize=9); axR.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT_P}/{args.out_prefix}_fragility_curve.png", dpi=140)
    print(f"wrote {OUT_P}/{args.out_prefix}_fragility_curve.png")


if __name__ == "__main__":
    main()
