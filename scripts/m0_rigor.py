"""Statistical + boundary rigor for the M0 fragility benchmark (CPU, on existing preds).

(2) Per-organ Dice vs R with bootstrap 95% CIs; R1->R8 drop with paired Wilcoxon p + paired Cohen's d;
    tail-vs-large drop paired test.
(3) Boundary metrics NSD@(in-plane spacing) and HD95 per organ at R in {1,4,8} (bbox-cropped surface distances).

  python m0_rigor.py --root <Dataset501_MRIfrag>
"""
import os, glob, json, argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt, binary_erosion
from scipy.stats import wilcoxon
from paths import RESULTS as OUT_R
import labels as L


def dice(a, b):
    s = a.sum() + b.sum()
    return 2.0 * np.logical_and(a, b).sum() / s if s else np.nan


def boot_ci(x, n=2000, seed=0):
    x = np.asarray([v for v in x if not np.isnan(v)])
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    bs = [rng.choice(x, len(x), replace=True).mean() for _ in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def cohen_d_paired(d):
    d = np.asarray([v for v in d if not np.isnan(v)])
    return float(d.mean() / d.std(ddof=1)) if len(d) > 1 and d.std() > 0 else np.nan


def surf_dists(a, b, spacing):
    """bbox-cropped surface-to-surface distances (mm), both directions."""
    if a.sum() == 0 or b.sum() == 0:
        return None, None
    ab = a | b
    sl = tuple(slice(max(p.min() - 2, 0), p.max() + 3) for p in np.where(ab))
    a, b = a[sl], b[sl]
    a_s = a ^ binary_erosion(a); b_s = b ^ binary_erosion(b)
    if a_s.sum() == 0 or b_s.sum() == 0:
        return None, None
    dt_b = distance_transform_edt(~b_s, sampling=spacing)
    dt_a = distance_transform_edt(~a_s, sampling=spacing)
    return dt_b[a_s], dt_a[b_s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--preds_tpl", default="predsTs_{tag}")
    ap.add_argument("--R", type=float, nargs="+", default=[1, 2, 4, 6, 8])
    ap.add_argument("--R_bound", type=float, nargs="+", default=[1, 4, 8])
    args = ap.parse_args()
    organs = L.ABDO
    gts = sorted(glob.glob(f"{args.root}/labelsTs/*.nii.gz"))
    tag = lambda R: "clean" if R <= 1 else f"R{int(R)}"

    # per-case per-organ Dice + boundary
    dsc = {R: {o: {} for o in organs} for R in args.R}
    nsd = {R: {o: {} for o in args.R_bound and organs} for R in args.R_bound}
    hd95 = {R: {o: {} for o in organs} for R in args.R_bound}
    for gp in gts:
        c = os.path.basename(gp)[:-7]
        gi = nib.load(gp); g = np.asanyarray(gi.dataobj).astype(np.int16)
        sp = tuple(float(z) for z in gi.header.get_zooms()[:3]); tau = min(sp[0], sp[1])
        for R in args.R:
            pp = f"{args.root}/{args.preds_tpl.format(tag=tag(R))}/{c}.nii.gz"
            if not os.path.exists(pp):
                continue
            p = np.asanyarray(nib.load(pp).dataobj).astype(np.int16)
            for o in organs:
                gm = g == o
                if gm.sum() < 30:
                    continue
                pm = p == o
                dsc[R][o][c] = dice(pm, gm)
                if R in args.R_bound:
                    da, db = surf_dists(pm, gm, sp)
                    if da is not None:
                        nsd[R][o][c] = (np.sum(da <= tau) + np.sum(db <= tau)) / (len(da) + len(db))
                        hd95[R][o][c] = max(np.percentile(da, 95), np.percentile(db, 95))

    Rmax = max(args.R); R1 = args.R[0]
    rows = []
    print(f"\n=== M0 per-organ Dice vs R with 95% CI + R{int(R1)}->R{int(Rmax)} drop stats (n={len(gts)}) ===")
    print(f"{'organ':12} {'tail':4} {'Dice@R1':>16} {'Dice@R8':>16} {'drop':>7} {'p(wilcox)':>10} {'d':>6}")
    for o, nm in organs.items():
        d1 = dsc[R1][o]; d8 = dsc[Rmax][o]
        common = sorted(set(d1) & set(d8))
        a = np.array([d1[c] for c in common]); b = np.array([d8[c] for c in common])
        drop = a - b
        m1, m8 = np.nanmean(a), np.nanmean(b)
        ci1, ci8 = boot_ci(a), boot_ci(b)
        try:
            p = wilcoxon(a, b).pvalue if len(common) > 5 and np.any(a != b) else np.nan
        except Exception:
            p = np.nan
        d = cohen_d_paired(drop)
        rows.append(dict(organ=nm, tail=o in L.TAIL, dice_R1=m1, ci_R1=ci1, dice_R8=m8, ci_R8=ci8,
                         drop=float(np.nanmean(drop)), wilcoxon_p=float(p), cohen_d=d,
                         nsd_R1=float(np.nanmean(list(nsd.get(R1, {}).get(o, {}).values()) or [np.nan])),
                         nsd_R8=float(np.nanmean(list(nsd.get(Rmax, {}).get(o, {}).values()) or [np.nan])),
                         hd95_R1=float(np.nanmean(list(hd95.get(R1, {}).get(o, {}).values()) or [np.nan])),
                         hd95_R8=float(np.nanmean(list(hd95.get(Rmax, {}).get(o, {}).values()) or [np.nan]))))
        print(f"{nm:12} {'*' if o in L.TAIL else ' ':4} {m1:.3f}[{ci1[0]:.2f},{ci1[1]:.2f}]  "
              f"{m8:.3f}[{ci8[0]:.2f},{ci8[1]:.2f}] {np.nanmean(drop):+7.3f} {p:10.2e} {d:6.2f}")

    # boundary table
    print(f"\n=== boundary metrics: NSD@{'inplane':>0} / HD95 (mm) at R1 vs R8 ===")
    print(f"{'organ':12} {'NSD_R1':>7} {'NSD_R8':>7} {'HD95_R1':>8} {'HD95_R8':>8}")
    for r in rows:
        print(f"{r['organ']:12} {r['nsd_R1']:7.3f} {r['nsd_R8']:7.3f} {r['hd95_R1']:8.1f} {r['hd95_R8']:8.1f}")

    # tail vs large, per-case paired
    per_case_tail, per_case_large = [], []
    cases = set.intersection(*[set(dsc[R1][o]) & set(dsc[Rmax][o]) for o in organs]) if organs else set()
    for c in cases:
        td = [dsc[R1][o][c] - dsc[Rmax][o][c] for o in L.TAIL if c in dsc[R1][o] and c in dsc[Rmax][o]]
        ld = [dsc[R1][o][c] - dsc[Rmax][o][c] for o in organs if o not in L.TAIL and c in dsc[R1][o] and c in dsc[Rmax][o]]
        if td and ld:
            per_case_tail.append(np.mean(td)); per_case_large.append(np.mean(ld))
    pt, pl = np.array(per_case_tail), np.array(per_case_large)
    try:
        p_tl = wilcoxon(pt, pl).pvalue
    except Exception:
        p_tl = np.nan
    print(f"\nTAIL vs LARGE drop (per-case paired, n={len(pt)}): "
          f"tail {pt.mean():+.3f} vs large {pl.mean():+.3f}, Δ={pt.mean()-pl.mean():+.3f}, "
          f"Wilcoxon p={p_tl:.2e}, Cohen's d={cohen_d_paired(pt-pl):.2f}")

    json.dump(dict(rows=rows, tail_vs_large=dict(tail=float(pt.mean()), large=float(pl.mean()),
              p=float(p_tl), d=cohen_d_paired(pt - pl), n=len(pt))),
              open(f"{OUT_R}/m0_rigor.json", "w"), indent=2)
    print(f"\nwrote {OUT_R}/m0_rigor.json")


if __name__ == "__main__":
    main()
