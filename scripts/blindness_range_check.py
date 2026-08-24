"""GATE CHECK for the metric-blindness result (the proposed new paper spine).

The claim: SSIM correlates with Dice at rho=0.54 POOLED across R but only 0.037 WITHIN a fixed R -> "at a given
acceleration, image quality tells you nothing about task success."

THE THREAT: RESTRICTED RANGE. If SSIM barely varies across cases within a fixed R, a near-zero within-R correlation is
a statistical artifact (you cannot correlate a constant), not a finding. This script decides that:
  - per-R spread of case-level SSIM (std, IQR, min-max, coefficient of variation)
  - per-R spread of case-level Dice, for a fragile and a robust organ (is there outcome variance to explain?)
  - the within-R Spearman alongside those spreads
  - a DISATTENUATED estimate: correlation corrected for range restriction (Thorndike case II), i.e. what the
    within-R correlation WOULD be if SSIM varied as much within R as it does overall.

VERDICT logic:
  REAL       SSIM varies materially within R (CV above threshold) AND within-R rho stays ~0 even disattenuated
             -> image quality genuinely carries no case-level task information. Finding stands.
  ARTIFACT   SSIM is near-constant within R -> the null correlation is range restriction. Finding must be dropped
             or reframed (e.g. report the pooled/within decomposition only as a caution, not as evidence).

  python scripts/blindness_range_check.py --root nnUNet_results/Dataset501_MRIfrag --R 2 4 6 8
-> outputs/results/blindness_range_check.json
"""
import os, glob, json, argparse
import numpy as np
import nibabel as nib
from skimage.metrics import structural_similarity as ssim
from scipy.stats import spearmanr, pearsonr
import labels as L


def dice(a, b):
    s = a.sum() + b.sum()
    return 2.0 * np.logical_and(a, b).sum() / s if s else np.nan


def case_ssim(clean, rec, n_central=30):
    z0 = max(0, clean.shape[2] // 2 - n_central // 2)
    vals = []
    for z in range(z0, min(clean.shape[2], z0 + n_central)):
        ref = clean[:, :, z]
        if ref.max() <= 0: continue
        vals.append(ssim(ref, rec[:, :, z], data_range=float(ref.max())))
    return float(np.mean(vals)) if vals else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="nnUNet_results/Dataset501_MRIfrag")
    ap.add_argument("--preds_tpl", default="predsTs_{tag}")
    ap.add_argument("--R", type=float, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--max_cases", type=int, default=60)
    ap.add_argument("--out", default="outputs/results/blindness_range_check.json")
    args = ap.parse_args()

    FRAGILE, ROBUST = 4, 5                                  # adrenal-ish vs liver (ids from labels.py ABDO)
    frag_id = next((o for o in L.ABDO if o in L.TAIL), FRAGILE)
    rob_id = next((o for o in L.ABDO if o not in L.TAIL), ROBUST)
    gts = sorted(glob.glob(f"{args.root}/labelsTs/*.nii.gz"))[:args.max_cases]
    print(f"{len(gts)} test cases, R={args.R}\n  fragile probe = {L.ABDO.get(frag_id)}   robust probe = {L.ABDO.get(rob_id)}")

    perR = {}
    all_ssim = []
    for R in args.R:
        tag = f"R{int(R)}"
        pdir = f"{args.root}/{args.preds_tpl.format(tag=tag)}"
        idir, cdir = f"{args.root}/imagesTs_{tag}", f"{args.root}/imagesTs_clean"
        S, Df, Dr = [], [], []
        for gp in gts:
            case = os.path.basename(gp)[:-7]
            pp, cp, rp = f"{pdir}/{case}.nii.gz", f"{cdir}/{case}_0000.nii.gz", f"{idir}/{case}_0000.nii.gz"
            if not (os.path.exists(pp) and os.path.exists(cp) and os.path.exists(rp)): continue
            clean = np.asanyarray(nib.load(cp).dataobj).astype(np.float32)
            rec = np.asanyarray(nib.load(rp).dataobj).astype(np.float32)
            gt = np.asanyarray(nib.load(gp).dataobj).astype(np.int16)
            pr = np.asanyarray(nib.load(pp).dataobj).astype(np.int16)
            s = case_ssim(clean, rec)
            if not np.isfinite(s): continue
            S.append(s)
            Df.append(dice(pr == frag_id, gt == frag_id) if (gt == frag_id).sum() >= 30 else np.nan)
            Dr.append(dice(pr == rob_id, gt == rob_id) if (gt == rob_id).sum() >= 30 else np.nan)
        S, Df, Dr = np.array(S), np.array(Df), np.array(Dr)
        all_ssim.extend(S.tolist())
        def stats_of(v):
            v = v[np.isfinite(v)]
            if len(v) < 3: return None
            q1, q3 = np.percentile(v, [25, 75])
            return {"n": int(len(v)), "mean": round(float(v.mean()), 4), "std": round(float(v.std(ddof=1)), 4),
                    "min": round(float(v.min()), 4), "max": round(float(v.max()), 4),
                    "IQR": round(float(q3 - q1), 4), "CV_pct": round(float(100 * v.std(ddof=1) / (abs(v.mean()) + 1e-9)), 2)}
        rec_ = {"ssim": stats_of(S), "dice_fragile": stats_of(Df), "dice_robust": stats_of(Dr)}
        for nm, D in [("fragile", Df), ("robust", Dr)]:
            m = np.isfinite(S) & np.isfinite(D)
            if m.sum() >= 4:
                rec_[f"rho_within_{nm}"] = round(float(spearmanr(S[m], D[m]).correlation), 3)
                rec_[f"pearson_within_{nm}"] = round(float(pearsonr(S[m], D[m])[0]), 3)
        perR[str(int(R))] = rec_
        s_ = rec_["ssim"]
        print(f"  R={int(R)}: SSIM {s_['mean']:.3f} +-{s_['std']:.4f} [{s_['min']:.3f},{s_['max']:.3f}] CV={s_['CV_pct']}%  "
              f"| rho_within fragile={rec_.get('rho_within_fragile')} robust={rec_.get('rho_within_robust')}")

    # ---- disattenuation: what would within-R rho be if SSIM varied as much as it does POOLED? ----
    sd_pooled = float(np.std(all_ssim, ddof=1))
    dis = {}
    for R, rec_ in perR.items():
        sd_within = rec_["ssim"]["std"]
        for nm in ["fragile", "robust"]:
            r = rec_.get(f"pearson_within_{nm}")
            if r is None: continue
            ratio = sd_pooled / (sd_within + 1e-12)                     # Thorndike case II
            denom = np.sqrt(1 + r**2 * (ratio**2 - 1))
            dis[f"R{R}_{nm}"] = round(float(r * ratio / denom), 3)

    ssim_cv = float(np.mean([perR[R]["ssim"]["CV_pct"] for R in perR]))
    mean_within = float(np.nanmean([perR[R].get("rho_within_fragile", np.nan) for R in perR]))
    dis_vals = [v for v in dis.values() if np.isfinite(v)]
    mean_dis = float(np.mean(np.abs(dis_vals))) if dis_vals else float("nan")
    if ssim_cv < 1.0:
        verdict = ("ARTIFACT — SSIM is nearly constant within R (CV<1%); a null within-R correlation is range "
                   "restriction, not evidence. Do NOT build the paper spine on this.")
    elif abs(mean_dis) < 0.35:
        verdict = (f"REAL — SSIM varies materially within R (mean CV={ssim_cv:.1f}%) and the correlation stays weak "
                   f"even after correcting for range restriction (|disattenuated r|~{mean_dis:.2f}). "
                   f"Image quality genuinely carries little case-level task information.")
    else:
        verdict = (f"PARTIAL — SSIM varies (CV={ssim_cv:.1f}%) but disattenuated |r|~{mean_dis:.2f} is non-trivial; "
                   f"range restriction explains part of the null. Report the decomposition with this caveat.")

    out = {"per_R": perR, "ssim_sd_pooled": round(sd_pooled, 4), "disattenuated_within_r": dis,
           "mean_ssim_CV_pct": round(ssim_cv, 2), "mean_within_rho_fragile": round(mean_within, 3),
           "VERDICT": verdict}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n  pooled SSIM sd = {sd_pooled:.4f}")
    print(f"  disattenuated within-R r: {dis}")
    print(f"\n>>> {verdict}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
