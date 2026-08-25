"""Two CPU audit analyses in one pass over the abdominal test cases (no GPU):
 (1) CASE-CLUSTERED bootstrap CI on the centroid->drop law — resample the 240 CASES (not organs), recompute the
     13-organ law each time -> honest CI that accounts for case-level variation (addresses 'CIs ignore case structure').
 (2) METRIC BLINDNESS is not SSIM-specific — per case at R8, does PSNR (and SSIM) predict per-organ tail Dice? If the
     within-R correlation is ~0 for BOTH, blindness is metric-general.
  python cpu_audit_extras.py [--smoke N]
-> outputs/results/cpu_audit_extras.json"""
import os, glob, json, argparse, numpy as np, nibabel as nib
from scipy.stats import spearmanr, pearsonr
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import sys; sys.path.insert(0, "scripts"); import labels as L
np.random.seed(0)
ROOT = "nnUNet_raw/Dataset501_MRIfrag"


def dice(a, b):
    s = a.sum() + b.sum(); return 2.0 * np.logical_and(a, b).sum() / s if s else np.nan


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", type=int, default=0); ap.add_argument("--boot", type=int, default=5000)
    args = ap.parse_args()
    law = {r["id"]: r["centroid"] for r in json.load(open("outputs/results/m0_law_v2.json"))["rows"]}
    gts = sorted(glob.glob(f"{ROOT}/labelsTs/*.nii.gz"))
    if args.smoke: gts = gts[:args.smoke]
    ids = list(L.ABDO)
    # per-case: per-organ Dice at clean & R8; case PSNR/SSIM of R8 image vs clean image; case tail Dice
    percase = []
    for gp in gts:
        case = os.path.basename(gp)[:-7]
        pc, p8 = f"{ROOT}/predsTs_clean/{case}.nii.gz", f"{ROOT}/predsTs_R8/{case}.nii.gz"
        ic, i8 = f"{ROOT}/imagesTs_clean/{case}_0000.nii.gz", f"{ROOT}/imagesTs_R8/{case}_0000.nii.gz"
        if not all(os.path.exists(x) for x in [pc, p8, ic, i8]): continue
        gt = np.asanyarray(nib.load(gp).dataobj).astype(np.int16)
        prc = np.asanyarray(nib.load(pc).dataobj).astype(np.int16); pr8 = np.asanyarray(nib.load(p8).dataobj).astype(np.int16)
        drop = {}
        for o in ids:
            if (gt == o).sum() >= 30:
                drop[o] = dice(prc == o, gt == o) - dice(pr8 == o, gt == o)
        imc = np.asanyarray(nib.load(ic).dataobj).astype(np.float32); im8 = np.asanyarray(nib.load(i8).dataobj).astype(np.float32)
        rng = float(imc.max() - imc.min()) + 1e-6
        cp = psnr(imc, im8, data_range=rng); cs = ssim(imc, im8, data_range=rng)
        tail = np.nanmean([dice(pr8 == o, gt == o) for o in L.TAIL if (gt == o).sum() >= 30])
        percase.append({"case": case, "drop": drop, "psnr": float(cp), "ssim": float(cs), "tail_dice": float(tail)})
    n = len(percase); print(f"  {n} cases scored", flush=True)

    # (1) case-clustered bootstrap of the centroid->drop law
    def law_rho(sample):
        md = {o: np.nanmean([c["drop"][o] for c in sample if o in c["drop"]]) for o in ids}
        oo = [o for o in ids if o in law and not np.isnan(md[o])]
        return spearmanr([law[o] for o in oo], [md[o] for o in oo]).correlation
    obs = law_rho(percase)
    boot = []
    for _ in range(args.boot if not args.smoke else 200):
        s = [percase[i] for i in np.random.randint(0, n, n)]
        r = law_rho(s)
        if r == r: boot.append(r)
    boot = np.array(boot)
    ci = [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)]

    # (1b) PATIENT-clustered bootstrap: the 240 series are 60 patients x 4 phases, so series are
    # NOT independent. Resample PATIENTS (all four of a patient's phases move together).
    import re as _re
    pats = {}
    for i, c in enumerate(percase):
        m = _re.match(r'test_(\d+)_', c["case"])
        pats.setdefault(m.group(1) if m else str(i), []).append(i)
    pkeys = list(pats)
    bootp = []
    for _ in range(args.boot if not args.smoke else 200):
        idx = [i for k in np.random.choice(pkeys, len(pkeys), replace=True) for i in pats[k]]
        r = law_rho([percase[i] for i in idx])
        if r == r: bootp.append(r)
    bootp = np.array(bootp)
    cip = [round(float(np.percentile(bootp, 2.5)), 3), round(float(np.percentile(bootp, 97.5)), 3)]
    print(f"  patient-clustered ({len(pkeys)} patients): 95% CI = {cip}")

    # (2) PSNR/SSIM within-R (R8) blindness: case metric vs case tail Dice
    P = np.array([c["psnr"] for c in percase]); S = np.array([c["ssim"] for c in percase]); T = np.array([c["tail_dice"] for c in percase])
    out = {"n_cases": n,
           "law_caseclustered": {"obs_spearman": round(float(obs), 3), "boot95_CI": ci, "boot_mean": round(float(boot.mean()), 3)},
           "metric_blindness_R8": {"psnr_std_across_cases": round(float(P.std()), 3), "ssim_std": round(float(S.std()), 3),
                                   "psnr_vs_tailDice_spearman": round(float(spearmanr(P, T).correlation), 3),
                                   "ssim_vs_tailDice_spearman": round(float(spearmanr(S, T).correlation), 3)},
           "law_patientclustered": {"n_patients": len(pkeys), "obs_spearman": round(float(obs), 3),
                                    "boot95_CI": cip, "boot_mean": round(float(bootp.mean()), 3)},
           "percase_R8": {"case": [c["case"] for c in percase],
                          "psnr": [round(float(x), 3) for x in P], "ssim": [round(float(x), 4) for x in S],
                          "tail_dice": [round(float(x), 4) for x in T]},
           "percase_drop": [{"case": c["case"], "drop": {str(k): round(float(v), 4) for k, v in c["drop"].items()}} for c in percase]}
    json.dump(out, open("outputs/results/cpu_audit_extras.json", "w"), indent=2)
    print("\n=== (1) case-clustered law CI ===")
    print(f"  centroid->drop Spearman = {obs:.3f}, case-clustered 95% CI = {ci}")
    print("=== (2) metric blindness at R8 (within fixed acceleration, across cases) ===")
    b = out["metric_blindness_R8"]
    print(f"  PSNR->tailDice rho = {b['psnr_vs_tailDice_spearman']} | SSIM->tailDice rho = {b['ssim_vs_tailDice_spearman']}  (both ~0 = metric-general blindness)")
    print("wrote cpu_audit_extras.json")


if __name__ == "__main__":
    main()
