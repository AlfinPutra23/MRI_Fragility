"""Validate the fragility-theory predictions on real data (CPU only). Computes, per organ, the discarded-energy scalar
Φ_R (fraction of the windowed-organ signal energy the R-fold VD mask removes), replicating kspace.py's exact operation,
then runs three falsifiable checks the theory-workflow proposed:
  T1  concavity : within an organ, across R, is ΔDice ∝ √Φ_R (concave) better than ∝ Φ_R (linear)?  [boundary-displacement δ∝√energy]
  T2  Φ beats c : does the *measured* removed-energy scalar Φ_R8 predict the drop at least as well as the centroid proxy (0.86)?
  T3  Occam gate: does the full Fragility Index FI=(SA/V)·√Φ/(contrast+C0) add over Φ_R ALONE (partial corr of SA/V, 1/contrast | Φ)?
  python theory_validate.py [--n 30]  -> outputs/results/theory_validate.json"""
import json, glob, os, argparse, numpy as np, nibabel as nib
from scipy.stats import spearmanr, pearsonr, rankdata
import sys; sys.path.insert(0, "scripts")
from kspace import vd_cartesian_mask
D = "nnUNet_raw/Dataset501_MRIfrag"; RS = [2, 4, 6, 8]


def partial_spearman(x, y, ctrl):
    xr, yr, cr = rankdata(x), rankdata(y), rankdata(ctrl)
    A = np.column_stack([np.ones_like(cr), cr])
    rx = xr - A @ np.linalg.lstsq(A, xr, rcond=None)[0]
    ry = yr - A @ np.linalg.lstsq(A, yr, rcond=None)[0]
    return float(spearmanr(rx, ry).correlation)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=30); a = ap.parse_args()
    law = {r["id"]: r for r in json.load(open("outputs/results/m0_law_v2.json"))["rows"]}
    dice = json.load(open("outputs/results/m0_fragility_dice.json"))
    ids = [i for i in law if law[i]["organ"] in dice]

    # ---- compute Φ_R(organ) = removed windowed-organ energy fraction, averaged over cases ----
    phi = {i: {R: [] for R in RS} for i in ids}
    gts = sorted(glob.glob(f"{D}/labelsTs/*.nii.gz"))[:a.n]
    for ci, gp in enumerate(gts):
        case = os.path.basename(gp)[:-7]; ip = f"{D}/imagesTs_clean/{case}_0000.nii.gz"
        if not os.path.exists(ip): continue
        img = np.asanyarray(nib.load(ip).dataobj).astype(np.float32)
        gt = np.asanyarray(nib.load(gp).dataobj).astype(np.int16)
        v = np.moveaxis(img, 2, 0); gv = np.moveaxis(gt, 2, 0); H = v.shape[1]     # slices-first, H = PE rows
        masks = {R: vd_cartesian_mask(H, R, seed=ci) for R in RS}                   # True = kept (matches pipeline)
        for i in ids:
            m = (gv == i)
            if m.sum() < 200: continue
            xo = v * m                                                             # windowed organ signal x_Ω
            K = np.fft.fftshift(np.fft.fft2(xo, axes=(-2, -1)), axes=(-2, -1))
            rowP = (np.abs(K) ** 2).sum(axis=(0, 2))                               # energy per PE row (H,)
            tot = rowP.sum()
            if tot <= 0: continue
            for R in RS: phi[i][R].append(float(rowP[~masks[R]].sum() / tot))
        if (ci + 1) % 10 == 0: print(f"  {ci+1}/{len(gts)} cases", flush=True)
    Phi = {i: {R: float(np.mean(phi[i][R])) if phi[i][R] else np.nan for R in RS} for i in ids}
    ids = [i for i in ids if all(not np.isnan(Phi[i][R]) for R in RS)]

    dd = lambda i, R: dice[law[i]["organ"]]["R1"] - dice[law[i]["organ"]][f"R{R}"]  # ΔDice(organ,R)

    # ---- T1: within-organ concavity (include R1 anchor: Φ=0, ΔDice=0) ----
    lin, sq = [], []
    for i in ids:
        x = np.array([0.0] + [Phi[i][R] for R in RS]); y = np.array([0.0] + [dd(i, R) for R in RS])
        lin.append(abs(pearsonr(x, y)[0])); sq.append(abs(pearsonr(np.sqrt(x), y)[0]))
    lin, sq = np.array(lin), np.array(sq)
    t1 = {"mean_pearson_linear": round(float(lin.mean()), 3), "mean_pearson_sqrt": round(float(sq.mean()), 3),
          "sqrt_wins_in_organs": int((sq > lin).sum()), "n_organs": len(ids),
          "verdict": "concave (√ better)" if sq.mean() > lin.mean() else "linear better"}

    # ---- T2: measured Φ_R8 vs centroid proxy, across organs ----
    cen = np.array([law[i]["centroid"] for i in ids]); phi8 = np.array([Phi[i][8] for i in ids])
    drop8 = np.array([dd(i, 8) for i in ids])
    t2 = {"n_organs": len(ids),
          "spearman_Phi8_vs_drop": round(float(spearmanr(phi8, drop8).correlation), 3),
          "spearman_centroid_vs_drop": round(float(spearmanr(cen, drop8).correlation), 3),
          "collinearity_Phi8_vs_centroid": round(float(spearmanr(phi8, cen).correlation), 3)}
    t2["verdict"] = ("Φ_R ≥ centroid" if t2["spearman_Phi8_vs_drop"] >= t2["spearman_centroid_vs_drop"]
                     else "centroid still better")

    # ---- T3: Occam gate — pooled (organ×R). Does SA/V or 1/contrast add OVER Φ_R? ----
    P, DR, SAV, INVC, FI = [], [], [], [], []
    C0 = float(np.median([law[i]["contrast"] for i in ids]))
    for i in ids:
        for R in RS:
            P.append(Phi[i][R]); DR.append(dd(i, R)); SAV.append(law[i]["sav"])
            INVC.append(1.0 / (law[i]["contrast"] + C0))
            FI.append(law[i]["sav"] * np.sqrt(Phi[i][R]) / (law[i]["contrast"] + C0))
    P, DR, SAV, INVC, FI = map(np.array, (P, DR, SAV, INVC, FI))
    t3 = {"n_points": len(P), "note": "points are organ-clustered (not independent); partials guard against double-counting",
          "spearman_Phi_alone": round(float(spearmanr(P, DR).correlation), 3),
          "spearman_FI": round(float(spearmanr(FI, DR).correlation), 3),
          "partial_SAV_given_Phi": round(partial_spearman(SAV, DR, P), 3),
          "partial_invContrast_given_Phi": round(partial_spearman(INVC, DR, P), 3)}
    t3["verdict"] = ("FI adds over Φ (multi-factor law holds)"
                     if (t3["spearman_FI"] > t3["spearman_Phi_alone"] + 0.02 and abs(t3["partial_SAV_given_Phi"]) > 0.2)
                     else "Occam: report scalar Φ energy-budget (FI adds little over Φ alone)")

    out = {"n_cases_for_Phi": len(gts), "n_organs": len(ids),
           "Phi_R_per_organ": {law[i]["organ"]: {f"R{R}": round(Phi[i][R], 4) for R in RS} for i in ids},
           "T1_concavity": t1, "T2_Phi_beats_centroid": t2, "T3_occam_gate": t3}
    json.dump(out, open("outputs/results/theory_validate.json", "w"), indent=2)
    print("\n=== T1 concavity (within-organ, across R) ===")
    print(f"  mean|Pearson| linear {t1['mean_pearson_linear']} vs √ {t1['mean_pearson_sqrt']} | √ wins {t1['sqrt_wins_in_organs']}/{t1['n_organs']} -> {t1['verdict']}")
    print("=== T2 Φ_R8 vs centroid (across organs) ===")
    print(f"  Φ_R8->drop {t2['spearman_Phi8_vs_drop']} | centroid->drop {t2['spearman_centroid_vs_drop']} | collinearity {t2['collinearity_Phi8_vs_centroid']} -> {t2['verdict']}")
    print("=== T3 Occam gate (pooled organ×R) ===")
    print(f"  Φ alone {t3['spearman_Phi_alone']} | FI {t3['spearman_FI']} | partial SA/V|Φ {t3['partial_SAV_given_Phi']} | partial 1/contrast|Φ {t3['partial_invContrast_given_Phi']}")
    print(f"  -> {t3['verdict']}")
    print("wrote theory_validate.json")


if __name__ == "__main__":
    main()
