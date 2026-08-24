"""SKM-TEA mechanism across ALL 44 extracted cases -> the Parseval energy->error law WITH ERROR BARS (the W1 fix; was n=1).
Reuses v5's exact leakage-corrected, echo-invariant computation: per structure x R, in-region |energy removed|
(predictor, leakage-corrected) vs |recon error| (effect), both qDESS echoes pooled, log-log Pearson r per case.
Aggregates per-case r across cases (mean +/- std) + a pooled scatter. -> skmtea_law_multicase.{png,json}"""
import os, glob, json, numpy as np, h5py, nibabel as nib
from scipy.stats import pearsonr, spearmanr
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"]})

RS = [2, 4, 6, 8]; PLOTS = "outputs/plots"; RES = "outputs/results"
LAB = {1: "patellar", 2: "femoral", 3: "tibial-med", 4: "tibial-lat", 5: "menisc-med", 6: "menisc-lat"}


def vd(W, R, acs=0.08, rng=None):
    rng = rng or np.random
    m = np.zeros(W, bool); c = W // 2; na = max(1, int(acs * W)); m[c - na // 2:c + na // 2 + 1] = True
    fr = np.abs(np.arange(W) - c); p = 1.0 / (fr + 1); p[m] = 0; p /= p.sum()
    m[rng.choice(W, W // R - m.sum(), False, p=p)] = True
    return m


MASKS = {R: vd(512, R, rng=np.random.RandomState(0)) for R in RS}   # fixed VD masks, identical across all cases


def case_points(h5path, segpath):
    """v5 mechanism for one case: pooled (structure x R) points over BOTH echoes."""
    seg = np.asanyarray(nib.load(segpath).dataobj).astype(np.int16)
    Xs, Ys = [], []
    with h5py.File(h5path, "r") as f:
        TGT = f["target"]; Z = TGT.shape[2]
        zs = [z for z in range(Z) if (seg[:, :, z] > 0).sum() > 200]
        for e in (0, 1):
            AER = {i: {R: [] for R in RS} for i in LAB}; ERR = {i: {R: [] for R in RS} for i in LAB}
            for z in zs:
                img = TGT[:, :, z, e, 0]; F = np.fft.fftshift(np.fft.fft2(img)); lz = seg[:, :, z]; cln = np.abs(img)
                rec = {R: np.abs(np.fft.ifft2(np.fft.ifftshift(np.where(MASKS[R][None, :], F, 0)))) for R in RS}
                for i in LAB:
                    M = lz == i
                    if M.sum() < 80:
                        continue
                    Fo = np.fft.fftshift(np.fft.fft2(img * M))                 # structure's own centered k-space
                    for R in RS:
                        For = Fo.copy(); For[:, MASKS[R]] = 0                  # keep ONLY removed lines
                        ef = np.abs(np.fft.ifft2(np.fft.ifftshift(For)))       # in-region own error field
                        AER[i][R].append(np.sqrt((ef[M] ** 2).sum()))         # leakage-corrected predictor
                        ERR[i][R].append(np.sqrt(((rec[R][M] - cln[M]) ** 2).sum()))
            for i in LAB:
                for R in RS:
                    if AER[i][R]:
                        Xs.append(np.mean(AER[i][R])); Ys.append(np.mean(ERR[i][R]))
    return np.array(Xs), np.array(Ys)


h5s = sorted(glob.glob("data/skmtea/kspace/**/*.h5", recursive=True))
print(f"found {len(h5s)} cases")
percase = {}; allX, allY = [], []
for h5path in h5s:
    cid = os.path.basename(h5path)[:-3]
    segp = f"data/skmtea/seg/{cid}_raw-data-track.nii.gz"
    if not os.path.exists(segp):
        print(f"  {cid}: no raw-data-track seg, skip"); continue
    try:
        X, Y = case_points(h5path, segp)
        if len(X) < 8:
            print(f"  {cid}: only {len(X)} pts, skip"); continue
        r = float(pearsonr(np.log(X), np.log(Y))[0])
        percase[cid] = r; allX.append(X); allY.append(Y)
        print(f"  {cid}: r={r:.3f}  ({len(X)} pts)", flush=True)
    except Exception as ex:
        print(f"  {cid}: FAIL {str(ex)[:60]}")

rs = np.array(list(percase.values()))
Xa = np.concatenate(allX); Ya = np.concatenate(allY)
pooled_r = float(pearsonr(np.log(Xa), np.log(Ya))[0]); pooled_sp = float(spearmanr(Xa, Ya).correlation)
out = {"n_cases": len(percase), "percase_r_mean": float(rs.mean()), "percase_r_std": float(rs.std()),
       "percase_r_min": float(rs.min()), "percase_r_max": float(rs.max()),
       "pooled_logr": pooled_r, "pooled_spearman": pooled_sp, "n_points": int(len(Xa)), "percase_r": percase}
os.makedirs(RES, exist_ok=True); json.dump(out, open(f"{RES}/skmtea_law_multicase.json", "w"), indent=2)
print("\n" + json.dumps({k: v for k, v in out.items() if k != "percase_r"}, indent=2))

# figure: per-case r distribution + pooled law
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
ax[0].hist(rs, bins=15, color="#1a9850", alpha=.85, edgecolor="white")
ax[0].axvline(rs.mean(), color="k", ls="--", lw=1.6, label=f"mean r = {rs.mean():.2f} ± {rs.std():.2f}")
ax[0].set_xlabel("per-case Pearson r (log-log)"); ax[0].set_ylabel("# cases"); ax[0].legend(frameon=False)
ax[0].set_title(f"Mechanism holds across {len(percase)} real cases", fontweight="bold")
for X, Y in zip(allX, allY):
    ax[1].scatter(X, Y, s=7, alpha=.22, color="#1a9850")
b, m = np.polynomial.polynomial.polyfit(np.log(Xa), np.log(Ya), 1)
xx = np.linspace(np.log(Xa).min(), np.log(Xa).max(), 50)
ax[1].plot(np.exp(xx), np.exp(b + m * xx), "k--", lw=1.6, label=f"pooled r = {pooled_r:.2f}")
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].legend(frameon=False)
ax[1].set_xlabel("|energy removed|  (in-region, leakage-corrected, log)")
ax[1].set_ylabel("|reconstruction error|  (log)")
ax[1].set_title(f"Pooled: {len(Xa)} (structure×R) points", fontweight="bold")
fig.suptitle(f"SKM-TEA real qDESS k-space — Parseval energy→error mechanism WITH ERROR BARS (n={len(percase)} cases)",
             fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig(f"{PLOTS}/skmtea_law_multicase.png", dpi=140, bbox_inches="tight"); plt.close(fig)
print("wrote skmtea_law_multicase.png")
