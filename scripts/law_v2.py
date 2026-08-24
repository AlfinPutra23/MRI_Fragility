"""LAW v2: can a physically-grounded predictor beat SA/V and absorb the hollow-organ outliers (colon/small_bowel)?

Motivated by (a) k-space undersampling removes HIGH-FREQUENCY energy first [MRI recon lit], and
(b) the spectral bias / frequency principle: nets learn low freq first, high freq worst [Xu 2019] -> an organ's
fragility should track HOW MUCH OF ITS SIGNAL LIVES IN HIGH SPATIAL FREQUENCIES, not just its surface/volume.

New predictors (all a priori from GT + clean image, no acceleration experiment):
  hf8_img  : fraction of the organ's image k-space energy beyond the R=8 radial cutoff (rho>1/8)  [the physical one]
  centroid : spectral centroid (mean radial freq, energy-weighted) of the organ's image patch     [scale-free HF]
  fdim     : box-counting fractal dimension of the organ boundary (tortuosity)                     [morphology]
vs baselines sav (surface/volume) and contrast. Compares Spearman r; 2-feature leave-one-out CV; new scatter.
  python law_v2.py --root <Dataset501_MRIfrag> --out_prefix m0 [--max_cases 40]
"""
import os, glob, json, argparse, importlib
import numpy as np, nibabel as nib
from scipy.ndimage import binary_erosion, sobel
from scipy.stats import spearmanr
from paths import RESULTS as OUT_R, PLOTS as OUT_P
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt


def radial_spec(patch):
    """2D power spectrum -> (radial energy profile summary): returns (hf_frac_at_R8, spectral_centroid)."""
    F = np.fft.fftshift(np.fft.fft2(patch)); P = np.abs(F) ** 2
    n = patch.shape[0]; c = n // 2
    y, x = np.mgrid[:n, :n]; rho = np.sqrt((y - c) ** 2 + (x - c) ** 2) / c      # 0..~1 (Nyquist=1)
    P[c, c] = 0.0                                                                # drop DC (mean) -> structure only
    tot = P.sum() + 1e-12
    hf8 = P[rho > 1 / 8].sum() / tot                                            # energy an R8 radial cut would lose
    centroid = (rho * P).sum() / tot                                           # mean radial frequency
    return float(hf8), float(centroid)


def box_fd(mask2d):
    """box-counting fractal dimension of the boundary of a 2D binary mask."""
    b = mask2d ^ binary_erosion(mask2d)
    if b.sum() < 8: return np.nan
    ys, xs = np.where(b); h = ys.max() - ys.min() + 1; w = xs.max() - xs.min() + 1
    sizes = [s for s in (2, 3, 4, 6, 8, 12, 16) if s < min(h, w)]
    if len(sizes) < 3: return np.nan
    yy, xx = ys - ys.min(), xs - xs.min(); counts = []
    for s in sizes:
        seen = set();
        for a, o in zip(yy // s, xx // s): seen.add((a, o))
        counts.append(len(seen))
    coef = np.polyfit(np.log(sizes), np.log(counts), 1)
    return float(-coef[0])                                                     # slope of log-count vs log(1/size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--out_prefix", default="m0")
    ap.add_argument("--labels_module", default="labels"); ap.add_argument("--max_cases", type=int, default=40)
    ap.add_argument("--patch", type=int, default=96)
    args = ap.parse_args()
    L = importlib.import_module(args.labels_module)
    frag = json.load(open(f"{OUT_R}/{args.out_prefix}_fragility_dice.json"))
    drop = {o: frag[nm]["R1"] - frag[nm]["R8"] for o, nm in L.ABDO.items() if nm in frag}

    gts = sorted(glob.glob(f"{args.root}/labelsTs/*.nii.gz"))
    step = max(len(gts) // args.max_cases, 1); gts = gts[::step]
    feats = {o: dict(sav=[], contrast=[], hf8_img=[], centroid=[], fdim=[]) for o in L.ABDO}
    print(f"law_v2: scanning {len(gts)} cases (patch {args.patch})...")
    for gp in gts:
        c = os.path.basename(gp)[:-7]; g = np.asanyarray(nib.load(gp).dataobj).astype(np.int16)
        ip = f"{args.root}/imagesTs_clean/{c}_0000.nii.gz"
        img = np.asanyarray(nib.load(ip).dataobj).astype(np.float32) if os.path.exists(ip) else None
        for o in L.ABDO:
            m = g == o
            if m.sum() < 40: continue
            zc = np.array([m[:, :, z].sum() for z in range(m.shape[2])]); z = int(zc.argmax())     # max-area slice
            m2 = m[:, :, z]
            if m2.sum() < 20: continue
            surf = m2 ^ binary_erosion(m2); feats[o]["sav"].append(surf.sum() / m2.sum())
            feats[o]["fdim"].append(box_fd(m2))
            # crop/pad organ bbox -> fixed square patch (preserves absolute spatial frequency)
            ys, xs = np.where(m2); cy, cx = (ys.min()+ys.max())//2, (xs.min()+xs.max())//2; h = args.patch//2
            sl = (slice(max(cy-h,0), cy+h), slice(max(cx-h,0), cx+h))
            if img is not None:
                ic = img[:, :, z][sl] * m2[sl]                                   # masked organ image content
                if ic.shape[0] > 8 and ic.shape[1] > 8:
                    pp = np.zeros((args.patch, args.patch), np.float32); pp[:ic.shape[0], :ic.shape[1]] = ic
                    hf8, cen = radial_spec(pp); feats[o]["hf8_img"].append(hf8); feats[o]["centroid"].append(cen)
                    gm = np.sqrt(sum(sobel(img[:, :, z][sl], axis=a)**2 for a in range(2)))
                    feats[o]["contrast"].append(float(gm[surf[sl]].mean()) if surf[sl].any() else np.nan)

    organs = [o for o in L.ABDO if feats[o]["sav"] and o in drop]
    rows = []
    for o in organs:
        r = dict(id=o, organ=L.ABDO[o], tail=o in L.TAIL, drop=drop[o])
        for k in ["sav", "contrast", "hf8_img", "centroid", "fdim"]:
            r[k] = float(np.nanmean(feats[o][k])) if feats[o][k] else np.nan
        rows.append(r)
    y = np.array([r["drop"] for r in rows])

    print(f"\n=== LAW v2: predictors vs fragility drop, n={len(rows)} organs ===")
    print(f"{'organ':12}{'tail':5}{'drop':>7}{'sav':>7}{'hf8_img':>9}{'centroid':>9}{'fdim':>7}{'contrast':>9}")
    for r in sorted(rows, key=lambda r: -r["drop"]):
        print(f"{r['organ']:12}{'*' if r['tail'] else '':5}{r['drop']:+7.3f}{r['sav']:7.3f}{r['hf8_img']:9.3f}{r['centroid']:9.3f}{r['fdim']:7.2f}{r['contrast']:9.1f}")

    print("\n--- single-feature Spearman r vs drop ---")
    res = {}
    for k in ["sav", "hf8_img", "centroid", "fdim", "contrast"]:
        x = np.array([r[k] for r in rows]); ok = ~np.isnan(x)
        sr, sp = spearmanr(x[ok], y[ok]); res[k] = dict(r=float(sr), p=float(sp))
        print(f"  {k:10} r={sr:+.3f}  (p={sp:.3f})  n={ok.sum()}")

    # 2-feature leave-one-out CV: sav + each new feature (guard against overfitting at small n)
    def loo_r2(cols):
        X = np.column_stack([[r[k] for r in rows] for k in cols]); ok = ~np.isnan(X).any(1)
        X, yy = X[ok], y[ok]; X = (X - X.mean(0)) / (X.std(0) + 1e-9); preds = np.zeros(len(yy))
        for i in range(len(yy)):
            tr = np.ones(len(yy), bool); tr[i] = False
            A = np.column_stack([X[tr], np.ones(tr.sum())]); b = np.linalg.lstsq(A, yy[tr], rcond=None)[0]
            preds[i] = np.append(X[i], 1) @ b
        ss = 1 - ((yy - preds)**2).sum() / (((yy - yy.mean())**2).sum() + 1e-12); return float(ss)
    print("\n--- leave-one-out CV R^2 (predictive, not in-sample) ---")
    for cols in [["sav"], ["hf8_img"], ["centroid"], ["sav", "hf8_img"], ["sav", "centroid"], ["sav", "fdim"], ["sav", "contrast"]]:
        print(f"  {'+'.join(cols):22} LOO R^2 = {loo_r2(cols):+.3f}")

    json.dump(dict(rows=rows, single=res), open(f"{OUT_R}/{args.out_prefix}_law_v2.json", "w"), indent=2)

    # scatter of the best new single predictor (highest |r|)
    best = max([k for k in ["hf8_img", "centroid", "fdim"]], key=lambda k: abs(res[k]["r"]))
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.3))
    for j, (k, ttl) in enumerate([("sav", "SA/V (old law)"), (best, f"{best} (physics-grounded)")]):
        for r in rows:
            ax[j].scatter(r[k], r["drop"], s=95, color="#d93025" if r["tail"] else "#2c5fb0", zorder=3)
            lbl = r["organ"] if r["organ"] in ("colon", "small_bowel", "esophagus", "adrenal_L", "liver", "pancreas") else ""
            if lbl: ax[j].annotate(lbl, (r[k], r["drop"]), fontsize=8, xytext=(4, 3), textcoords="offset points")
        ax[j].set_xlabel(k); ax[j].set_ylabel("fragility (R1->R8 drop)")
        ax[j].set_title(f"{ttl}\nSpearman r = {res[k]['r']:+.2f}", fontweight="bold"); ax[j].grid(alpha=.3)
    fig.suptitle("Law v2: does high-frequency energy predict fragility better than SA/V?", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .95]); fig.savefig(f"{OUT_P}/{args.out_prefix}_law_v2.png", dpi=145)
    print(f"\nwrote {OUT_R}/{args.out_prefix}_law_v2.json , {OUT_P}/{args.out_prefix}_law_v2.png  (best new = {best})")


if __name__ == "__main__":
    main()
