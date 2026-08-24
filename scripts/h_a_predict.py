"""H-A: is per-organ fragility PREDICTABLE from anatomy (a priori, no acceleration experiment)?
Correlate the M0 fragility drop with anatomical descriptors computed from GT + clean image:
  surface-to-volume ratio (SA/V), volume, boundary contrast (mean |grad| on the organ surface), HF-energy frac.
  python h_a_predict.py --root <Dataset501_MRIfrag>  [--out_prefix m0 --labels_module labels]
"""
import os, glob, json, argparse, importlib
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_erosion, sobel
from scipy.stats import spearmanr, pearsonr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from paths import RESULTS as OUT_R, PLOTS as OUT_P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--labels_module", default="labels")
    ap.add_argument("--out_prefix", default="m0")
    args = ap.parse_args()
    L = importlib.import_module(args.labels_module)
    frag = json.load(open(f"{OUT_R}/{args.out_prefix}_fragility_dice.json"))
    drop = {o: frag[nm]["R1"] - frag[nm]["R8"] for o, nm in L.ABDO.items() if nm in frag}

    gts = sorted(glob.glob(f"{args.root}/labelsTs/*.nii.gz"))
    desc = {o: dict(sav=[], vol=[], contrast=[]) for o in L.ABDO}
    for gp in gts:
        c = os.path.basename(gp)[:-7]
        gi = nib.load(gp); g = np.asanyarray(gi.dataobj).astype(np.int16)
        sp = np.array(gi.header.get_zooms()[:3], float); vvol = float(np.prod(sp))
        ip = f"{args.root}/imagesTs_clean/{c}_0000.nii.gz"
        img = np.asanyarray(nib.load(ip).dataobj).astype(np.float32) if os.path.exists(ip) else None
        for o in L.ABDO:
            m = g == o
            n = int(m.sum())
            if n < 30:
                continue
            sl = tuple(slice(max(p.min()-2, 0), p.max()+3) for p in np.where(m))
            mc = m[sl]
            surf = mc ^ binary_erosion(mc)
            desc[o]["sav"].append(surf.sum() / n)                 # surface-to-volume proxy
            desc[o]["vol"].append(n * vvol / 1000.0)              # cm^3
            if img is not None:
                ic = img[sl]
                gm = np.sqrt(sum(sobel(ic, axis=a)**2 for a in range(3)))
                desc[o]["contrast"].append(float(gm[surf].mean()) if surf.any() else np.nan)

    organs = [o for o in L.ABDO if desc[o]["sav"] and o in drop]
    rows = []
    for o in organs:
        rows.append(dict(id=o, organ=L.ABDO[o], tail=o in L.TAIL, drop=drop[o],
                         sav=float(np.mean(desc[o]["sav"])), vol_cm3=float(np.mean(desc[o]["vol"])),
                         contrast=float(np.nanmean(desc[o]["contrast"])) if desc[o]["contrast"] else np.nan))
    y = np.array([r["drop"] for r in rows])
    print(f"\n=== H-A: predict fragility (R1->R8 drop) from anatomy, n={len(rows)} organs ===")
    print(f"{'organ':12} {'tail':4} {'drop':>7} {'SA/V':>7} {'vol_cm3':>8} {'contrast':>9}")
    for r in sorted(rows, key=lambda r: -r["drop"]):
        print(f"{r['organ']:12} {'*' if r['tail'] else ' ':4} {r['drop']:+7.3f} {r['sav']:7.3f} {r['vol_cm3']:8.1f} {r['contrast']:9.1f}")

    out = {"n_organs": len(rows), "predictors": {}}
    print("\n--- predictor vs fragility-drop correlation (across organs) ---")
    for key, label in [("sav", "surface/volume"), ("vol_cm3", "volume"), ("contrast", "boundary contrast")]:
        x = np.array([r[key] for r in rows])
        ok = ~np.isnan(x)
        sr, sp = spearmanr(x[ok], y[ok]); pr, pp = pearsonr(np.log(x[ok]+1e-6) if key == "vol_cm3" else x[ok], y[ok])
        out["predictors"][key] = dict(spearman_r=float(sr), spearman_p=float(sp), pearson_r=float(pr), pearson_p=float(pp))
        print(f"  {label:18} Spearman r={sr:+.2f} (p={sp:.3f})   Pearson r={pr:+.2f} (p={pp:.3f})"
              + ("  [log-vol]" if key == "vol_cm3" else ""))
    json.dump(dict(rows=rows, **out), open(f"{OUT_R}/{args.out_prefix}_h_a.json", "w"), indent=2)

    # scatter: SA/V vs drop
    fig, ax = plt.subplots(figsize=(7, 5.2))
    for r in rows:
        ax.scatter(r["sav"], r["drop"], s=90, color="#d93025" if r["tail"] else "#2c5fb0", zorder=3)
        ax.annotate(r["organ"], (r["sav"], r["drop"]), fontsize=8, xytext=(4, 3), textcoords="offset points")
    sr = out["predictors"]["sav"]["spearman_r"]
    ax.set_xlabel("surface-to-volume ratio (a priori, from anatomy)"); ax.set_ylabel("fragility: Dice drop R1→R8")
    ax.set_title(f"H-A: fragility is predicted by surface-to-volume ratio\nSpearman r = {sr:+.2f} "
                 f"(red = small/tail organs)", fontweight="bold")
    ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(f"{OUT_P}/{args.out_prefix}_h_a.png", dpi=140)
    print(f"\nwrote {OUT_R}/{args.out_prefix}_h_a.json , {OUT_P}/{args.out_prefix}_h_a.png")


if __name__ == "__main__":
    main()
