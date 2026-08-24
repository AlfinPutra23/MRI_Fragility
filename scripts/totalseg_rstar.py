"""Whole-body R* (safe-acceleration budget) via a TotalSeg-MRI R-sweep {1,2,4,6,8}. Closes the budget-confound gap:
does centroid predict R* OVER baseline difficulty on the whole body (where the two dissociate), the way the W1 drop-law
does? GATED — do NOT launch until the user approves and a GPU is free. -> outputs/results/totalseg_rstar.json"""
import os, glob, json, argparse, tempfile, shutil
import numpy as np, nibabel as nib
from scipy.stats import spearmanr, rankdata
import sys; sys.path.insert(0, "scripts")
from totalseg_law import DATA, ABDO, dice, centroid_of, run_ts
from kspace import undersample_volume


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25); ap.add_argument("--R", type=int, nargs="+", default=[1, 2, 4, 6, 8])
    ap.add_argument("--tol", type=float, default=0.05); ap.add_argument("--device", default="gpu"); ap.add_argument("--fast", type=int, default=0)
    ap.add_argument("--min_n", type=int, default=3); ap.add_argument("--out", default="outputs/results/totalseg_rstar.json")
    args = ap.parse_args()
    subs = sorted(glob.glob(f"{DATA}/s*/"))[:args.n]
    tmp = tempfile.mkdtemp(prefix="tsrstar_")
    dacc = {R: {} for R in args.R}; cacc = {}
    for si, sd in enumerate(subs):
        sid = os.path.basename(sd.rstrip("/"))
        im = nib.load(f"{sd}/mri.nii.gz"); arr = np.asanyarray(im.dataobj).astype(np.float32)
        sax = int(np.argmax(im.header.get_zooms()[:3]))
        gts = {os.path.basename(f)[:-7]: (np.asanyarray(nib.load(f).dataobj) > 0) for f in glob.glob(f"{sd}/segmentations/*.nii.gz")}
        for k, gm in gts.items():
            if gm.sum() < 100: continue
            c = centroid_of(arr, gm, sax)
            if c == c: cacc.setdefault(k, []).append(c)
        for R in args.R:
            rec = undersample_volume(arr, R, seed=hash(sid) % 99991, slice_axis=sax)
            recp = f"{tmp}/{sid}_R{R}.nii.gz"; nib.save(nib.Nifti1Image(rec, im.affine, im.header), recp)
            od = f"{tmp}/{sid}_R{R}_seg"; run_ts(recp, od, args.device, fast=bool(args.fast))
            for k, gm in gts.items():
                pf = f"{od}/{k}.nii.gz"
                if not os.path.exists(pf) or gm.sum() < 100: continue
                d = dice(np.asanyarray(nib.load(pf).dataobj) > 0, gm)
                if d == d: dacc[R].setdefault(k, []).append(d)
            shutil.rmtree(od, ignore_errors=True); os.remove(recp)
        print(f"  {si+1}/{len(subs)} {sid} done", flush=True)
    Rs = sorted(args.R); rows = []
    for k in cacc:
        if any(len(dacc[R].get(k, [])) < args.min_n for R in Rs): continue
        dm = {R: float(np.mean(dacc[R][k])) for R in Rs}; d1 = dm[Rs[0]]
        rstar = Rs[0]
        for R in Rs:
            if d1 - dm[R] <= args.tol: rstar = R
        rows.append({"structure": k, "new_anatomy": k not in ABDO, "centroid": round(float(np.mean(cacc[k])), 4),
                     "rstar": rstar, "dice_R1": round(d1, 4), "n": len(dacc[Rs[-1]][k])})
    c = np.array([r["centroid"] for r in rows]); rst = np.array([r["rstar"] for r in rows]); d1 = np.array([r["dice_R1"] for r in rows])
    def z(v): return (v - v.mean()) / (v.std() + 1e-12)
    def partial(x, y, ctrl):
        xx, yy = rankdata(x), rankdata(y); A = np.column_stack([np.ones_like(xx), z(rankdata(ctrl))])
        return spearmanr(xx - A @ np.linalg.lstsq(A, xx, rcond=None)[0], yy - A @ np.linalg.lstsq(A, yy, rcond=None)[0]).correlation
    out = {"n_subjects": len(subs), "tol": args.tol, "n_structures": len(rows),
           "centroid_rstar_spearman": round(float(spearmanr(c, rst).correlation), 3) if len(rows) >= 4 else None,
           "partial_centroid_rstar_given_difficulty": round(float(partial(c, rst, d1)), 3) if len(rows) >= 4 else None,
           "rows": sorted(rows, key=lambda r: r["rstar"])}
    json.dump(out, open(args.out, "w"), indent=2); shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n=== WHOLE-BODY R* ({len(rows)} structures) ===")
    print(f"  centroid->R* = {out['centroid_rstar_spearman']} | partial|difficulty = {out['partial_centroid_rstar_given_difficulty']}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
