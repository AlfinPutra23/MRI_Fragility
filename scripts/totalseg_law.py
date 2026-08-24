"""TotalSegmentator-MRI generalization test for the spectral-centroid fragility law.
For a subset of subjects: undersample the MRI at R in {1,8}, run the pretrained `total_mr` model, measure per-structure
Dice(pred,GT), compute per-structure drop (R1->R8) and spectral centroid, then correlate across ~40-50 structures.
Tests whether centroid->fragility generalizes to whole-body anatomy (thorax/spine/pelvis/MSK) beyond abdomen.

  python totalseg_law.py --n 25 --R 1 8 --device gpu          (full run)
  python totalseg_law.py --smoke 1 --R 1 8 --device cpu       (smoke: 1 subject on CPU)
-> outputs/results/totalseg_law.json
"""
import os, glob, json, argparse, subprocess, tempfile, shutil
import numpy as np, nibabel as nib
from scipy.stats import spearmanr
import sys; sys.path.insert(0, "scripts")
from kspace import undersample_volume

TS = "/home/user/anaconda3/envs/totalseg/bin/TotalSegmentator"
DATA = "data/totalseg_mri/extracted"
ABDO = {"spleen", "kidney_right", "kidney_left", "gallbladder", "liver", "stomach", "pancreas",
        "adrenal_gland_right", "adrenal_gland_left", "esophagus", "small_bowel", "duodenum", "colon"}   # overlap w/ MRISeg


def dice(a, b):
    s = a.sum() + b.sum(); return 2 * np.logical_and(a, b).sum() / s if s else np.nan


def centroid_of(img, mask, slice_axis):
    """energy-weighted mean radial frequency of masked image content, averaged over slices containing the structure."""
    vals = []
    for z in range(img.shape[slice_axis]):
        m2 = np.take(mask, z, slice_axis)
        if m2.sum() < 30: continue
        x = np.take(img, z, slice_axis) * m2
        P = np.abs(np.fft.fftshift(np.fft.fft2(x))) ** 2
        H, W = x.shape; cy, cx = H // 2, W // 2
        yy, xx = np.mgrid[:H, :W]; rho = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
        tot = P.sum()
        if tot > 0: vals.append(float((rho * P).sum() / tot))
    return float(np.mean(vals)) if vals else np.nan


def run_ts(inp, outdir, device, fast=True):
    cmd = [TS, "-i", inp, "-o", outdir, "-ta", "total_mr", "-d", device, "-q"] + (["-f"] if fast else [])
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25); ap.add_argument("--R", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--device", default="gpu"); ap.add_argument("--smoke", type=int, default=0); ap.add_argument("--fast", type=int, default=1)
    ap.add_argument("--min_n", type=int, default=3, help="min subjects per structure to include (use 1 for smoke)")
    ap.add_argument("--out", default="outputs/results/totalseg_law.json")
    args = ap.parse_args()
    subs = sorted(glob.glob(f"{DATA}/s*/"))[:(args.smoke or args.n)]
    tmp = tempfile.mkdtemp(prefix="tslaw_")
    dice_acc = {R: {} for R in args.R}; cent_acc = {}
    for si, sd in enumerate(subs):
        sid = os.path.basename(sd.rstrip("/"))
        im = nib.load(f"{sd}/mri.nii.gz"); arr = np.asanyarray(im.dataobj).astype(np.float32)
        sax = int(np.argmax(im.header.get_zooms()[:3]))
        gts = {os.path.basename(f)[:-7]: (np.asanyarray(nib.load(f).dataobj) > 0) for f in glob.glob(f"{sd}/segmentations/*.nii.gz")}
        for k, gm in gts.items():
            if gm.sum() < 100: continue
            c = centroid_of(arr, gm, sax)
            if c == c: cent_acc.setdefault(k, []).append(c)
        for R in args.R:
            rec = undersample_volume(arr, R, seed=hash(sid) % 99991, slice_axis=sax)
            recp = f"{tmp}/{sid}_R{R}.nii.gz"; nib.save(nib.Nifti1Image(rec, im.affine, im.header), recp)
            od = f"{tmp}/{sid}_R{R}_seg"
            run_ts(recp, od, args.device, fast=bool(args.fast))
            for k, gm in gts.items():
                pf = f"{od}/{k}.nii.gz"
                if not os.path.exists(pf) or gm.sum() < 100: continue
                d = dice(np.asanyarray(nib.load(pf).dataobj) > 0, gm)
                if d == d: dice_acc[R].setdefault(k, []).append(d)
            shutil.rmtree(od, ignore_errors=True); os.remove(recp)
        print(f"  {si+1}/{len(subs)} {sid} done", flush=True)
    r0, r1 = args.R[0], args.R[-1]
    rows = []
    for k in sorted(set(cent_acc) & set(dice_acc[r0]) & set(dice_acc[r1])):
        if len(dice_acc[r0][k]) < args.min_n or len(dice_acc[r1][k]) < args.min_n: continue
        d_a, d_b, c = np.mean(dice_acc[r0][k]), np.mean(dice_acc[r1][k]), np.mean(cent_acc[k])
        rows.append({"structure": k, "new_anatomy": k not in ABDO, "centroid": round(float(c), 4),
                     f"dice_R{r0}": round(float(d_a), 4), f"dice_R{r1}": round(float(d_b), 4),
                     "drop": round(float(d_a - d_b), 4), "n": len(dice_acc[r1][k])})
    cen = [r["centroid"] for r in rows]; drp = [r["drop"] for r in rows]
    rho, p = (spearmanr(cen, drp) if len(rows) >= 4 else (float("nan"), float("nan")))
    new = [r for r in rows if r["new_anatomy"]]
    rho_new = (spearmanr([r["centroid"] for r in new], [r["drop"] for r in new])[0] if len(new) >= 4 else float("nan"))
    out = {"n_subjects": len(subs), "R": args.R, "n_structures": len(rows),
           "law_all_spearman": round(float(rho), 3), "law_all_p": (float(p) if p == p else None),
           "law_new_anatomy_spearman": round(float(rho_new), 3), "n_new_anatomy": len(new),
           "rows": sorted(rows, key=lambda r: -r["drop"])}
    json.dump(out, open(args.out, "w"), indent=2)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n=== TOTALSEG-MRI LAW (n={len(subs)} subj, {len(rows)} structures) ===")
    print(f"  centroid->drop Spearman = {rho:.3f} (p={p:.4g})  [ALL structures]")
    print(f"  NEW anatomy only (n={len(new)}): Spearman = {rho_new:.3f}")
    print(f"  most fragile: {[(r['structure'], r['drop']) for r in out['rows'][:5]]}")
    print("  VERDICT:", "LAW GENERALIZES" if rho > 0.4 else ("weak/does not generalize" if rho < 0.2 else "moderate"))
    print("wrote totalseg_law.json")


if __name__ == "__main__":
    main()
