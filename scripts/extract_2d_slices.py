"""B1 data prep: extract 2D axial slices (image + organ labels) from MRISegmentator for the joint
mask+seg training. Keeps slices containing abdominal organs, center-crops to a fixed size.
Saves compact .npz: images (N,S,S) float16, labels (N,S,S) uint8.

  python extract_2d_slices.py --split train --size 256 --out ../data/slices_train.npz
"""
import os, glob, argparse
import numpy as np
import nibabel as nib
from paths import MRISEG_REL as REL
import labels as L


def crop_center(arr2d, lab2d, size):
    """center-crop/pad to size x size, centered on the labeled-organ bounding box (fallback: image center)."""
    ys, xs = np.where(np.isin(lab2d, list(L.ABDO)))
    cy, cx = (int(ys.mean()), int(xs.mean())) if len(ys) else (arr2d.shape[0] // 2, arr2d.shape[1] // 2)
    out_i = np.zeros((size, size), arr2d.dtype); out_l = np.zeros((size, size), lab2d.dtype)
    y0, x0 = cy - size // 2, cx - size // 2
    sy0, sx0 = max(-y0, 0), max(-x0, 0)
    y0, x0 = max(y0, 0), max(x0, 0)
    h = min(size - sy0, arr2d.shape[0] - y0); w = min(size - sx0, arr2d.shape[1] - x0)
    out_i[sy0:sy0+h, sx0:sx0+w] = arr2d[y0:y0+h, x0:x0+w]
    out_l[sy0:sy0+h, sx0:sx0+w] = lab2d[y0:y0+h, x0:x0+w]
    return out_i, out_l


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--min_organ_vox", type=int, default=200)   # slice must have this many abdo voxels
    ap.add_argument("--max_slices_per_vol", type=int, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()
    idir, ldir = ("ImageTr", "labelsTr") if args.split == "train" else ("ImageTs", "labelsTs")
    lps = sorted(glob.glob(f"{REL}/{ldir}/*.nii.gz"))
    if args.smoke:
        lps = lps[:args.smoke]
    imgs, labs = [], []
    abdo = list(L.ABDO)
    for i, lp in enumerate(lps):
        case = os.path.basename(lp)[:-7]
        ip = f"{REL}/{idir}/{case}_0000.nii.gz"
        if not os.path.exists(ip):
            continue
        im = np.asanyarray(nib.load(ip).dataobj).astype(np.float32)
        la = np.asanyarray(nib.load(lp).dataobj).astype(np.int16)
        # per-slice abdominal-organ voxel count; prefer slices with tail organs
        counts = np.isin(la, abdo).sum(axis=(0, 1))
        tail_counts = np.isin(la, list(L.TAIL)).sum(axis=(0, 1))
        cand = np.where(counts >= args.min_organ_vox)[0]
        cand = sorted(cand, key=lambda z: -tail_counts[z])[:args.max_slices_per_vol]
        mx = np.percentile(im, 99.5) + 1e-6
        for z in cand:
            ci, cl = crop_center(im[:, :, z], la[:, :, z], args.size)
            imgs.append((np.clip(ci / mx, 0, 1)).astype(np.float16))   # normalized 0..1
            labs.append(cl.astype(np.uint8))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(lps)} vols, {len(imgs)} slices")
    imgs = np.stack(imgs); labs = np.stack(labs)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, images=imgs, labels=labs)
    print(f"saved {imgs.shape[0]} slices {imgs.shape[1:]} -> {args.out}  "
          f"({imgs.nbytes/1e6:.0f}MB img)  tail-organ slices kept first")


if __name__ == "__main__":
    main()
