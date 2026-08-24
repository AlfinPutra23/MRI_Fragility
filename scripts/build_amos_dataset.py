"""M3: build an nnU-Net v2 dataset from AMOS22-MRI with retrospective k-space undersampling.
Train = 40 AMOS train vols (undersampled @R_train); test = 20 AMOS valid vols (undersampled @ each R_test).
Through-plane (slice) axis is auto-detected per volume as the largest-spacing axis.

  python build_amos_dataset.py --raw_out $nnUNet_raw --id 502 --R_train 2 --R_test 1 2 4 6 8
"""
import os, glob, json, argparse, shutil
import numpy as np
import nibabel as nib
from kspace import undersample_volume
from paths import AMOS
import amos_labels as L


def pairs(split):
    idir, ldir = ("train/imagesTr", "train/labelsTr") if split == "train" else ("valid/imagesVa", "valid/labelsVa")
    out = []
    for ip in sorted(glob.glob(f"{AMOS}/{idir}/*.nii.gz")):
        b = os.path.basename(ip)
        lp = f"{AMOS}/{ldir}/{b}"
        if os.path.exists(lp):
            out.append((b[:-7], ip, lp))     # case id = basename w/o .nii.gz
    return out


def write_us(ip, dst, R, seed):
    im = nib.load(ip)
    arr = np.asanyarray(im.dataobj).astype(np.float32)
    sax = int(np.argmax(im.header.get_zooms()[:3]))      # through-plane = largest spacing
    rec = undersample_volume(arr, R, seed=seed, slice_axis=sax)
    nib.save(nib.Nifti1Image(rec, im.affine, im.header), dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_out", required=True)
    ap.add_argument("--id", type=int, default=502)
    ap.add_argument("--R_train", type=float, default=2)
    ap.add_argument("--R_test", type=float, nargs="+", default=[1, 2, 4, 6, 8])
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()
    root = f"{args.raw_out}/Dataset{args.id:03d}_AMOSfrag"
    for d in ("imagesTr", "labelsTr", "labelsTs"):
        os.makedirs(f"{root}/{d}", exist_ok=True)

    tr, ts = pairs("train"), pairs("test")
    if args.smoke:
        tr, ts = tr[:args.smoke], ts[:args.smoke]
    print(f"AMOS train {len(tr)} | test {len(ts)} | R_train={args.R_train} R_test={args.R_test}")

    for i, (case, ip, lp) in enumerate(tr):
        write_us(ip, f"{root}/imagesTr/{case}_0000.nii.gz", args.R_train, seed=hash(case) % 99991)
        shutil.copy(lp, f"{root}/labelsTr/{case}.nii.gz")
        if (i + 1) % 10 == 0:
            print(f"  train {i+1}/{len(tr)}")
    for R in args.R_test:
        tag = "clean" if R <= 1 else f"R{int(R)}"
        os.makedirs(f"{root}/imagesTs_{tag}", exist_ok=True)
        for case, ip, lp in ts:
            write_us(ip, f"{root}/imagesTs_{tag}/{case}_0000.nii.gz", R, seed=hash(case) % 99991)
        print(f"  test imagesTs_{tag} done ({len(ts)})")
    for case, ip, lp in ts:
        shutil.copy(lp, f"{root}/labelsTs/{case}.nii.gz")

    dj = dict(channel_names={"0": "MRI"}, labels=L.dataset_json_labels(), numTraining=len(tr),
              file_ending=".nii.gz", description=f"AMOS22-MRI k-space undersampled @R_train={args.R_train}")
    json.dump(dj, open(f"{root}/dataset.json", "w"), indent=2)
    print(f"wrote {root}/dataset.json ({len(dj['labels'])} labels incl bg)\nDONE -> {root}")


if __name__ == "__main__":
    main()
