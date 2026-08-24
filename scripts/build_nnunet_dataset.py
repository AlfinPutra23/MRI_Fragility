"""M0 step 1: build an nnU-Net v2 dataset from MRISegmentator-Abdomen with retrospective
k-space undersampling.

Produces (under <raw_out>/Dataset<ID>_MRIfrag/):
  imagesTr/<case>_0000.nii.gz   train images undersampled at R_train   (+ labelsTr/<case>.nii.gz)
  imagesTs_R{R}/<case>_0000.nii.gz   test images undersampled at each R in R_test
  labelsTs/<case>.nii.gz        test GT (once; identical across R)
  dataset.json
Each *series* (patient x {PRE,ART,VEN,DEL}) is one case -> 540 train / 240 test.

Example (full):
  python build_nnunet_dataset.py --raw_out $nnUNet_raw --id 501 --R_train 2 --R_test 1 2 4 6 8
Smoke test (2 cases each):
  python build_nnunet_dataset.py --raw_out /tmp/raw --id 999 --smoke 2
"""
import os, glob, json, argparse, shutil
import numpy as np
import nibabel as nib
from kspace import undersample_volume
from paths import MRISEG_REL as REL_DEFAULT
import labels as L


def cases(rel, split):
    idir, ldir = ("ImageTr", "labelsTr") if split == "train" else ("ImageTs", "labelsTs")
    out = []
    for lp in sorted(glob.glob(f"{rel}/{ldir}/*.nii.gz")):
        case = os.path.basename(lp)[:-7]                       # e.g. train_001_VEN
        ip = f"{rel}/{idir}/{case}_0000.nii.gz"
        if os.path.exists(ip):
            out.append((case, ip, lp))
    return out


def write_us(ip, dst, R, seed):
    im = nib.load(ip)
    arr = np.asanyarray(im.dataobj).astype(np.float32)
    rec = undersample_volume(arr, R, seed=seed)
    nib.save(nib.Nifti1Image(rec, im.affine, im.header), dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rel", default=REL_DEFAULT, help="MRISegmentator Release dir")
    ap.add_argument("--raw_out", required=True, help="nnUNet_raw dir")
    ap.add_argument("--id", type=int, default=501)
    ap.add_argument("--R_train", type=float, default=2)
    ap.add_argument("--R_test", type=float, nargs="+", default=[1, 2, 4, 6, 8])
    ap.add_argument("--smoke", type=int, default=0, help="limit to N cases per split (validation)")
    args = ap.parse_args()

    root = f"{args.raw_out}/Dataset{args.id:03d}_MRIfrag"
    os.makedirs(f"{root}/imagesTr", exist_ok=True)
    os.makedirs(f"{root}/labelsTr", exist_ok=True)
    os.makedirs(f"{root}/labelsTs", exist_ok=True)

    tr = cases(args.rel, "train")
    ts = cases(args.rel, "test")
    if args.smoke:
        tr, ts = tr[:args.smoke], ts[:args.smoke]
    print(f"train cases: {len(tr)}  | test cases: {len(ts)}  | R_train={args.R_train} R_test={args.R_test}")

    for i, (case, ip, lp) in enumerate(tr):
        write_us(ip, f"{root}/imagesTr/{case}_0000.nii.gz", args.R_train, seed=hash(case) % 99991)
        shutil.copy(lp, f"{root}/labelsTr/{case}.nii.gz")
        if (i + 1) % 50 == 0:
            print(f"  train {i+1}/{len(tr)}")
    print("train done")

    for R in args.R_test:
        tag = "clean" if R <= 1 else f"R{int(R)}"
        d = f"{root}/imagesTs_{tag}"
        os.makedirs(d, exist_ok=True)
        for case, ip, lp in ts:
            write_us(ip, f"{d}/{case}_0000.nii.gz", R, seed=hash(case) % 99991)
        print(f"  test imagesTs_{tag} done ({len(ts)})")
    for case, ip, lp in ts:                                    # GT once
        shutil.copy(lp, f"{root}/labelsTs/{case}.nii.gz")

    dj = dict(channel_names={"0": "MRI"}, labels=L.dataset_json_labels(),
              numTraining=len(tr), file_ending=".nii.gz",
              description=f"MRISegmentator-Abdomen, k-space undersampled @R_train={args.R_train}")
    json.dump(dj, open(f"{root}/dataset.json", "w"), indent=2)
    print(f"\nwrote {root}/dataset.json  ({len(dj['labels'])} labels incl. background)")
    print(f"DONE -> {root}")


if __name__ == "__main__":
    main()
