"""Upgrade: mixed-R training dataset. Each train case is undersampled at a RANDOM R in {2,4,6,8}
(instead of all @R2), so the model sees R8-like blur during training. Test sets are reused from
Dataset501 (identical), so we only build imagesTr/labelsTr here.

  python build_mixedr_dataset.py --raw_out $nnUNet_raw --id 504
Then train the seed-matched trainers on 502... err 504 and predict on Dataset501's imagesTs_*.
"""
import os, glob, json, argparse, shutil
import numpy as np
import nibabel as nib
from kspace import undersample_volume
from paths import MRISEG_REL as REL
import labels as L


def train_cases(rel):
    out = []
    for lp in sorted(glob.glob(f"{rel}/labelsTr/*.nii.gz")):
        case = os.path.basename(lp)[:-7]
        ip = f"{rel}/ImageTr/{case}_0000.nii.gz"
        if os.path.exists(ip):
            out.append((case, ip, lp))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_out", required=True)
    ap.add_argument("--id", type=int, default=504)
    ap.add_argument("--R_choices", type=float, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    root = f"{args.raw_out}/Dataset{args.id:03d}_MRIfragMixedR"
    os.makedirs(f"{root}/imagesTr", exist_ok=True)
    os.makedirs(f"{root}/labelsTr", exist_ok=True)

    tr = train_cases(REL)
    if args.smoke:
        tr = tr[:args.smoke]
    assigns = {}
    counts = {int(R): 0 for R in args.R_choices}
    for i, (case, ip, lp) in enumerate(tr):
        R = int(rng.choice(args.R_choices))
        assigns[case] = R; counts[R] += 1
        im = nib.load(ip)
        arr = np.asanyarray(im.dataobj).astype(np.float32)
        rec = undersample_volume(arr, R, seed=hash(case) % 99991)
        nib.save(nib.Nifti1Image(rec, im.affine, im.header), f"{root}/imagesTr/{case}_0000.nii.gz")
        shutil.copy(lp, f"{root}/labelsTr/{case}.nii.gz")
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(tr)}  (R counts so far: {counts})")

    dj = dict(channel_names={"0": "MRI"}, labels=L.dataset_json_labels(), numTraining=len(tr),
              file_ending=".nii.gz", description=f"MRISegmentator k-space MIXED-R train (R in {args.R_choices})")
    json.dump(dj, open(f"{root}/dataset.json", "w"), indent=2)
    json.dump(assigns, open(f"{root}/mixedR_assignments.json", "w"), indent=2)
    print(f"\nper-case R counts: {counts}")
    print(f"wrote {root}/dataset.json + mixedR_assignments.json\nDONE -> {root}")
    print("Test sets: reuse Dataset501's imagesTs_*/labelsTs at predict/eval time (identical).")


if __name__ == "__main__":
    main()
