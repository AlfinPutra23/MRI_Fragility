"""Build an AMOS mixed-R training set (Dataset506): each of the 40 AMOS train vols undersampled at a RANDOM R in
{2,4,6,8} (instead of all @R2). Test is REUSED from Dataset502 (identical), so we only build imagesTr/labelsTr.
Mirrors build_mixedr_dataset.py (MRISeg->504) for AMOS->506.

  python build_amos_mixedr.py --raw_out $nnUNet_raw --id 506
"""
import os, glob, json, argparse, shutil
import numpy as np
import nibabel as nib
from kspace import undersample_volume
from paths import AMOS
import amos_labels as L


def train_pairs():
    out = []
    for ip in sorted(glob.glob(f"{AMOS}/train/imagesTr/*.nii.gz")):
        b = os.path.basename(ip); lp = f"{AMOS}/train/labelsTr/{b}"
        if os.path.exists(lp): out.append((b[:-7], ip, lp))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_out", required=True)
    ap.add_argument("--id", type=int, default=506)
    ap.add_argument("--R_choices", type=float, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    root = f"{args.raw_out}/Dataset{args.id:03d}_AMOSfragMixedR"
    os.makedirs(f"{root}/imagesTr", exist_ok=True); os.makedirs(f"{root}/labelsTr", exist_ok=True)

    tr = train_pairs()
    assigns = {}; counts = {int(R): 0 for R in args.R_choices}
    for case, ip, lp in tr:
        R = int(rng.choice(args.R_choices)); assigns[case] = R; counts[R] += 1
        im = nib.load(ip); arr = np.asanyarray(im.dataobj).astype(np.float32)
        sax = int(np.argmax(im.header.get_zooms()[:3]))                      # through-plane = largest spacing (AMOS convention)
        rec = undersample_volume(arr, R, seed=hash(case) % 99991, slice_axis=sax)
        nib.save(nib.Nifti1Image(rec, im.affine, im.header), f"{root}/imagesTr/{case}_0000.nii.gz")
        shutil.copy(lp, f"{root}/labelsTr/{case}.nii.gz")
    dj = dict(channel_names={"0": "MRI"}, labels=L.dataset_json_labels(), numTraining=len(tr),
              file_ending=".nii.gz", description="AMOS22-MRI mixed-R (R in {2,4,6,8}) training set; test reused from Dataset502")
    json.dump(dj, open(f"{root}/dataset.json", "w"), indent=2)
    json.dump({"assignments": assigns, "counts": counts}, open(f"{root}/mixedR_assignments.json", "w"), indent=2)
    print(f"AMOS mixed-R built: {len(tr)} train vols, R counts={counts} -> {root}")


if __name__ == "__main__":
    main()
