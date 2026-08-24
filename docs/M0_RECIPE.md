# M0 — Per-organ MRI-acceleration fragility gate (run recipe)

**Goal:** produce the **per-organ Dice-vs-R fragility curve** that proves or kills the premise:
*as MRI acceleration R grows, small organs' segmentation collapses faster than large organs', while image
metrics (SSIM/PSNR) stay high.* Train one nnU-Net on mildly-accelerated data, test it across R, plot per-organ Dice.

All scripts use **relative paths** (`scripts/paths.py`) — the whole project folder is relocatable.

---
## Already done (no-GPU de-risk; results in `outputs/`)
- **Data verified:** MRISegmentator-Abdomen — 780 series (195 pts × {PRE,ART,VEN,DEL}), 135/60 split, 62 structs,
  magnitude MRI, ~1.1×1.1×3.0 mm.
- **Gradient-imbalance premise PASSES:** liver/adrenal = **448×** volume (= loss-gradient mass) ratio
  → `scripts/m0_audit.py`, `outputs/plots/m0_organ_volume_imbalance.png`.
- **k-space simulator validated:** SSIM 0.97→0.75, PSNR 38→27 dB for R=2→8 → `scripts/m0_kspace_sim.py`,
  `outputs/plots/m0_kspace_sim.png`, `outputs/results/m0_kspace_metrics.json`.

## scripts/
| file | role |
|---|---|
| `paths.py` | relocatable project paths (root, data, outputs) — imported by all |
| `labels.py` | authoritative 62-label map + ABDO(13) + TAIL{4,6,11,12,13,17} |
| `kspace.py` | canonical retrospective undersampling (variable-density Cartesian + ACS) |
| `build_nnunet_dataset.py` | MRISegmentator → nnU-Net v2 dataset, undersampled @R |
| `fragility_eval.py` | per-organ Dice-vs-R curve + metric-blindness panel + verdict |
| `m0_audit.py`, `m0_kspace_sim.py` | the no-GPU de-risk scripts (already run) |

---
## 1. New env
```bash
conda create -n mrifrag python=3.11 -y && conda activate mrifrag
pip install torch --index-url https://download.pytorch.org/whl/cu128   # match your CUDA
pip install -r scripts/requirements.txt
# nnU-Net env vars (point inside the project so everything stays together):
P=/media/user/B4864CD4864C98AE/mri_fragility
export nnUNet_raw=$P/nnUNet_raw nnUNet_preprocessed=$P/nnUNet_preprocessed nnUNet_results=$P/nnUNet_results
mkdir -p $nnUNet_raw $nnUNet_preprocessed $nnUNet_results
```

## 2. Build the dataset (CPU, ~5–10 min, ~12 GB)
```bash
cd $P/scripts
python build_nnunet_dataset.py --raw_out $nnUNet_raw --id 501 --R_train 2 --R_test 1 2 4 6 8
```
- Each patient×phase = one case (540 train / 240 test). Train images undersampled @R=2; test folders
  `imagesTs_{clean,R2,R4,R6,R8}`; GT in `labelsTs/`.
- `--R_train 2` = "trained mild, deployed across R" (realistic). Try `--R_train 1` for the pure
  information-fragility variant. (Robust method later = train with mixed-R augmentation.)

## 3. Preprocess + train (GPU)
```bash
nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity
nnUNetv2_train 501 3d_fullres 0          # ~8–24 h. Fast de-risk: add  -tr nnUNetTrainer_250epochs
```

## 4. Predict at each R (GPU, fast)
```bash
D=$nnUNet_raw/Dataset501_MRIfrag
for tag in clean R2 R4 R6 R8; do
  nnUNetv2_predict -i $D/imagesTs_$tag -o $D/predsTs_$tag -d 501 -c 3d_fullres -f 0
done
```

## 5. The fragility curve (CPU)
```bash
python fragility_eval.py --root $D --R 1 2 4 6 8
# -> outputs/plots/m0_fragility_curve.png , outputs/results/m0_fragility_dice.json
#    prints mean Dice drop TAIL vs LARGE + "PREMISE HOLDS / WEAK" verdict
```

## The gate
- **HOLDS** (tail Dice drop ≫ large-organ drop, SSIM ~flat) → proceed to **M1** (during-training gradient
  measurement) then the gradient-rebalancing method.
- **WEAK** → method premise dies → pivot to the benchmark-only paper (still publishable).

## Notes
- Disk: dataset ~12 GB + nnU-Net preprocessed ~12 GB + results — keep ~40 GB free (1 TB disk has 1.4 TB).
- GPU is the user's machine — train when one is free.
- AMOS22-MRI (60 vols, `data/amos_mri/`) is the secondary/held-out twin for generalization.
- If you relocate the project, the python paths auto-adjust; only re-export the nnU-Net env vars.
