#!/usr/bin/env bash
# WACV fix "single fold": train nnU-Net folds 1-4 (fold 0 already done) on Dataset501, predict each fold
# across R, then aggregate per-fold fragility -> mean +- std (cross-validated benchmark + law with error bars).
# nnU-Net does 5-fold CV on the 540 TRAIN cases; each fold = a different 432/108 train/val split -> 5 models.
set -uo pipefail
ENVBIN=${ENVBIN:-/home/user/anaconda3/envs/mrifrag/bin}
P=${P:-/media/user/B4864CD4864C98AE/mri_fragility}
export nnUNet_raw=$P/nnUNet_raw nnUNet_preprocessed=$P/nnUNet_preprocessed nnUNet_results=$P/nnUNet_results
export PYTHONUNBUFFERED=1
TR=nnUNetTrainer_250epochs; D=$nnUNet_raw/Dataset501_MRIfrag

# --- train folds 1-4 in parallel across 2 GPUs (fold 0 exists) ---
echo "[$(date '+%F %T')] waiting for a free GPU..."; until ! ps -eo cmd|grep -qE "[n]nUNetv2_train"; do sleep 300; done
train_fold(){  # gpu fold
  local mdir="$nnUNet_results/Dataset501_MRIfrag/${TR}__nnUNetPlans__3d_fullres/fold_$2"
  [ -f "$mdir/checkpoint_final.pth" ] && { echo "SKIP fold $2"; return; }
  CUDA_VISIBLE_DEVICES=$1 "$ENVBIN/nnUNetv2_train" 501 3d_fullres $2 -tr "$TR" > "$P/outputs/logs/fold${2}.log" 2>&1
}
for pair in "1 2" "3 4"; do
  set -- $pair
  train_fold 0 $1 & train_fold 1 $2 & wait
done

# --- predict each fold across R (no-TTA), then per-fold fragility ---
for f in 1 2 3 4; do
  mdir="$nnUNet_results/Dataset501_MRIfrag/${TR}__nnUNetPlans__3d_fullres/fold_$f"
  [ -f "$mdir/checkpoint_final.pth" ] || { echo "SKIP fold $f: no final checkpoint (not trained)"; continue; }
  for tag in clean R2 R4 R6 R8; do
    o="$D/predsF${f}_$tag"
    [ "$(ls "$o"/*.nii.gz 2>/dev/null|wc -l)" -eq 240 ] && continue
    CUDA_VISIBLE_DEVICES=0 "$ENVBIN/nnUNetv2_predict" -i "$D/imagesTs_$tag" -o "$o" -d 501 -c 3d_fullres -f $f -tr "$TR" --disable_tta >> "$P/outputs/logs/multifold_predict.log" 2>&1
  done
  # guard: only run fragility_eval if predictions are complete, else it emits a NaN figure/json (the earlier garbage)
  [ "$(ls "$D/predsF${f}_R8"/*.nii.gz 2>/dev/null|wc -l)" -eq 240 ] || { echo "SKIP fold $f eval: predictions incomplete"; continue; }
  cd "$P/scripts"; "$ENVBIN/python" fragility_eval.py --root "$D" --preds_tpl "predsF${f}_{tag}" --R 1 2 4 6 8 --out_prefix "fold${f}" --kspace_metrics /dev/null 2>&1 | grep -E "PREMISE|TAIL"
done

echo "==================== MULTIFOLD AGGREGATE @ $(date '+%T') ===================="
cd "$P"   # aggregate + figure use paths relative to the project ROOT (not scripts/) -- this was a latent bug
"$ENVBIN/python" scripts/multifold_aggregate.py && "$ENVBIN/python" scripts/make_multifold_figure.py
echo "==================== MULTIFOLD DONE @ $(date '+%F %T') ===================="
