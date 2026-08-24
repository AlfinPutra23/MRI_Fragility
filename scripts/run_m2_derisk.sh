#!/usr/bin/env bash
# M2-entry de-risk: fragility-weighted-loss seg model vs uniform (M0), across R.
# AUTO-FIRES after the gradlog training frees GPU0. Chained, idempotent.
set -euo pipefail
ENVBIN=${ENVBIN:-/home/user/anaconda3/envs/mrifrag/bin}
P=${P:-/media/user/B4864CD4864C98AE/mri_fragility}
VAR=${VAR:-/home/user/anaconda3/envs/mrifrag/lib/python3.11/site-packages/nnunetv2/training/nnUNetTrainer/variants}
export nnUNet_raw=$P/nnUNet_raw nnUNet_preprocessed=$P/nnUNet_preprocessed nnUNet_results=$P/nnUNet_results
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
TR=nnUNetTrainer_FragWeighted
D=$nnUNet_raw/Dataset501_MRIfrag
MODELDIR="$nnUNet_results/Dataset501_MRIfrag/${TR}__nnUNetPlans__3d_fullres/fold_0"

echo "[$(date '+%F %T')] waiting for gradlog training to free GPU0..."
until ! ps -eo cmd | grep -q "[n]nUNetv2_train 501 3d_fullres 0 -tr nnUNetTrainer_GradLog"; do sleep 120; done
echo "[$(date '+%F %T')] GPU0 free — starting fragility-weighted de-risk"

cp "$P/scripts/nnunet_fragweighted_trainer.py" "$VAR/"          # ensure trainer installed

echo "==================== TRAIN ($TR, 250ep) ===================="
if [ -f "$MODELDIR/checkpoint_final.pth" ]; then
  echo "SKIP: training already complete."
elif [ -f "$MODELDIR/checkpoint_latest.pth" ]; then
  "$ENVBIN/nnUNetv2_train" 501 3d_fullres 0 -tr "$TR" --c
else
  "$ENVBIN/nnUNetv2_train" 501 3d_fullres 0 -tr "$TR"
fi

echo "==================== PREDICT across R (no-TTA) ===================="
for tag in clean R2 R4 R6 R8; do
  if [ "$(ls "$D/predsW_$tag"/*.nii.gz 2>/dev/null | wc -l)" -eq 240 ]; then
    echo "SKIP predsW_$tag (240 done)"; continue; fi
  "$ENVBIN/nnUNetv2_predict" -i "$D/imagesTs_$tag" -o "$D/predsW_$tag" \
      -d 501 -c 3d_fullres -f 0 -tr "$TR" --disable_tta
done

echo "==================== COMPARE vs uniform (M0) ===================="
cd "$P/scripts"
"$ENVBIN/python" m2_compare.py --root "$D" --R 1 2 4 6 8
echo "==================== M2 DERISK DONE @ $(date '+%F %T') ===================="
