#!/usr/bin/env bash
# Recover the crashed FragTopK variant: retrain on GPU1 (fixed DC_and_topk loss), predict, then
# re-run the 4-way sweep compare once all variants are predicted.
set -uo pipefail
ENVBIN=${ENVBIN:-/home/user/anaconda3/envs/mrifrag/bin}
P=${P:-/media/user/B4864CD4864C98AE/mri_fragility}
export nnUNet_raw=$P/nnUNet_raw nnUNet_preprocessed=$P/nnUNet_preprocessed nnUNet_results=$P/nnUNet_results
export CUDA_VISIBLE_DEVICES=1
TR=nnUNetTrainer_FragTopK_s42
D=$nnUNet_raw/Dataset501_MRIfrag
MDIR=$nnUNet_results/Dataset501_MRIfrag/${TR}__nnUNetPlans__3d_fullres/fold_0

echo "[$(date '+%F %T')] TRAIN $TR on GPU1"
if [ ! -f "$MDIR/checkpoint_final.pth" ]; then
  if [ -f "$MDIR/checkpoint_latest.pth" ]; then
    "$ENVBIN/nnUNetv2_train" 501 3d_fullres 0 -tr "$TR" --c > "$P/outputs/logs/sweep_${TR}.log" 2>&1
  else
    "$ENVBIN/nnUNetv2_train" 501 3d_fullres 0 -tr "$TR" > "$P/outputs/logs/sweep_${TR}.log" 2>&1
  fi
fi

echo "[$(date '+%F %T')] PREDICT $TR (clean,R8)"
for tag in clean R8; do
  o="$D/predsSW_${TR}_$tag"; rm -rf "$o"
  "$ENVBIN/nnUNetv2_predict" -i "$D/imagesTs_$tag" -o "$o" -d 501 -c 3d_fullres -f 0 -tr "$TR" --disable_tta \
      >> "$P/outputs/logs/sweep_predict.log" 2>&1
done

echo "[$(date '+%F %T')] wait for the other 3 variants' R8 preds (from main sweep)..."
for tr in Uniform_s42 FragW4_s42 FragW2_s42; do
  until [ "$(ls "$D/predsSW_nnUNetTrainer_${tr}_R8"/*.nii.gz 2>/dev/null | wc -l)" -eq 240 ]; do sleep 120; done
done

echo "[$(date '+%F %T')] FINAL 4-way COMPARE"
cd "$P/scripts"
"$ENVBIN/python" sweep_compare.py --root "$D"
echo "[$(date '+%F %T')] TOPK RECOVER DONE"
