#!/usr/bin/env bash
# Mixed-R upgrade: build (CPU now) -> preprocess (CPU) -> wait for a free GPU -> train seed-matched
# Uniform + FragW4 on mixed-R -> predict on Dataset501's test -> 2x2 ablation compare. Idempotent.
set -uo pipefail
ENVBIN=${ENVBIN:-/home/user/anaconda3/envs/mrifrag/bin}
P=${P:-/media/user/B4864CD4864C98AE/mri_fragility}
VAR=${VAR:-/home/user/anaconda3/envs/mrifrag/lib/python3.11/site-packages/nnunetv2/training/nnUNetTrainer/variants}
export nnUNet_raw=$P/nnUNet_raw nnUNet_preprocessed=$P/nnUNet_preprocessed nnUNet_results=$P/nnUNet_results
D501=$nnUNet_raw/Dataset501_MRIfrag
D504=$nnUNet_raw/Dataset504_MRIfragMixedR
cd "$P/scripts"

echo "==================== BUILD mixed-R (CPU) @ $(date '+%F %T') ===================="
if [ "$(ls "$D504/imagesTr"/*.nii.gz 2>/dev/null | wc -l)" -ge 540 ]; then echo "SKIP build"; else
  "$ENVBIN/python" build_mixedr_dataset.py --raw_out "$nnUNet_raw" --id 504; fi

echo "==================== PREPROCESS 504 (CPU) @ $(date '+%F %T') ===================="
if ls "$nnUNet_preprocessed/Dataset504_MRIfragMixedR/nnUNetPlans_3d_fullres/"*.b2nd >/dev/null 2>&1; then echo "SKIP preprocess"; else
  "$ENVBIN/nnUNetv2_plan_and_preprocess" -d 504 --verify_dataset_integrity -np 6; fi

echo "[$(date '+%F %T')] waiting for a free GPU (AMOS + λ-runs to finish)..."
until ! ps -eo cmd | grep -qE "[n]nUNetv2_train|[n]nUNetv2_predict"; do sleep 300; done
export CUDA_VISIBLE_DEVICES=0
cp "$P/scripts/nnunet_sweep_trainers.py" "$VAR/"

for TR in nnUNetTrainer_Uniform_s42 nnUNetTrainer_FragW4_s42; do
  MDIR=$nnUNet_results/Dataset504_MRIfragMixedR/${TR}__nnUNetPlans__3d_fullres/fold_0
  echo "[$(date '+%F %T')] TRAIN $TR on mixed-R (504)"
  if [ ! -f "$MDIR/checkpoint_final.pth" ]; then
    if [ -f "$MDIR/checkpoint_latest.pth" ]; then "$ENVBIN/nnUNetv2_train" 504 3d_fullres 0 -tr "$TR" --c > "$P/outputs/logs/mixedr_$TR.log" 2>&1
    else "$ENVBIN/nnUNetv2_train" 504 3d_fullres 0 -tr "$TR" > "$P/outputs/logs/mixedr_$TR.log" 2>&1; fi
  fi
  for tag in clean R8; do          # predict the 504 model on Dataset501's identical test images
    o="$D501/predsMIXEDR_${TR}_$tag"
    [ "$(ls "$o"/*.nii.gz 2>/dev/null | wc -l)" -eq 240 ] && continue
    "$ENVBIN/nnUNetv2_predict" -i "$D501/imagesTs_$tag" -o "$o" -d 504 -c 3d_fullres -f 0 -tr "$TR" --disable_tta \
        >> "$P/outputs/logs/mixedr_predict.log" 2>&1
  done
done

echo "==================== MIXED-R 2x2 COMPARE @ $(date '+%F %T') ===================="
"$ENVBIN/python" mixedr_compare.py --root "$D501"
echo "==================== MIXED-R DONE @ $(date '+%F %T') ===================="
