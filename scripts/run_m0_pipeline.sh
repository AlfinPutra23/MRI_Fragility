#!/usr/bin/env bash
# M0 full pipeline: build dataset -> preprocess -> train (250ep, GPU0) -> predict@R -> fragility curve.
# Chained with set -e so a failed/invalid earlier stage never reaches the multi-hour train.
set -euo pipefail

ENVBIN=${ENVBIN:-/home/user/anaconda3/envs/mrifrag/bin}
P=${P:-/media/user/B4864CD4864C98AE/mri_fragility}
export nnUNet_raw=$P/nnUNet_raw
export nnUNet_preprocessed=$P/nnUNet_preprocessed
export nnUNet_results=$P/nnUNet_results
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}     # idle GPU
TR=nnUNetTrainer_250epochs
D=$nnUNet_raw/Dataset501_MRIfrag

banner(){ echo; echo "==================== $* @ $(date '+%F %T') ===================="; echo; }

banner "STAGE 1/5 build dataset (CPU)"
cd "$P/scripts"
if [ "$(ls "$D/imagesTr" 2>/dev/null | wc -l)" -eq 540 ] && [ "$(ls "$D/imagesTs_R8" 2>/dev/null | wc -l)" -eq 240 ]; then
  echo "SKIP: dataset already built (540 train / 240 test)."
else
  "$ENVBIN/python" build_nnunet_dataset.py --raw_out "$nnUNet_raw" --id 501 --R_train 2 --R_test 1 2 4 6 8
fi

banner "STAGE 2/5 plan_and_preprocess + verify integrity (CPU)"
if [ "$(ls "$nnUNet_preprocessed/Dataset501_MRIfrag/nnUNetPlans_3d_fullres/"*.b2nd 2>/dev/null | wc -l)" -ge 1080 ]; then
  echo "SKIP: preprocessing already done (>=1080 b2nd files)."
else
  "$ENVBIN/nnUNetv2_plan_and_preprocess" -d 501 --verify_dataset_integrity -np 8
fi

banner "STAGE 3/5 train 3d_fullres fold0 ($TR) on GPU $CUDA_VISIBLE_DEVICES"
MODELDIR="$nnUNet_results/Dataset501_MRIfrag/${TR}__nnUNetPlans__3d_fullres/fold_0"
if [ -f "$MODELDIR/checkpoint_final.pth" ]; then
  echo "SKIP: training already complete (checkpoint_final.pth present)."
elif [ -f "$MODELDIR/checkpoint_latest.pth" ]; then
  echo "RESUME: continuing from checkpoint_latest.pth"
  "$ENVBIN/nnUNetv2_train" 501 3d_fullres 0 -tr "$TR" --c
else
  "$ENVBIN/nnUNetv2_train" 501 3d_fullres 0 -tr "$TR"
fi

banner "STAGE 4/5 predict at each R (GPU)"
for tag in clean R2 R4 R6 R8; do
  echo "--- predict $tag ---"
  "$ENVBIN/nnUNetv2_predict" -i "$D/imagesTs_$tag" -o "$D/predsTs_$tag" \
      -d 501 -c 3d_fullres -f 0 -tr "$TR" --disable_tta
done

banner "STAGE 5/5 fragility curve (CPU)"
"$ENVBIN/python" fragility_eval.py --root "$D" --R 1 2 4 6 8

banner "M0 PIPELINE DONE"
