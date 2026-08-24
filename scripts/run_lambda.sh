#!/usr/bin/env bash
# λ-extension upgrade: train FragW8 (×8) + FragW6 (×6) on GPU1 (AMOS is on GPU0), predict @{clean,R8},
# then the dose-response compare vs the existing ×2/×4/uniform. Idempotent.
set -uo pipefail
ENVBIN=${ENVBIN:-/home/user/anaconda3/envs/mrifrag/bin}
P=${P:-/media/user/B4864CD4864C98AE/mri_fragility}
VAR=${VAR:-/home/user/anaconda3/envs/mrifrag/lib/python3.11/site-packages/nnunetv2/training/nnUNetTrainer/variants}
export nnUNet_raw=$P/nnUNet_raw nnUNet_preprocessed=$P/nnUNet_preprocessed nnUNet_results=$P/nnUNet_results
export CUDA_VISIBLE_DEVICES=1
D=$nnUNet_raw/Dataset501_MRIfrag
cp "$P/scripts/nnunet_sweep_trainers.py" "$VAR/"

for TR in nnUNetTrainer_FragW8_s42 nnUNetTrainer_FragW6_s42; do
  MDIR=$nnUNet_results/Dataset501_MRIfrag/${TR}__nnUNetPlans__3d_fullres/fold_0
  echo "[$(date '+%F %T')] TRAIN $TR (GPU1)"
  if [ ! -f "$MDIR/checkpoint_final.pth" ]; then
    if [ -f "$MDIR/checkpoint_latest.pth" ]; then "$ENVBIN/nnUNetv2_train" 501 3d_fullres 0 -tr "$TR" --c > "$P/outputs/logs/sweep_$TR.log" 2>&1
    else "$ENVBIN/nnUNetv2_train" 501 3d_fullres 0 -tr "$TR" > "$P/outputs/logs/sweep_$TR.log" 2>&1; fi
  else echo "SKIP train"; fi
  for tag in clean R8; do
    o="$D/predsSW_${TR}_$tag"
    [ "$(ls "$o"/*.nii.gz 2>/dev/null | wc -l)" -eq 240 ] && continue
    "$ENVBIN/nnUNetv2_predict" -i "$D/imagesTs_$tag" -o "$o" -d 501 -c 3d_fullres -f 0 -tr "$TR" --disable_tta \
        >> "$P/outputs/logs/sweep_predict.log" 2>&1
  done
done

echo "[$(date '+%F %T')] LAMBDA dose-response compare"
cd "$P/scripts"; "$ENVBIN/python" lambda_compare.py --root "$D"
echo "[$(date '+%F %T')] LAMBDA DONE"
