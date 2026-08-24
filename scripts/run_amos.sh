#!/usr/bin/env bash
# M3 generalization: AMOS22-MRI fragility benchmark. Build+preprocess (CPU) run immediately;
# training waits for a free GPU (so it doesn't fight the sweep). Idempotent.
set -uo pipefail
ENVBIN=${ENVBIN:-/home/user/anaconda3/envs/mrifrag/bin}
P=${P:-/media/user/B4864CD4864C98AE/mri_fragility}
export nnUNet_raw=$P/nnUNet_raw nnUNet_preprocessed=$P/nnUNet_preprocessed nnUNet_results=$P/nnUNet_results
D=$nnUNet_raw/Dataset502_AMOSfrag
TR=nnUNetTrainer_250epochs
MDIR=$nnUNet_results/Dataset502_AMOSfrag/${TR}__nnUNetPlans__3d_fullres/fold_0
cd "$P/scripts"

echo "==================== AMOS build (CPU) @ $(date '+%F %T') ===================="
if [ "$(ls "$D/imagesTr"/*.nii.gz 2>/dev/null | wc -l)" -ge 40 ]; then echo "SKIP build"; else
  "$ENVBIN/python" build_amos_dataset.py --raw_out "$nnUNet_raw" --id 502 --R_train 2 --R_test 1 2 4 6 8; fi

echo "==================== AMOS preprocess (CPU) @ $(date '+%F %T') ===================="
if ls "$nnUNet_preprocessed/Dataset502_AMOSfrag/nnUNetPlans_3d_fullres/"*.b2nd >/dev/null 2>&1; then echo "SKIP preprocess"; else
  "$ENVBIN/nnUNetv2_plan_and_preprocess" -d 502 --verify_dataset_integrity -np 4; fi

echo "[$(date '+%F %T')] waiting for a free GPU (sweep/topk to finish)..."
until ! ps -eo cmd | grep -qE "[n]nUNetv2_train|[n]nUNetv2_predict"; do sleep 300; done
export CUDA_VISIBLE_DEVICES=0

echo "==================== AMOS train (GPU0) @ $(date '+%F %T') ===================="
if [ -f "$MDIR/checkpoint_final.pth" ]; then echo "SKIP train"; else
  "$ENVBIN/nnUNetv2_train" 502 3d_fullres 0 -tr "$TR"; fi

echo "==================== AMOS predict @R @ $(date '+%F %T') ===================="
for tag in clean R2 R4 R6 R8; do
  o="$D/predsTs_$tag"
  [ "$(ls "$o"/*.nii.gz 2>/dev/null | wc -l)" -ge 20 ] && { echo "SKIP $tag"; continue; }
  "$ENVBIN/nnUNetv2_predict" -i "$D/imagesTs_$tag" -o "$o" -d 502 -c 3d_fullres -f 0 -tr "$TR" --disable_tta
done

echo "==================== AMOS fragility curve @ $(date '+%F %T') ===================="
"$ENVBIN/python" fragility_eval.py --root "$D" --R 1 2 4 6 8 --labels_module amos_labels --out_prefix amos \
    --kspace_metrics /dev/null
echo "==================== AMOS DONE @ $(date '+%F %T') ===================="
