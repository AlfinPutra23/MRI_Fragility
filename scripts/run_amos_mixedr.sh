#!/usr/bin/env bash
# AMOS mixed-R: build (CPU) -> preprocess (CPU) -> wait for genuinely-idle GPU -> train 250epochs on mixed-R (506)
# -> predict on Dataset502 test @R8+clean -> compare to the R2-trained baseline. Idempotent, GPU-gated, never fights
# the user's training. Detached via setsid so it survives session teardown.
set -uo pipefail
ENVBIN=/home/user/anaconda3/envs/mrifrag/bin
P=/media/user/B4864CD4864C98AE/mri_fragility
export nnUNet_raw=$P/nnUNet_raw nnUNet_preprocessed=$P/nnUNet_preprocessed nnUNet_results=$P/nnUNet_results
D502=$nnUNet_raw/Dataset502_AMOSfrag
D506=$nnUNet_raw/Dataset506_AMOSfragMixedR
TR=nnUNetTrainer_250epochs
MDIR=$nnUNet_results/Dataset506_AMOSfragMixedR/${TR}__nnUNetPlans__3d_fullres/fold_0
LOG=$P/outputs/logs/amos_mixedr.log
cd "$P/scripts"

echo "[$(date '+%F %T')] ===== BUILD AMOS mixed-R (CPU) =====" | tee -a "$LOG"
if [ "$(ls "$D506/imagesTr"/*.nii.gz 2>/dev/null | wc -l)" -ge 40 ]; then echo "  skip build" | tee -a "$LOG"; else
  "$ENVBIN/python" build_amos_mixedr.py --raw_out "$nnUNet_raw" --id 506 >>"$LOG" 2>&1; fi

echo "[$(date '+%F %T')] ===== PREPROCESS 506 (CPU) =====" | tee -a "$LOG"
if ls "$nnUNet_preprocessed/Dataset506_AMOSfragMixedR/nnUNetPlans_3d_fullres/"*.b2nd >/dev/null 2>&1; then echo "  skip preprocess" | tee -a "$LOG"; else
  "$ENVBIN/nnUNetv2_plan_and_preprocess" -d 506 --verify_dataset_integrity -np 4 >>"$LOG" 2>&1; fi

echo "[$(date '+%F %T')] ===== waiting for a genuinely idle GPU (<3000 MiB) =====" | tee -a "$LOG"
while true; do
  FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' '$2<3000{print $1; exit}')
  [ -n "$FREE" ] && break
  sleep 120
done
export CUDA_VISIBLE_DEVICES=$FREE
echo "[$(date '+%F %T')] ===== TRAIN 506 on GPU$FREE =====" | tee -a "$LOG"
if [ -f "$MDIR/checkpoint_final.pth" ]; then echo "  skip train" | tee -a "$LOG"; else
  if [ -f "$MDIR/checkpoint_latest.pth" ]; then "$ENVBIN/nnUNetv2_train" 506 3d_fullres 0 -tr "$TR" --c >>"$LOG" 2>&1
  else "$ENVBIN/nnUNetv2_train" 506 3d_fullres 0 -tr "$TR" >>"$LOG" 2>&1; fi
fi

echo "[$(date '+%F %T')] ===== PREDICT 506 on AMOS test (R8 + clean) =====" | tee -a "$LOG"
for tag in clean R8; do
  o="$D502/predsMIXEDR_$tag"
  [ "$(ls "$o"/*.nii.gz 2>/dev/null | wc -l)" -ge 20 ] && { echo "  skip predict $tag" | tee -a "$LOG"; continue; }
  CUDA_VISIBLE_DEVICES=$FREE "$ENVBIN/nnUNetv2_predict" -i "$D502/imagesTs_$tag" -o "$o" -d 506 -c 3d_fullres -f 0 -tr "$TR" --disable_tta >>"$LOG" 2>&1
done

echo "[$(date '+%F %T')] ===== COMPARE =====" | tee -a "$LOG"
cd "$P" && "$ENVBIN/python" scripts/amos_mixedr_compare.py >>"$LOG" 2>&1
echo "[$(date '+%F %T')] ===== AMOS MIXED-R DONE =====" | tee -a "$LOG"
