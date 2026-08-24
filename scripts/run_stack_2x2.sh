#!/usr/bin/env bash
# STACK 2x2 (audit-corrected): predict the EXISTING mixed-R-trained model (Dataset504 Uniform_s42) on Dataset501's
# R8zf + R8cs test inputs -> completes cells C,D of the training-fix x input-fix 2x2 (A,B already = predsRECON_*).
# INFERENCE ONLY, no training. Single GPU (GPU env). Idempotent (skips finished 240-case pred dirs).
set -uo pipefail
ENVBIN=${ENVBIN:-/home/user/anaconda3/envs/mrifrag/bin}
P=${P:-/media/user/B4864CD4864C98AE/mri_fragility}; GPU=${GPU:-0}
VAR=${VAR:-/home/user/anaconda3/envs/mrifrag/lib/python3.11/site-packages/nnunetv2/training/nnUNetTrainer/variants}
export nnUNet_raw=$P/nnUNet_raw nnUNet_preprocessed=$P/nnUNet_preprocessed nnUNet_results=$P/nnUNet_results
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=$nnUNet_raw/Dataset501_MRIfrag; TR=nnUNetTrainer_Uniform_s42; cd "$P"
cp "$P/scripts/nnunet_sweep_trainers.py" "$VAR/" 2>/dev/null   # ensure the s42 trainer class is importable

for tag in R8zf R8cs; do
  o="$D/predsSTACK_${TR}_$tag"
  if [ "$(ls "$o"/*.nii.gz 2>/dev/null | wc -l)" -ne 240 ]; then
    echo "[$(date '+%F %T')] predict mixedR($TR, D504) on imagesTs_$tag  (GPU$GPU)"
    CUDA_VISIBLE_DEVICES=$GPU "$ENVBIN/nnUNetv2_predict" -i "$D/imagesTs_$tag" -o "$o" \
        -d 504 -c 3d_fullres -f 0 -tr "$TR" --disable_tta >> "$P/outputs/logs/stack_predict.log" 2>&1
  else
    echo "[$(date '+%F %T')] $tag already done (240)"
  fi
done

echo "==================== STACK 2x2 COMPARE @ $(date '+%F %T') ===================="
cd "$P"; "$ENVBIN/python" scripts/stack_compare.py
touch "$D/predsSTACK_${TR}_R8cs/summary.done"
echo "==================== STACK 2x2 DONE @ $(date '+%F %T') ===================="
