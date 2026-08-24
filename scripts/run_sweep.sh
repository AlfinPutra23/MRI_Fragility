#!/usr/bin/env bash
# Seed-controlled loss-weighting sweep across BOTH GPUs (2 rounds of 2), then predict + compare.
# GPU1 has the user's small process but ~22GB free; we only use free capacity, never kill anything.
set -uo pipefail
ENVBIN=${ENVBIN:-/home/user/anaconda3/envs/mrifrag/bin}
P=${P:-/media/user/B4864CD4864C98AE/mri_fragility}
VAR=${VAR:-/home/user/anaconda3/envs/mrifrag/lib/python3.11/site-packages/nnunetv2/training/nnUNetTrainer/variants}
export nnUNet_raw=$P/nnUNet_raw nnUNet_preprocessed=$P/nnUNet_preprocessed nnUNet_results=$P/nnUNet_results
D=$nnUNet_raw/Dataset501_MRIfrag
LOGD=$P/outputs/logs
RES=$nnUNet_results/Dataset501_MRIfrag
cp "$P/scripts/nnunet_sweep_trainers.py" "$VAR/"

train(){ # gpu trainer
  local mdir="$RES/${2}__nnUNetPlans__3d_fullres/fold_0"
  if [ -f "$mdir/checkpoint_final.pth" ]; then echo "SKIP train $2 (done)"; return 0; fi
  CUDA_VISIBLE_DEVICES=$1 "$ENVBIN/nnUNetv2_train" 501 3d_fullres 0 -tr "$2" > "$LOGD/sweep_$2.log" 2>&1
}
predict(){ # gpu trainer tag
  local o="$D/predsSW_${2}_$3"
  if [ "$(ls "$o"/*.nii.gz 2>/dev/null | wc -l)" -eq 240 ]; then echo "SKIP predict $2 $3"; return 0; fi
  CUDA_VISIBLE_DEVICES=$1 "$ENVBIN/nnUNetv2_predict" -i "$D/imagesTs_$3" -o "$o" \
      -d 501 -c 3d_fullres -f 0 -tr "$2" --disable_tta >> "$LOGD/sweep_predict.log" 2>&1
}

echo "==================== ROUND 1 train @ $(date '+%F %T') ===================="
train 0 nnUNetTrainer_Uniform_s42 & A=$!
train 1 nnUNetTrainer_FragW4_s42 &  B=$!
wait $A $B
echo "==================== ROUND 2 train @ $(date '+%F %T') ===================="
train 0 nnUNetTrainer_FragW2_s42 &   A=$!
train 1 nnUNetTrainer_FragTopK_s42 & B=$!
wait $A $B

echo "==================== PREDICT (clean,R8) @ $(date '+%F %T') ===================="
for tag in clean R8; do
  predict 0 nnUNetTrainer_Uniform_s42 $tag & predict 1 nnUNetTrainer_FragW4_s42 $tag & wait
  predict 0 nnUNetTrainer_FragW2_s42 $tag &  predict 1 nnUNetTrainer_FragTopK_s42 $tag & wait
done

echo "==================== COMPARE @ $(date '+%F %T') ===================="
cd "$P/scripts"
"$ENVBIN/python" sweep_compare.py --root "$D"
echo "==================== SWEEP DONE @ $(date '+%F %T') ===================="
