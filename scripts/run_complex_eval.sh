#!/usr/bin/env bash
# GPU-gated eval of the COMPLEX MULTICOIL forward model (#1). Waits for complex_forward.py to produce all 960 images,
# then segments imagesTs_R{2,4,6,8}_cx with the EXISTING Uniform_s42 nnU-Net (inference only), runs fragility_eval +
# complex_compare -> does per-organ fragility ordering + centroid law survive a realistic A=M·F·S? Never fights the
# user's GPUs (predicts only when a card is genuinely idle: <3000 MiB AND no train/proxy proc). Detached, disk light.
set -uo pipefail
ENVBIN=/home/user/anaconda3/envs/mrifrag/bin
P=/media/user/B4864CD4864C98AE/mri_fragility; cd "$P"
export nnUNet_raw=$P/nnUNet_raw nnUNet_preprocessed=$P/nnUNet_preprocessed nnUNet_results=$P/nnUNet_results
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=$nnUNet_raw/Dataset501_MRIfrag; TR=nnUNetTrainer_Uniform_s42
log=$P/outputs/logs/complex_eval.log; say(){ echo "[$(date '+%F %T')] $*" | tee -a "$log"; }

wait_free_gpu(){ for t in $(seq 1 360); do for c in 0 1; do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c 2>/dev/null | tr -d ' ')
    busy=$(ps -eo args 2>/dev/null | grep -cE "[n]nUNetv2_train|[n]nUNetv2_predict|b1_join[t]|knee_sampl[e]")
    [ "${m:-99999}" -lt 3000 ] && [ "$busy" -eq 0 ] && { echo $c; return; }
  done; sleep 120; done; echo -1; }

say "=== complex-eval armed: wait for 960 cx images, then GPU-gated inference ==="
for i in $(seq 1 400); do
  n=$(ls $D/imagesTs_R2_cx/*.nii.gz $D/imagesTs_R4_cx/*.nii.gz $D/imagesTs_R6_cx/*.nii.gz $D/imagesTs_R8_cx/*.nii.gz 2>/dev/null | wc -l)
  [ "$n" -ge 960 ] && { say "all 960 cx images ready"; break; }
  [ $((i % 6)) -eq 1 ] && say "waiting for complex_forward ($n/960)"; sleep 120
done
[ -e "$D/predsCX_clean" ] || ln -s "$D/predsTs_clean" "$D/predsCX_clean"   # clean pred is forward-model-independent

for R in 2 4 6 8; do
  o=$D/predsCX_R$R
  [ "$(ls $o/*.nii.gz 2>/dev/null | wc -l)" -eq 240 ] && { say "R$R already done"; continue; }
  g=$(wait_free_gpu); say "predict R$R cx on GPU$g"
  CUDA_VISIBLE_DEVICES=$g "$ENVBIN/nnUNetv2_predict" -i "$D/imagesTs_R${R}_cx" -o "$o" -d 501 -c 3d_fullres -f 0 -tr "$TR" --disable_tta >> "$log" 2>&1
done

say "fragility_eval (complex forward)"
"$ENVBIN/python" scripts/fragility_eval.py --root "$D" --preds_tpl "predsCX_{tag}" --R 1 2 4 6 8 --out_prefix cx >> "$log" 2>&1
say "complex_compare"
"$ENVBIN/python" scripts/complex_compare.py >> "$log" 2>&1
say "=== COMPLEX EVAL DONE (see complex_compare.json / complex_forward.png) ==="
