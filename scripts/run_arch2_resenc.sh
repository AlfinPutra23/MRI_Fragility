#!/usr/bin/env bash
# CLEAN 3D 2nd-architecture: nnU-Net ResEnc (ResidualEncoderUNet -- genuinely different network, SAME 3D pipeline/eval)
# with the seed-matched Uniform_s42 trainer. Reuses the existing 3d_fullres preprocessed data (ResEnc data_identifier =
# nnUNetPlans_3d_fullres). GPU-gated train -> predict imagesTs_{clean,R2,R4,R6,R8} -> fragility_eval -> compare fragility
# ordering with the plain nnU-Net. Never fights the user's GPUs.
set -uo pipefail
ENVBIN=/home/user/anaconda3/envs/mrifrag/bin
P=/media/user/B4864CD4864C98AE/mri_fragility; cd "$P"
export nnUNet_raw=$P/nnUNet_raw nnUNet_preprocessed=$P/nnUNet_preprocessed nnUNet_results=$P/nnUNet_results
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VAR=/home/user/anaconda3/envs/mrifrag/lib/python3.11/site-packages/nnunetv2/training/nnUNetTrainer/variants
D=$nnUNet_raw/Dataset501_MRIfrag; TR=nnUNetTrainer_Uniform_s42; PLANS=nnUNetResEncUNetMPlans
log=$P/outputs/logs/arch2_resenc.log; say(){ echo "[$(date '+%F %T')] $*" | tee -a "$log"; }
LOCK=$P/outputs/.resenc_launched; [ -f "$LOCK" ] && { say "lock present; exit"; exit 0; }; touch "$LOCK"

wait_gpu(){ for t in $(seq 1 900); do for c in 0 1; do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c 2>/dev/null | tr -d ' ')
    busy=$(ps -eo args 2>/dev/null | grep -cE "[n]nUNetv2_train|[n]nUNetv2_predict|b1_join[t]|seg_resnet_fra[g]|knee_sampl[e]")
    [ "${m:-99999}" -lt 3000 ] && [ "$busy" -eq 0 ] && { echo $c; return; }
  done; sleep 120; done; echo -1; }

cp $P/scripts/nnunet_sweep_trainers.py $VAR/ 2>/dev/null
MDIR=$nnUNet_results/Dataset501_MRIfrag/${TR}__${PLANS}__3d_fullres/fold_0
if [ ! -f "$MDIR/checkpoint_final.pth" ]; then
  g=$(wait_gpu); say "train ResEnc (ResidualEncoderUNet, Uniform_s42) on GPU$g"
  if [ -f "$MDIR/checkpoint_latest.pth" ]; then CUDA_VISIBLE_DEVICES=$g $ENVBIN/nnUNetv2_train 501 3d_fullres 0 -p $PLANS -tr $TR --c >> $log 2>&1
  else CUDA_VISIBLE_DEVICES=$g $ENVBIN/nnUNetv2_train 501 3d_fullres 0 -p $PLANS -tr $TR >> $log 2>&1; fi
fi
for tag in clean R2 R4 R6 R8; do
  o=$D/predsRESENC_$tag
  [ "$(ls $o/*.nii.gz 2>/dev/null | wc -l)" -eq 240 ] && { say "$tag done"; continue; }
  g=$(wait_gpu); say "predict $tag with ResEnc on GPU$g"
  CUDA_VISIBLE_DEVICES=$g $ENVBIN/nnUNetv2_predict -i $D/imagesTs_$tag -o $o -d 501 -c 3d_fullres -f 0 -tr $TR -p $PLANS --disable_tta >> $log 2>&1
done
say "fragility_eval (ResEnc)"
$ENVBIN/python scripts/fragility_eval.py --root $D --preds_tpl "predsRESENC_{tag}" --R 1 2 4 6 8 --out_prefix resenc >> $log 2>&1
say "compare ResEnc vs nnU-Net fragility ordering"
$ENVBIN/python scripts/arch2_resenc_compare.py >> $log 2>&1
say "=== ResEnc 3D 2nd-arch DONE (arch2_resenc_compare.json) ==="
