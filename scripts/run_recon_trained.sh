#!/usr/bin/env bash
# #2 DOMAIN-MATCHED RECON baseline (fair W3 opponent). Waits for imagesTr_R8cs (540, from recon_train.py), builds
# Dataset505 (recon images as training input + same labels), preprocesses, trains Uniform_s42 ON the recon distribution,
# predicts Dataset501's R8cs test, then compares recon-trained vs mixed-R (both @ R8cs). GPU-gated (never fights user).
set -uo pipefail
ENVBIN=/home/user/anaconda3/envs/mrifrag/bin
P=/media/user/B4864CD4864C98AE/mri_fragility; cd "$P"
export nnUNet_raw=$P/nnUNet_raw nnUNet_preprocessed=$P/nnUNet_preprocessed nnUNet_results=$P/nnUNet_results
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VAR=/home/user/anaconda3/envs/mrifrag/lib/python3.11/site-packages/nnunetv2/training/nnUNetTrainer/variants
D501=$nnUNet_raw/Dataset501_MRIfrag; D505=$nnUNet_raw/Dataset505_MRIfragReconTr; TR=nnUNetTrainer_Uniform_s42
log=$P/outputs/logs/recon_trained.log; say(){ echo "[$(date '+%F %T')] $*" | tee -a "$log"; }

wait_gpu(){ for t in $(seq 1 600); do for c in 0 1; do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c 2>/dev/null | tr -d ' ')
    busy=$(ps -eo args 2>/dev/null | grep -cE "[n]nUNetv2_train|[n]nUNetv2_predict|b1_join[t]|seg_resnet_fra[g]|knee_sampl[e]")
    [ "${m:-99999}" -lt 3000 ] && [ "$busy" -eq 0 ] && { echo $c; return; }
  done; sleep 120; done; echo -1; }

say "=== recon-trained armed: waiting for imagesTr_R8cs (540) ==="
for i in $(seq 1 900); do n=$(ls $D501/imagesTr_R8cs/*.nii.gz 2>/dev/null | wc -l); [ "$n" -ge 540 ] && { say "train recon complete (540)"; break; }
  [ $((i % 15)) -eq 1 ] && say "train recon $n/540"; sleep 120; done

if [ ! -f "$D505/dataset.json" ]; then
  say "building Dataset505 (recon-trained)"; mkdir -p $D505/imagesTr $D505/labelsTr
  for f in $D501/imagesTr_R8cs/*.nii.gz; do ln -sf "$f" "$D505/imagesTr/$(basename "$f")"; done
  for f in $D501/labelsTr/*.nii.gz;    do ln -sf "$f" "$D505/labelsTr/$(basename "$f")"; done
  cp $D501/dataset.json $D505/dataset.json
fi
if ! ls $nnUNet_preprocessed/Dataset505_MRIfragReconTr/nnUNetPlans_3d_fullres/*.b2nd >/dev/null 2>&1; then
  say "preprocess 505"; $ENVBIN/nnUNetv2_plan_and_preprocess -d 505 --verify_dataset_integrity -np 6 >> $log 2>&1
fi
cp $P/scripts/nnunet_sweep_trainers.py $VAR/ 2>/dev/null

MDIR=$nnUNet_results/Dataset505_MRIfragReconTr/${TR}__nnUNetPlans__3d_fullres/fold_0
if [ ! -f "$MDIR/checkpoint_final.pth" ]; then
  g=$(wait_gpu); say "train 505 (recon-trained) on GPU$g"
  if [ -f "$MDIR/checkpoint_latest.pth" ]; then CUDA_VISIBLE_DEVICES=$g $ENVBIN/nnUNetv2_train 505 3d_fullres 0 -tr $TR --c >> $log 2>&1
  else CUDA_VISIBLE_DEVICES=$g $ENVBIN/nnUNetv2_train 505 3d_fullres 0 -tr $TR >> $log 2>&1; fi
fi
o=$D501/predsRECONTR_R8cs
if [ "$(ls $o/*.nii.gz 2>/dev/null | wc -l)" -ne 240 ]; then
  g=$(wait_gpu); say "predict R8cs with recon-trained model on GPU$g"
  CUDA_VISIBLE_DEVICES=$g $ENVBIN/nnUNetv2_predict -i $D501/imagesTs_R8cs -o $o -d 505 -c 3d_fullres -f 0 -tr $TR --disable_tta >> $log 2>&1
fi
say "compare recon-trained vs mixed-R vs clean (all @ R8cs)"
$ENVBIN/python scripts/recon_trained_compare.py >> $log 2>&1
say "=== recon-trained DONE (recon_trained.json) ==="
