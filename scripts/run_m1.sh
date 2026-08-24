#!/usr/bin/env bash
# M1 de-risk — fire AFTER M0 finishes (needs the trained checkpoint + predsTs_*).
#   (a) metric-blindness: image SSIM vs per-organ Dice dissociation  (CPU, reuses M0 preds)
#   (b) mechanism probe : per-organ seg-loss gradient mass on the M0 net (GPU0, no new training)
set -euo pipefail
ENVBIN=${ENVBIN:-/home/user/anaconda3/envs/mrifrag/bin}
P=${P:-/media/user/B4864CD4864C98AE/mri_fragility}
export nnUNet_raw=$P/nnUNet_raw
export nnUNet_preprocessed=$P/nnUNet_preprocessed
export nnUNet_results=$P/nnUNet_results
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
TR=${TR:-nnUNetTrainer_250epochs}
D=$nnUNet_raw/Dataset501_MRIfrag
cd "$P/scripts"

echo "==================== M1(a) metric-blindness @ $(date '+%F %T') ===================="
"$ENVBIN/python" m1_metric_blindness.py --root "$D" --R 2 4 6 8

echo "==================== M1(b) gradient-mass probe @ $(date '+%F %T') ===================="
"$ENVBIN/python" m1_gradient_probe.py --dataset_id 501 --tr "$TR" --n_cases 8

echo "==================== M1 DONE ===================="
