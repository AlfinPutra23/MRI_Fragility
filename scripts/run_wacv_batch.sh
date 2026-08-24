#!/usr/bin/env bash
# WACV overnight batch: fires when the B1 multi-seed finishes, then runs (no contention, in order):
#   1. B1 sample (clean) + figure   2. B1 + recon (4 variants) + figure   3. multi-fold benchmark (folds 1-4)
set -uo pipefail
ENVBIN=${ENVBIN:-/home/user/anaconda3/envs/mrifrag/bin}
P=${P:-/media/user/B4864CD4864C98AE/mri_fragility}
export nnUNet_raw=$P/nnUNet_raw nnUNet_preprocessed=$P/nnUNet_preprocessed nnUNet_results=$P/nnUNet_results
export PYTHONUNBUFFERED=1; cd "$P"; PY="$ENVBIN/python"

echo "[$(date '+%F %T')] waiting for B1 multi-seed (b1_joint) to finish..."
until ! ps -eo cmd | grep -q "[b]1_joint.py"; do sleep 200; done
echo "[$(date '+%F %T')] multi-seed done -> WACV batch starts"

echo "==================== 1. B1 SAMPLE @ $(date '+%T') ===================="
CUDA_VISIBLE_DEVICES=0 $PY scripts/b1_joint.py --mask learned --loss fragweighted --R 8 --epochs 60 --tag sample_ours --save_samples > outputs/logs/b1_sample_ours.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY scripts/b1_joint.py --mask vardensity --loss uniform      --R 8 --epochs 60 --tag sample_vd   --save_samples > outputs/logs/b1_sample_vd.log 2>&1 &
wait
$PY scripts/make_b1_sample_figure.py 2>&1 | tail -2

echo "==================== 2. B1 + RECON @ $(date '+%T') ===================="
RSPECS=("random_fixed random uniform" "vd_fixed vardensity uniform" "loupe_uniform learned uniform" "ours learned fragweighted")
i=0
while [ $i -lt ${#RSPECS[@]} ]; do
  for g in 0 1; do
    [ $i -ge ${#RSPECS[@]} ] && break
    read bt mask loss <<< "${RSPECS[$i]}"
    [ -f "outputs/results/b1_${bt}_recon.json" ] || \
      CUDA_VISIBLE_DEVICES=$g $PY scripts/b1_joint.py --mask $mask --loss $loss --recon --R 8 --epochs 60 \
          --tag "${bt}_recon" > "outputs/logs/b1_${bt}_recon.log" 2>&1 &
    i=$((i+1))
  done
  wait
done
$PY scripts/b1_recon_compare.py 2>&1 | tail -8

echo "==================== 3. MULTI-FOLD BENCHMARK @ $(date '+%T') ===================="
bash scripts/run_multifold.sh

echo "==================== WACV BATCH DONE @ $(date '+%F %T') ===================="
