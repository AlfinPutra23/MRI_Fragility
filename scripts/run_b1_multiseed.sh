#!/usr/bin/env bash
# B1 multi-seed: 5 sampling variants x 3 seeds, 2 GPUs in parallel (2 jobs at a time). Idempotent.
set -uo pipefail
ENVBIN=${ENVBIN:-/home/user/anaconda3/envs/mrifrag/bin}
P=${P:-/media/user/B4864CD4864C98AE/mri_fragility}
cd "$P"
export PYTHONUNBUFFERED=1
SPECS=("random_fixed random uniform" "equi_fixed equispaced uniform" "vd_fixed vardensity uniform" \
       "loupe_uniform learned uniform" "ours learned fragweighted")
jobs=(); for s in 0 1 2; do for spec in "${SPECS[@]}"; do jobs+=("$spec $s"); done; done

i=0
while [ $i -lt ${#jobs[@]} ]; do
  pids=()
  for g in 0 1; do
    [ $i -ge ${#jobs[@]} ] && break
    read bt mask loss seed <<< "${jobs[$i]}"; tag="${bt}_s${seed}"
    if [ ! -f "outputs/results/b1_${tag}.json" ]; then
      echo "[$(date '+%T')] GPU$g: $tag"
      CUDA_VISIBLE_DEVICES=$g "$ENVBIN/python" scripts/b1_joint.py --mask $mask --loss $loss --R 8 \
          --epochs 60 --tag "$tag" --seed $seed > "outputs/logs/b1_${tag}.log" 2>&1 &
      pids+=($!)
    fi
    i=$((i+1))
  done
  [ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
done

echo "==================== B1 MULTISEED COMPARE @ $(date '+%F %T') ===================="
"$ENVBIN/python" scripts/b1_multiseed_compare.py
echo "==================== B1 MULTISEED DONE @ $(date '+%F %T') ===================="
