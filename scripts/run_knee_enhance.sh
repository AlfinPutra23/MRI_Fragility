#!/usr/bin/env bash
set -uo pipefail
P=/media/user/B4864CD4864C98AE/mri_fragility; cd "$P"
log=$P/outputs/logs/knee_enhance.log; say(){ echo "[$(date '+%F %T')] $*" | tee -a "$log"; }
say "armed: waiting for an idle GPU (<3000 MiB AND no condseg/train proc)"
for t in $(seq 1 300); do
  for c in 0 1; do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c 2>/dev/null|tr -d ' ')
    busy=$(ps -eo args 2>/dev/null|grep -cE "condseg_[kc]|[n]nUNetv2_train|seg_resnet_fra[g]")
    [ "${m:-99999}" -lt 3000 ] && [ "$busy" -eq 0 ] && { say "GPU$c idle -> launch enhancement"
      CUDA_VISIBLE_DEVICES=$c PYTHONNOUSERSITE=1 /home/user/anaconda3/envs/magicnet/bin/python -u scripts/condseg_knee_enhance.py --ntrain 30 --ntest 14 --epochs 45 --seeds 0 1 >> "$log" 2>&1
      say "enhancement DONE"; exit 0; }
  done; sleep 120
done
