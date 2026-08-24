#!/usr/bin/env bash
set -uo pipefail
P=/media/user/B4864CD4864C98AE/mri_fragility; cd "$P"
log=$P/outputs/logs/final_polish.log; say(){ echo "[$(date '+%F %T')] $*"|tee -a "$log"; }
say "queued: wait for idle GPU -> (a) condseg_knee_full (3 seeds+Wilcoxon+recon baseline) -> (b) money_figure"
for t in $(seq 1 400); do
  free=-1; for c in 0 1; do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c 2>/dev/null|tr -d ' ')
    busy=$(ps -eo args 2>/dev/null|grep -cE "condseg_[kce]|[n]nUNetv2_train|seg_resnet_fra[g]")
    [ "${m:-99999}" -lt 3000 ] && [ "$busy" -eq 0 ] && { free=$c; break; }
  done
  [ "$free" -ge 0 ] && break; sleep 120
done
say "(a) full real-k-space method on GPU$free"
CUDA_VISIBLE_DEVICES=$free PYTHONNOUSERSITE=1 /home/user/anaconda3/envs/magicnet/bin/python -u scripts/condseg_knee_full.py --seeds 0 1 2 >> "$log" 2>&1
say "(b) money figure"; /home/user/anaconda3/bin/python scripts/money_figure.py >> "$log" 2>&1
say "=== FINAL POLISH DONE ==="
