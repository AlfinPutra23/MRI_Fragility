#!/usr/bin/env bash
set -uo pipefail
P=/media/user/B4864CD4864C98AE/mri_fragility; cd "$P"
log=$P/outputs/logs/fgtdr.log; say(){ echo "[$(date '+%F %T')] $*"|tee -a "$log"; }
say "queued: FG-TDR method experiment — wait for a GENUINELY idle GPU (<3000 MiB), never fight user training"
for t in $(seq 1 720); do
  for c in 0 1; do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c 2>/dev/null|tr -d ' ')
    [ "${m:-99999}" -lt 3000 ] && { say "GPU$c idle -> run FG-TDR (4 arms x 2 seeds)"
      CUDA_VISIBLE_DEVICES=$c PYTHONNOUSERSITE=1 /home/user/anaconda3/envs/magicnet/bin/python -u scripts/condseg_knee_fgtdr.py --seeds 0 1 >> "$log" 2>&1
      say "=== FG-TDR DONE (fgtdr.json) ==="; exit 0; }
  done
  [ $((t%15)) -eq 1 ] && say "GPUs busy (user training) — waiting"; sleep 120
done
