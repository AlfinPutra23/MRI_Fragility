#!/usr/bin/env bash
set -uo pipefail
P=/media/user/B4864CD4864C98AE/mri_fragility; cd "$P"
log=$P/outputs/logs/knee_recon.log; say(){ echo "[$(date '+%F %T')] $*"|tee -a "$log"; }
say "queued: wait for a GENUINELY idle GPU (<3000 MiB) before running learned-recon baseline (never fight user training)"
for t in $(seq 1 720); do   # up to 24h
  for c in 0 1; do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c 2>/dev/null|tr -d ' ')
    [ "${m:-99999}" -lt 3000 ] && { say "GPU$c idle -> run learned-recon baseline"
      CUDA_VISIBLE_DEVICES=$c PYTHONNOUSERSITE=1 /home/user/anaconda3/envs/magicnet/bin/python -u scripts/condseg_knee_recon.py --epochs 35 --recon_epochs 30 --seeds 0 1 >> "$log" 2>&1
      /home/user/anaconda3/bin/python scripts/money_figure.py >> "$log" 2>&1
      say "=== learned-recon baseline DONE ==="; exit 0; }
  done
  [ $((t%15)) -eq 1 ] && say "GPUs busy (user training) — waiting"; sleep 120
done
