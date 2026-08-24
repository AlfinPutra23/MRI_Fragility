#!/bin/bash
# GPU-gated launcher for the FAIR ablation LADDER (two-stage-v2 + FFL + FG-TDR + mixed-R, budget-matched). Waits for a
# GENUINELY idle GPU (<3000 MiB) — never fights user training, and queues behind the boundary run. -> ladder.json
cd /media/user/B4864CD4864C98AE/mri_fragility
LOG=outputs/logs/ladder.log
PY=/home/user/anaconda3/envs/magicnet/bin/python
log(){ echo "[$(date '+%F %T')] $1" | tee -a "$LOG"; }
log "queued: fair ablation ladder (waits for idle GPU; will start after the boundary run frees GPU0)"
while true; do
  [ -f outputs/results/ladder.json ] && { log "ladder.json already exists -> nothing to do"; break; }
  FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' '$2<3000{print $1; exit}')
  if [ -n "$FREE" ]; then
    log "GPU$FREE idle -> launch ladder (CUDA_VISIBLE_DEVICES=$FREE)"
    CUDA_VISIBLE_DEVICES=$FREE PYTHONNOUSERSITE=1 $PY -u scripts/condseg_knee_ladder.py --seeds 0 1 >>"$LOG" 2>&1
    log "=== LADDER DONE (ladder.json) ==="
    break
  fi
  log "no idle GPU (boundary run / user training) — waiting"; sleep 300
done
