#!/bin/bash
# GPU-gated launcher for the FG-TDR BOUNDARY-prior ablation. Waits for a GENUINELY idle GPU (<3000 MiB) and
# never fights the user's training. -> outputs/results/fgtdr_bnd.json
cd /media/user/B4864CD4864C98AE/mri_fragility
LOG=outputs/logs/fgtdr_bnd.log
PY=/home/user/anaconda3/envs/magicnet/bin/python
log(){ echo "[$(date '+%F %T')] $1" | tee -a "$LOG"; }
log "queued: FG-TDR boundary-W ablation (region vs boundary vs mixed-R, 2 seeds)"
while true; do
  FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' '$2<3000{print $1; exit}')
  if [ -n "$FREE" ]; then
    log "GPU$FREE idle -> launch (CUDA_VISIBLE_DEVICES=$FREE)"
    CUDA_VISIBLE_DEVICES=$FREE PYTHONNOUSERSITE=1 $PY -u scripts/condseg_knee_fgtdr.py --seeds 0 1 >>"$LOG" 2>&1
    log "=== FG-TDR boundary-W DONE (fgtdr_bnd.json) ==="
    break
  fi
  log "GPUs busy (user training) — waiting"; sleep 300
done
