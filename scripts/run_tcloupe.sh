#!/bin/bash
# Target-conditioned LOUPE make-or-break: generic vs conditional, 3 seeds. GPU-gated (<3000 MiB), setsid-detached,
# idempotent. Never fights the user's training or the AMOS run.
cd /media/user/B4864CD4864C98AE/mri_fragility
LOG=outputs/logs/tcloupe.log
PY=/home/user/anaconda3/envs/magicnet/bin/python
[ -f outputs/results/conditional_loupe.json ] && { echo "[$(date '+%F %T')] result exists -> skip" >>"$LOG"; exit 0; }
echo "[$(date '+%F %T')] TC-LOUPE queued, waiting for a genuinely idle GPU" >>"$LOG"
while true; do
  FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' '$2<3000{print $1; exit}')
  [ -n "$FREE" ] && break
  sleep 120
done
echo "[$(date '+%F %T')] TC-LOUPE launch on GPU$FREE (3 seeds x 2 arms x 60 epochs)" >>"$LOG"
CUDA_VISIBLE_DEVICES=$FREE PYTHONNOUSERSITE=1 $PY -u scripts/conditional_loupe.py --seeds 0 1 2 --R 8 --epochs 60 >>"$LOG" 2>&1
echo "[$(date '+%F %T')] TC-LOUPE DONE" >>"$LOG"
