#!/bin/bash
cd /media/user/B4864CD4864C98AE/mri_fragility
LOG=outputs/logs/queue.log
PY=/home/user/anaconda3/envs/magicnet/bin/python
[ -f outputs/results/intervention_v3.json ] && { echo "[$(date '+%F %T')] v3 exists -> skip" >> "$LOG"; exit 0; }
while true; do
  FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' '$2<3000{print $1; exit}')
  [ -n "$FREE" ] && break
  sleep 120
done
echo "[$(date '+%F %T')] INTERVENTION v3 (width-matched, full deletion): GPU$FREE -> launch" >> "$LOG"
CUDA_VISIBLE_DEVICES=$FREE PYTHONNOUSERSITE=1 $PY -u scripts/intervention_v3.py --E_pct 1.5 >> "$LOG" 2>&1
echo "[$(date '+%F %T')] INTERVENTION v3 DONE" >> "$LOG"
