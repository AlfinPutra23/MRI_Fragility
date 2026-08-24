#!/bin/bash
# Wait for the running FG-Seg make-or-break to finish, then run the FIXED equal-energy intervention.
# (v1 of the intervention was invalidated by a no-op energy budget; see intervention_INVALID_v1.json.)
cd /media/user/B4864CD4864C98AE/mri_fragility
LOG=outputs/logs/queue.log
PY=/home/user/anaconda3/envs/magicnet/bin/python
while pgrep -f "[f]gseg_control.py" >/dev/null; do sleep 120; done
echo "[$(date '+%F %T')] CHAIN: FG-Seg finished -> launching FIXED intervention (v2, matched budget 1.49% of image energy)" >> "$LOG"
# wait for a genuinely idle GPU so we never fight the user's training
while true; do
  FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' '$2<3000{print $1; exit}')
  [ -n "$FREE" ] && break
  sleep 120
done
echo "[$(date '+%F %T')] CHAIN: GPU$FREE idle -> intervention v2 launch" >> "$LOG"
CUDA_VISIBLE_DEVICES=$FREE PYTHONNOUSERSITE=1 $PY -u scripts/intervention_equalenergy.py >> "$LOG" 2>&1
echo "[$(date '+%F %T')] CHAIN: intervention v2 DONE" >> "$LOG"
