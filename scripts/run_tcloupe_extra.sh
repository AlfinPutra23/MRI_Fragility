#!/bin/bash
cd /media/user/B4864CD4864C98AE/mri_fragility
LOG=outputs/logs/tcloupe_extra.log
PY=/home/user/anaconda3/envs/magicnet/bin/python
[ -f outputs/results/conditional_loupe_s3456.json ] && { echo "[$(date '+%F %T')] extra seeds exist -> skip to merge" >>"$LOG"; }
if [ ! -f outputs/results/conditional_loupe_s3456.json ]; then
  echo "[$(date '+%F %T')] extra TC-LOUPE seeds 3-6 queued, waiting for idle GPU" >>"$LOG"
  while true; do
    FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' '$2<3000{print $1; exit}')
    [ -n "$FREE" ] && break; sleep 120
  done
  echo "[$(date '+%F %T')] launch seeds 3-6 on GPU$FREE" >>"$LOG"
  CUDA_VISIBLE_DEVICES=$FREE PYTHONNOUSERSITE=1 $PY -u scripts/conditional_loupe.py --seeds 3 4 5 6 --R 8 --epochs 60 \
    --out outputs/results/conditional_loupe_s3456.json >>"$LOG" 2>&1
fi
echo "[$(date '+%F %T')] MERGE all 7 seeds" >>"$LOG"
$PY scripts/merge_loupe.py outputs/results/conditional_loupe_s012.json outputs/results/conditional_loupe_s3456.json >>"$LOG" 2>&1
echo "[$(date '+%F %T')] TC-LOUPE EXTRA DONE" >>"$LOG"
