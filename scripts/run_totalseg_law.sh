#!/bin/bash
# Full TotalSeg-MRI law run: 25 subjects, R{1,8}, GPU-gated, idempotent, detached.
cd /media/user/B4864CD4864C98AE/mri_fragility
LOG=outputs/logs/totalseg_law.log
[ -f outputs/results/totalseg_law.json ] && { echo "[$(date '+%F %T')] result exists -> skip" >>"$LOG"; exit 0; }
echo "[$(date '+%F %T')] TotalSeg law queued, waiting for idle GPU" >>"$LOG"
while true; do
  FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' '$2<3000{print $1; exit}')
  [ -n "$FREE" ] && break; sleep 120
done
echo "[$(date '+%F %T')] launch on GPU$FREE (25 subjects, R 1 8, total_mr fast)" >>"$LOG"
CUDA_VISIBLE_DEVICES=$FREE /home/user/anaconda3/envs/totalseg/bin/python scripts/totalseg_law.py \
  --n 25 --R 1 8 --device gpu --fast 1 --min_n 3 >>"$LOG" 2>&1
echo "[$(date '+%F %T')] TotalSeg law DONE" >>"$LOG"
