#!/bin/bash
# TotalSeg-MRI law at FULL RESOLUTION (no --fast) — fairer number vs the nnU-Net abdominal law. GPU-gated: smoke->full.
cd /media/user/B4864CD4864C98AE/mri_fragility
LOG=outputs/logs/totalseg_law_fullres.log
PY=/home/user/anaconda3/envs/totalseg/bin/python
[ -f outputs/results/totalseg_law_fullres.json ] && { echo "[$(date '+%F %T')] exists->skip" >>"$LOG"; exit 0; }
echo "[$(date '+%F %T')] fullres queued, waiting for idle GPU" >>"$LOG"
while true; do FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits|awk -F', *' '$2<3000{print $1;exit}'); [ -n "$FREE" ]&&break; sleep 120; done
echo "[$(date '+%F %T')] GPU$FREE: full-res 1-subject SMOKE (pulls full-res weights)" >>"$LOG"
CUDA_VISIBLE_DEVICES=$FREE $PY scripts/totalseg_law.py --smoke 1 --R 1 8 --device gpu --fast 0 --min_n 1 --out outputs/results/totalseg_law_fullres_smoke.json >>"$LOG" 2>&1
ok=$(/home/user/anaconda3/bin/python -c "
import json
try:
  d=json.load(open('outputs/results/totalseg_law_fullres_smoke.json')); r=d['rows']; k=[c for c in r[0] if c.startswith('dice_R1')][0]
  print('OK' if len(r)>=10 and sum(1 for x in r if x[k]>0.3)>=5 else 'FAIL')
except Exception as e: print('FAIL',e)" 2>&1)
echo "[$(date '+%F %T')] fullres smoke verify: $ok" >>"$LOG"
[ "${ok:0:2}" != "OK" ] && { echo "[$(date '+%F %T')] FULLRES SMOKE FAILED" >>"$LOG"; exit 0; }
echo "[$(date '+%F %T')] GPU$FREE: FULL-RES 25-subject run (~5x slower)" >>"$LOG"
CUDA_VISIBLE_DEVICES=$FREE $PY scripts/totalseg_law.py --n 25 --R 1 8 --device gpu --fast 0 --min_n 3 --out outputs/results/totalseg_law_fullres.json >>"$LOG" 2>&1
echo "[$(date '+%F %T')] fullres DONE" >>"$LOG"
