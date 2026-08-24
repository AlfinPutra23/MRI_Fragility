#!/bin/bash
# USER-AUTHORIZED immediate run ("run it now"): bypass the idle-GPU gate for the make-or-break only, on the GPU passed
# in as $1 (default 0), then hand control back to the NORMAL gated queue for the intervention + ladder.
# Never kills anything; the user's training on the other GPU is untouched.
cd /media/user/B4864CD4864C98AE/mri_fragility
LOG=outputs/logs/queue.log
PY=/home/user/anaconda3/envs/magicnet/bin/python
G=${1:-0}
echo "[$(date '+%F %T')] RUN-NOW: FG-Seg make-or-break on GPU$G (user-authorized, idle-gate bypassed; resumes from checkpoint)" >> "$LOG"
CUDA_VISIBLE_DEVICES=$G PYTHONNOUSERSITE=1 $PY -u scripts/fgseg_control.py --seeds 0 1 2 >> "$LOG" 2>&1
echo "[$(date '+%F %T')] RUN-NOW: make-or-break exited -> handing back to the gated queue (intervention + ladder)" >> "$LOG"
exec bash scripts/run_queue.sh
