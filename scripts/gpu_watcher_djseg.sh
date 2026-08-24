#!/usr/bin/env bash
# Chain the DJ-Seg HF ablation AFTER the mitigation ablation: wait until mitigation is done (all 12 mit jsons OR
# no b1_joint running) AND both GPUs are free (user not training), then launch run_djseg_hf.sh. Never fights the
# user's GPU. Detached, lock-guarded, self-terminating.
set -uo pipefail
P=/media/user/B4864CD4864C98AE/mri_fragility
cd "$P"
log="$P/outputs/logs/djseg_watcher.log"
LOCK="$P/outputs/.djseg_launched"
THRESH=4000; NEED=2
say(){ echo "[$(date '+%F %T')] $*" >> "$log"; }
mit_done(){ [ "$(ls outputs/results/b1_mit_*.json 2>/dev/null | wc -l)" -ge 12 ]; }

say "DJ-Seg watcher armed: run HF ablation once mitigation done + both GPUs < ${THRESH}MiB (poll 300s, max 24h)"
free=0
for i in $(seq 1 288); do          # 288 * 300s = 24h
  [ -f "$LOCK" ] && { say "lock present -> already launched; exiting"; exit 0; }
  [ -f outputs/results/b1_djhf10_s2.json ] && { say "DJ-HF already complete; exiting"; exit 0; }
  if ! mit_done; then [ $((i % 6)) -eq 1 ] && say "waiting for mitigation to finish ($(ls outputs/results/b1_mit_*.json 2>/dev/null|wc -l)/12 jsons)"; sleep 300; continue; fi
  m0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
  m1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' ')
  if [ "${m0:-99999}" -lt "$THRESH" ] && [ "${m1:-99999}" -lt "$THRESH" ] && ! pgrep -f "[b]1_joint.py" >/dev/null; then
    free=$((free+1)); say "mitigation done + GPUs free (${m0}/${m1} MiB) [${free}/${NEED}]"
    if [ "$free" -ge "$NEED" ]; then
      touch "$LOCK"; say "LAUNCHING run_djseg_hf.sh"
      setsid nohup bash scripts/run_djseg_hf.sh >> outputs/logs/djseg_run.log 2>&1 < /dev/null &
      say "launched (pid $!); watcher exiting"; exit 0
    fi
  else
    free=0; [ $((i % 6)) -eq 1 ] && say "GPUs busy (${m0}/${m1} MiB) — waiting"
  fi
  sleep 300
done
say "gave up after 24h"
