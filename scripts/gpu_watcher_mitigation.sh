#!/usr/bin/env bash
# Idle-GPU watcher: polls until BOTH GPUs are free (user's training done), then auto-launches the mitigation
# ablation. Detached + self-terminating + lock-guarded so it never double-launches or fights the user's GPU work.
set -uo pipefail
P=/media/user/B4864CD4864C98AE/mri_fragility
cd "$P"
log="$P/outputs/logs/mitigation_watcher.log"
LOCK="$P/outputs/.mitigation_launched"
THRESH=4000          # MiB; both GPUs below this = free (idle uses ~0.3 / ~1.2 GB for display)
NEED=2               # consecutive free checks before launching (avoids a transient dip)
say(){ echo "[$(date '+%F %T')] $*" >> "$log"; }

say "watcher armed: launch run_mitigation.sh when both GPUs < ${THRESH} MiB for ${NEED} checks (poll 300s, max 48h)"
free=0
for i in $(seq 1 576); do          # 576 * 300s = 48h
  [ -f "$LOCK" ] && { say "lock present -> already launched; exiting"; exit 0; }
  [ -f outputs/results/b1_mit_all_s2.json ] && { say "mitigation already complete; exiting"; exit 0; }
  m0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
  m1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' ')
  if [ "${m0:-99999}" -lt "$THRESH" ] && [ "${m1:-99999}" -lt "$THRESH" ] && ! pgrep -f "[b]1_joint.py" >/dev/null; then
    free=$((free+1)); say "both GPUs free (${m0}/${m1} MiB)  [${free}/${NEED}]"
    if [ "$free" -ge "$NEED" ]; then
      touch "$LOCK"
      say "LAUNCHING run_mitigation.sh"
      setsid nohup bash scripts/run_mitigation.sh >> outputs/logs/mitigation_run.log 2>&1 < /dev/null &
      say "launched run_mitigation (pid $!); watcher exiting"
      exit 0
    fi
  else
    free=0
    [ $((i % 6)) -eq 1 ] && say "GPUs busy (${m0}/${m1} MiB) — waiting"
  fi
  sleep 300
done
say "gave up after 48h without a free GPU"
