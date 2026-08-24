#!/usr/bin/env bash
# Launch the Focal-Tversky nnU-Net training on a FREE GPU once the 2-D proxy is done. Never fights the user's
# training (waits for a genuinely idle card). Detached, lock-guarded, self-terminating.
set -uo pipefail
P=/media/user/B4864CD4864C98AE/mri_fragility; cd "$P"
log="$P/outputs/logs/ft_watcher.log"; LOCK="$P/outputs/.ft_launched"
DONE="$P/nnUNet_results/Dataset501_MRIfrag/nnUNetTrainer_FocalTversky_s42__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"
say(){ echo "[$(date '+%F %T')] $*" >> "$log"; }
say "FT watcher armed: train Focal-Tversky when proxy done + a GPU < 4000 MiB (poll 300s, max 24h)"
for i in $(seq 1 288); do
  [ -f "$LOCK" ] && { say "lock present -> already launched; exit"; exit 0; }
  [ -f "$DONE" ] && { say "already trained; exit"; exit 0; }
  if pgrep -f "[b]1_joint.py" >/dev/null; then [ $((i % 6)) -eq 1 ] && say "proxy still running; waiting"; sleep 300; continue; fi
  free_gpu=-1
  for g in 0 1; do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g | tr -d ' ')
    [ "${m:-99999}" -lt 4000 ] && { free_gpu=$g; break; }
  done
  if [ "$free_gpu" -ge 0 ]; then
    touch "$LOCK"; say "GPU$free_gpu free -> LAUNCHING run_focaltversky.sh"
    GPU=$free_gpu setsid nohup bash scripts/run_focaltversky.sh >> outputs/logs/ft_run.log 2>&1 < /dev/null &
    say "launched (pid $!); exit"; exit 0
  fi
  [ $((i % 6)) -eq 1 ] && say "no free GPU yet; waiting"
  sleep 300
done
say "gave up after 24h"
