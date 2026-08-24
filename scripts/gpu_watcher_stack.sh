#!/usr/bin/env bash
# Launch the STACK 2x2 inference once a GPU is genuinely free (< 6000 MiB). NEVER kills anything, never fights the
# user's or my own running jobs -- it only WAITS then predicts. Idempotent via a lock; polls 270s (stays in cache), 24h.
set -uo pipefail
P=/media/user/B4864CD4864C98AE/mri_fragility; cd "$P"
log="$P/outputs/logs/stack_watcher.log"; LOCK="$P/outputs/.stack_launched"
D=nnUNet_raw/Dataset501_MRIfrag; TR=nnUNetTrainer_Uniform_s42
say(){ echo "[$(date '+%F %T')] $*" >> "$log"; }
say "stack watcher armed: predict mixedR on R8zf+R8cs when a GPU is GENUINELY idle (<3000 MiB AND no train/predict/b1_joint proc) (poll 270s, up to 24h)"
for i in $(seq 1 320); do
  [ -f "$LOCK" ] && { say "lock present; exit"; exit 0; }
  [ -f "$D/predsSTACK_${TR}_R8cs/summary.done" ] && { say "already complete; exit"; exit 0; }
  # wait until NO nnU-Net train/predict and NO b1_joint proxy (mine) are running -> don't fight user or self
  busy=$(ps -eo args 2>/dev/null | grep -cE "[n]nUNetv2_train|[n]nUNetv2_predict|b1_join[t]")
  free_gpu=-1
  if [ "$busy" -eq 0 ]; then
    for g in 0 1; do
      m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g | tr -d ' ')
      [ "${m:-99999}" -lt 3000 ] && { free_gpu=$g; break; }
    done
  fi
  if [ "$free_gpu" -ge 0 ]; then
    touch "$LOCK"; say "GPU$free_gpu free -> launching run_stack_2x2.sh"
    GPU=$free_gpu setsid nohup bash scripts/run_stack_2x2.sh >> outputs/logs/stack_run.log 2>&1 < /dev/null &
    say "launched (pid $!); exit"; exit 0
  fi
  [ $((i % 8)) -eq 1 ] && say "no free GPU yet (GPU0/1 both busy); waiting"
  sleep 270
done
say "gave up after 24h"
