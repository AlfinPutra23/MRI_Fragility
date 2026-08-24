#!/usr/bin/env bash
# Run the recon-then-segment predict+compare once CS recon is done AND a GPU is free (never fights user/other jobs).
set -uo pipefail
P=/media/user/B4864CD4864C98AE/mri_fragility; cd "$P"
log="$P/outputs/logs/recon_watcher.log"; LOCK="$P/outputs/.recon_launched"
D=nnUNet_raw/Dataset501_MRIfrag
say(){ echo "[$(date '+%F %T')] $*" >> "$log"; }
say "recon watcher armed: run predict+compare when imagesTs_R8cs=240 AND a GPU < 6000 MiB (poll 300s, 24h)"
for i in $(seq 1 288); do
  [ -f "$LOCK" ] && { say "lock present; exit"; exit 0; }
  [ -f "$D/predsRECON_R8cs/summary.done" ] && exit 0
  n=$(ls "$D/imagesTs_R8cs"/*.nii.gz 2>/dev/null | wc -l)
  if [ "$n" -lt 240 ]; then [ $((i%6)) -eq 1 ] && say "CS recon in progress ($n/240)"; sleep 300; continue; fi
  free_gpu=-1
  for g in 0 1; do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g|tr -d ' '); [ "${m:-99999}" -lt 6000 ] && { free_gpu=$g; break; }; done
  if [ "$free_gpu" -ge 0 ]; then
    touch "$LOCK"; say "GPU$free_gpu free + recon ready -> launching run_recon_baseline.sh"
    GPU=$free_gpu setsid nohup bash scripts/run_recon_baseline.sh >> outputs/logs/recon_run.log 2>&1 < /dev/null &
    say "launched (pid $!); exit"; exit 0
  fi
  [ $((i%6)) -eq 1 ] && say "recon ready ($n/240) but no free GPU; waiting"
  sleep 300
done
say "gave up after 24h"
