#!/bin/bash
# Single ordered, GPU-gated queue. Priority: (1) FG-Seg make-or-break (the DECISIVE test) then (2) the fair ladder.
# Runs ONE at a time on the first genuinely-idle GPU (<3000 MiB) — never fights user training; each skips if already
# done (so it's blackout-resume-safe alongside per-run checkpointing).
cd /media/user/B4864CD4864C98AE/mri_fragility
LOG=outputs/logs/queue.log
PY=/home/user/anaconda3/envs/magicnet/bin/python
log(){ echo "[$(date '+%F %T')] $1" | tee -a "$LOG"; }

run_one(){  # $1=script  $2=result_json  $3=args  $4=name
  if [ -f "outputs/results/$2" ]; then log "$4: $2 exists -> skip"; return; fi
  log "$4: queued (waiting for idle GPU)"
  while true; do
    # only launch if no OTHER of our jobs is already training (mutual exclusion)
    if ! pgrep -f "condseg_knee_fgtdr.py\|condseg_knee_ladder.py\|fgseg_control.py\|intervention_equalenergy.py" >/dev/null; then
      FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' '$2<3000{print $1; exit}')
      if [ -n "$FREE" ]; then
        log "$4: GPU$FREE idle -> launch"
        CUDA_VISIBLE_DEVICES=$FREE PYTHONNOUSERSITE=1 $PY -u scripts/$1 $3 >>"$LOG" 2>&1
        log "$4: DONE ($2)"; return
      fi
    fi
    sleep 300
  done
}

log "=== queue start ==="
# REORDERED 2026-07-19: the equal-energy intervention is now FIRST. A late audit found per-organ fragility is
# confounded by BASELINE DIFFICULTY (clean-Dice vs drop rho=-0.93, stronger than centroid's +0.86; partial
# centroid|cleanDice n.s.). The intervention is WITHIN-ORGAN, so difficulty is constant by construction — it is now
# the experiment the whole frequency claim depends on. FG-Seg (a likely-null method result) waits; its 6/9 runs
# are checkpointed and it resumes from them.
run_one intervention_equalenergy.py intervention.json  ""               "equal-energy intervention (CAUSAL LINCHPIN)"
run_one fgseg_control.py            fgseg_control.json  "--seeds 0 1 2"  "FG-Seg make-or-break (resumes 6/9)"
# --- WACV 2026 SPRINT (deadline Aug 23 2026, 5 weeks): the 7-arm ladder is CUT. It costs ~18h GPU to sharpen a
# method contribution that is already a documented NEGATIVE (FG-TDR loses to mixed-R). GPU time goes to the LAW
# instead: causal intervention (above) -> corrected forward model -> robustness. Re-enable only if time remains.
# run_one condseg_knee_ladder.py    ladder.json         "--seeds 0 1"    "fair ablation ladder"   # CUT for sprint
log "=== queue complete ==="
