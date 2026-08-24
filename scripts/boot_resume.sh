#!/bin/bash
# Auto-resume the FG-TDR boundary ablation after a reboot (e.g. power outage). Guarded so it does NOT re-run a finished
# experiment. Installed as an @reboot cron entry. The underlying run checkpoints per-seed and resumes from the last seed.
for i in $(seq 1 30); do [ -d /media/user/B4864CD4864C98AE/mri_fragility ] && break; sleep 10; done   # wait for external disk to mount
cd /media/user/B4864CD4864C98AE/mri_fragility || exit 0
# resume whichever run is unfinished (boundary first, then the fair ladder — both checkpoint per-seed)
if [ ! -f outputs/results/fgtdr_bnd.json ]; then
  echo "[$(date '+%F %T')] @reboot -> resuming FG-TDR boundary ablation" >> outputs/logs/boot_resume.log
  exec bash scripts/run_fgtdr_bnd.sh
elif [ ! -f outputs/results/fgseg_control.json ] || [ ! -f outputs/results/intervention.json ] || [ ! -f outputs/results/ladder.json ]; then
  echo "[$(date '+%F %T')] @reboot -> resuming queue (FG-Seg make-or-break + equal-energy intervention + ladder)" >> outputs/logs/boot_resume.log
  exec bash scripts/run_queue.sh
fi
exit 0                                                     # all done -> nothing to resume
