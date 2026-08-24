#!/usr/bin/env bash
# GPU-gated FULL 2nd-architecture run: train the ResU-Net (different family than nnU-Net) on 150 abdominal cases,
# eval per-organ Dice @R{2,4,6,8}, then compare the fragility ordering with nnU-Net. Never fights the user's GPUs:
# runs only when a card is genuinely idle (<3000 MiB AND no train/predict/proxy proc). Detached. (complex_forward is
# CPU-only so it does NOT block this.)
set -uo pipefail
P=/media/user/B4864CD4864C98AE/mri_fragility; cd "$P"
PY=/home/user/anaconda3/envs/magicnet/bin/python
log=$P/outputs/logs/arch2_run.log; say(){ echo "[$(date '+%F %T')] $*" | tee -a "$log"; }
LOCK=$P/outputs/.arch2_launched; [ -f "$LOCK" ] && { say "lock present; exit"; exit 0; }

say "=== arch2 armed: waiting for a genuinely idle GPU, then train ResU-Net (150 cases, 30 ep) ==="
g=-1
for t in $(seq 1 480); do
  for c in 0 1; do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c 2>/dev/null | tr -d ' ')
    busy=$(ps -eo args 2>/dev/null | grep -cE "[n]nUNetv2_train|[n]nUNetv2_predict|b1_join[t]|knee_sampl[e]|seg_resnet_fra[g]")
    [ "${m:-99999}" -lt 3000 ] && [ "$busy" -eq 0 ] && { g=$c; break; }
  done
  [ "$g" -ge 0 ] && break
  [ $((t % 10)) -eq 1 ] && say "no idle GPU yet; waiting"; sleep 120
done
[ "$g" -lt 0 ] && { say "gave up waiting for GPU"; exit 1; }
touch "$LOCK"; say "GPU$g idle -> training ResU-Net (150 cases)"
CUDA_VISIBLE_DEVICES=$g PYTHONNOUSERSITE=1 "$PY" -u scripts/seg_resnet_frag.py --n_train 150 --epochs 30 >> "$log" 2>&1
say "compare ResU-Net vs nnU-Net fragility ordering"
/home/user/anaconda3/bin/python scripts/arch2_compare.py >> "$log" 2>&1
say "=== arch2 DONE (arch2_compare.json) ==="
