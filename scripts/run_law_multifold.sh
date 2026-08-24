#!/usr/bin/env bash
# Queued follow-up: when the clean multifold finishes, compute the spectral-centroid law + R* PER FOLD and
# aggregate to mean+-std (error bars on law v2). CPU-only. Single detached watcher, self-terminates. Launch detached.
set -uo pipefail
ENVBIN=${ENVBIN:-/home/user/anaconda3/envs/mrifrag/bin}
P=${P:-/media/user/B4864CD4864C98AE/mri_fragility}
cd "$P"; export PYTHONUNBUFFERED=1; PY="$ENVBIN/python"; D="$P/nnUNet_raw/Dataset501_MRIfrag"

echo "[$(date '+%F %T')] waiting for multifold DONE marker..."
i=0
until grep -q "MULTIFOLD DONE" outputs/logs/multifold.log 2>/dev/null; do
  sleep 300; i=$((i+1)); [ $i -ge 200 ] && { echo "[$(date '+%T')] gave up after ~16h"; exit 1; }
done
echo "[$(date '+%F %T')] multifold done -> per-fold law_v2 + R* (fold0=m0)"

for pf in m0 fold1 fold2 fold3 fold4; do
  [ -f "outputs/results/${pf}_fragility_dice.json" ] || { echo "SKIP $pf (no fragility json)"; continue; }
  "$PY" scripts/law_v2.py --root "$D" --out_prefix "$pf" --max_cases 40 2>&1 | grep -E "centroid |sav " | sed "s/^/[$pf] /"
  "$PY" scripts/predict_rstar.py --prefix "$pf" 2>&1 | grep -E "Spearman" | sed "s/^/[$pf] /"
done

echo "==================== LAW-MULTIFOLD AGGREGATE ===================="
"$PY" scripts/law_multifold_aggregate.py
echo "[$(date '+%F %T')] LAW MULTIFOLD DONE"
