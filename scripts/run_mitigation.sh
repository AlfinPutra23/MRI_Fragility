#!/usr/bin/env bash
# MITIGATION ablation: can we recover the fragile organs at R8? Fixed variable-density acquisition, test @R8;
# vary only the training recipe. baseline / +mixed-R / +Focal-Tversky / +both, x3 seeds, 2 GPUs. Idempotent.
set -uo pipefail
ENVBIN=${ENVBIN:-/home/user/anaconda3/envs/mrifrag/bin}
P=${P:-/media/user/B4864CD4864C98AE/mri_fragility}
cd "$P"; export PYTHONUNBUFFERED=1; PY="$ENVBIN/python"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation OOM (added after user-training resume incident)

jobs=()
for s in 0 1 2; do
  jobs+=("baseline|--loss uniform|$s")                              # fixed R8, Dice-CE (the 'broken' model)
  jobs+=("mixedr|--loss uniform --mixed_r|$s")                      # + mixed-R augmentation
  jobs+=("tversky|--loss fragweighted --tversky|$s")               # + Focal-Tversky recall loss (tail-weighted)
  jobs+=("all|--loss fragweighted --mixed_r --tversky|$s")         # MITIGATION = all three (+ oversampling, on by default)
done

i=0
while [ $i -lt ${#jobs[@]} ]; do
  pids=()
  for g in 0 1; do
    [ $i -ge ${#jobs[@]} ] && break
    IFS='|' read name flags seed <<< "${jobs[$i]}"; tag="mit_${name}_s${seed}"
    if [ ! -f "outputs/results/b1_${tag}.json" ]; then
      echo "[$(date '+%T')] GPU$g: $tag  ($flags)"
      CUDA_VISIBLE_DEVICES=$g $PY scripts/b1_joint.py --mask vardensity --R 8 --epochs 60 $flags \
          --tag "$tag" --seed "$seed" > "outputs/logs/b1_${tag}.log" 2>&1 &
      pids+=($!)
    fi
    i=$((i+1))
  done
  [ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
done

echo "==================== MITIGATION RESULTS @ $(date '+%F %T') ===================="
"$PY" - <<'PY'
import json, glob, os, collections, statistics as st
TAIL = ["gallbladder", "esophagus", "pancreas", "adrenal_R", "adrenal_L", "duodenum"]
def tm(po, ks): v = [po[o] for o in ks if o in po]; return sum(v)/len(v) if v else float('nan')
by = collections.defaultdict(lambda: collections.defaultdict(list))
for f in glob.glob("outputs/results/b1_mit_*.json"):
    d = json.load(open(f)); po = d.get("per_organ", {})
    k = os.path.basename(f)[7:].rsplit("_s", 1)[0]                  # strip 'b1_mit_' and '_s<seed>.json'
    by[k]["tail"].append(tm(po, TAIL)); by[k]["adr"].append(tm(po, ["adrenal_R", "adrenal_L"]))
    by[k]["large"].append(d.get("large", float('nan')))
def ms(x):
    x = [v for v in x if v == v]
    return (f"{sum(x)/len(x):.3f}±{st.pstdev(x):.3f}" if len(x) > 1 else (f"{x[0]:.3f}" if x else "-"))
print(f"{'recipe':26}{'tail Dice@R8':>16}{'adrenal':>13}{'large':>13}")
for k, lab in [("baseline", "baseline (broken)"), ("mixedr", "+ mixed-R"), ("tversky", "+ Focal-Tversky"),
               ("all", "MITIGATION (all)")]:
    if k in by: print(f"{lab:26}{ms(by[k]['tail']):>16}{ms(by[k]['adr']):>13}{ms(by[k]['large']):>13}")
b = by.get("baseline", {}).get("tail", []); a = by.get("all", {}).get("tail", [])
if b and a: print(f"\n-> mitigation recovers tail Dice by {sum(a)/len(a)-sum(b)/len(b):+.3f}  (large organs unchanged)")
PY
echo "==================== MITIGATION DONE @ $(date '+%F %T') ===================="
