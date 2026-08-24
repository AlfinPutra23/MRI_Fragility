#!/usr/bin/env bash
# DJ-Seg HF ablation: does the HF-emphasis loss ADD on top of mixed-R + Focal-Tversky + oversampling?
# Reference = the mitigation 'all' recipe (b1_mit_all_s*, NO HF). Here: all+HF at lambda in {0.5,1.0} x3 seeds.
# 2-D proxy, fixed variable-density @R8, 2 GPUs, idempotent. Honest: if HF adds nothing, we report the null.
set -uo pipefail
ENVBIN=${ENVBIN:-/home/user/anaconda3/envs/mrifrag/bin}
P=${P:-/media/user/B4864CD4864C98AE/mri_fragility}
cd "$P"; export PYTHONUNBUFFERED=1; PY="$ENVBIN/python"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

jobs=()
for s in 0 1 2; do
  jobs+=("djhf05|--loss fragweighted --mixed_r --tversky --hf_w 0.5|$s")   # DJ-Seg with HF (lambda 0.5)
  jobs+=("djhf10|--loss fragweighted --mixed_r --tversky --hf_w 1.0|$s")   # DJ-Seg with HF (lambda 1.0)
done

i=0
while [ $i -lt ${#jobs[@]} ]; do
  pids=()
  for g in 0 1; do
    [ $i -ge ${#jobs[@]} ] && break
    IFS='|' read name flags seed <<< "${jobs[$i]}"; tag="${name}_s${seed}"
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

echo "==================== DJ-Seg HF ABLATION @ $(date '+%F %T') ===================="
"$PY" - <<'PY'
import json, glob, os, collections, statistics as st
TAIL = ["gallbladder", "esophagus", "pancreas", "adrenal_R", "adrenal_L", "duodenum"]
def tm(po, ks): v = [po[o] for o in ks if o in po]; return sum(v)/len(v) if v else float('nan')
def load(pat):
    by = collections.defaultdict(list)
    for f in glob.glob(pat):
        d = json.load(open(f)); po = d.get("per_organ", {})
        by["tail"].append(tm(po, TAIL)); by["adr"].append(tm(po, ["adrenal_R", "adrenal_L"])); by["large"].append(d.get("large", float('nan')))
    return by
def ms(x):
    x = [v for v in x if v == v]; return f"{sum(x)/len(x):.3f}±{st.pstdev(x):.3f}" if len(x) > 1 else (f"{x[0]:.3f}" if x else "-")
rows = [("DJ (mixed-R+Tversky, NO HF)", "outputs/results/b1_mit_all_s*.json"),
        ("DJ + HF (lambda 0.5)",        "outputs/results/b1_djhf05_s*.json"),
        ("DJ + HF (lambda 1.0)",        "outputs/results/b1_djhf10_s*.json")]
print(f"{'recipe':30}{'tail Dice@R8':>16}{'adrenal':>13}{'large':>13}")
base = None
for lab, pat in rows:
    by = load(pat)
    if not by["tail"]: continue
    print(f"{lab:30}{ms(by['tail']):>16}{ms(by['adr']):>13}{ms(by['large']):>13}")
    m = [v for v in by["tail"] if v == v]; m = sum(m)/len(m) if m else float('nan')
    if base is None: base = m
    elif m == m: print(f"{'   -> HF delta vs DJ:':30}{m-base:+.3f} tail Dice")
PY
echo "==================== DJ-Seg HF DONE @ $(date '+%F %T') ===================="
