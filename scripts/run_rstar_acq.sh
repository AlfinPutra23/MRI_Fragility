#!/usr/bin/env bash
# R*-GUIDED ACQUISITION test (the novel-method attempt). Same budget (R=8), same UNIFORM loss -> isolates the
# ACQUISITION effect: does a fragility-density mask (spend lines where low-R* organs need them) beat plain
# variable-density on the fragile organs? Also LOUPE (learned) as the upper bound. 3 seeds, 2 GPUs, idempotent.
set -uo pipefail
ENVBIN=${ENVBIN:-/home/user/anaconda3/envs/mrifrag/bin}
P=${P:-/media/user/B4864CD4864C98AE/mri_fragility}
cd "$P"; export PYTHONUNBUFFERED=1; PY="$ENVBIN/python"; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

jobs=()   # AUDITED masks: fragA/B protect HIGH freq (21/4/7 low/mid/high) vs VD (23/8/1), same budget
for s in 0 1 2; do
  jobs+=("acq_vd|--mask vardensity|$s")                                       # baseline acquisition (23/8/1)
  jobs+=("acq_fragA|--mask fragility --frag_lam 1.0 --frag_floor 0.01|$s")    # R*-guided moderate high (21/4/7)
  jobs+=("acq_fragB|--mask fragility --frag_lam 1.0 --frag_floor 0.005|$s")   # R*-guided stronger high (21/3/8)
  jobs+=("acq_loupe|--mask learned|$s")                                       # learned upper bound
done

i=0
while [ $i -lt ${#jobs[@]} ]; do
  pids=()
  for g in 0 1; do
    [ $i -ge ${#jobs[@]} ] && break
    IFS='|' read name flags seed <<< "${jobs[$i]}"; tag="${name}_s${seed}"
    if [ ! -f "outputs/results/b1_${tag}.json" ]; then
      echo "[$(date '+%T')] GPU$g: $tag  ($flags)"
      CUDA_VISIBLE_DEVICES=$g $PY scripts/b1_joint.py --R 8 --epochs 60 --loss uniform $flags \
          --tag "$tag" --seed "$seed" > "outputs/logs/b1_${tag}.log" 2>&1 &
      pids+=($!)
    fi
    i=$((i+1))
  done
  [ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
done

echo "==================== R*-GUIDED ACQUISITION RESULTS @ $(date '+%F %T') ===================="
"$PY" - <<'PY'
import json, glob, os, collections, statistics as st
TAIL=["gallbladder","esophagus","pancreas","adrenal_R","adrenal_L","duodenum"]
def tm(po,ks): v=[po[o] for o in ks if o in po]; return sum(v)/len(v) if v else float('nan')
def load(pat):
    by=collections.defaultdict(list)
    for f in glob.glob(pat):
        d=json.load(open(f)); po=d.get("per_organ",{}); t=tm(po,TAIL); lg=d.get("large",float('nan'))
        if t==0.0 and lg==0.0: continue
        by["tail"].append(t); by["adr"].append(tm(po,["adrenal_R","adrenal_L"])); by["large"].append(lg)
    return by
def ms(x):
    x=[v for v in x if v==v]; return f"{sum(x)/len(x):.3f}±{st.pstdev(x):.3f}" if len(x)>1 else (f"{x[0]:.3f}" if x else "-")
print(f"{'acquisition':28}{'tailDice@R8':>15}{'adrenal':>13}{'large':>13}")
ref=None
for lab,key in [("variable-density (base)","acq_vd"),("R*-guided fragA hi","acq_fragA"),("R*-guided fragB hi+","acq_fragB"),("LOUPE (learned, upper bnd)","acq_loupe")]:
    by=load(f"outputs/results/b1_{key}_s*.json")
    if not by["tail"]: continue
    tv=[v for v in by['tail'] if v==v]; mt=sum(tv)/len(tv)
    d=f"  ({mt-ref:+.3f} vs VD)" if ref is not None else ""
    if ref is None: ref=mt
    print(f"{lab:28}{ms(by['tail']):>15}{ms(by['adr']):>13}{ms(by['large']):>13}{d}")
PY
echo "==================== R*-GUIDED ACQ DONE @ $(date '+%F %T') ===================="
