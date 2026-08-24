#!/usr/bin/env bash
# FRAGILITY-AWARE RECONSTRUCTION (the method motivated by our metric-blindness result). The SOTA (recon-then-seg)
# trains the recon for IMAGE QUALITY (L1/SSIM) -- which we PROVED is blind to per-organ fragility. So: train the
# recon to reconstruct FRAGILE ORGANS faithfully (per-pixel fragility-weighted recon loss). Compare on tail Dice.
# Same seg (uniform loss), same mask (vardensity), same budget R8, 3 seeds -> isolates the RECON objective.
set -uo pipefail
ENVBIN=${ENVBIN:-/home/user/anaconda3/envs/mrifrag/bin}
P=${P:-/media/user/B4864CD4864C98AE/mri_fragility}
cd "$P"; export PYTHONUNBUFFERED=1; PY="$ENVBIN/python"; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

jobs=()
for s in 0 1 2; do
  jobs+=("fr_zf|--mask vardensity|$s")                          # zero-filled -> segment (no recon)
  jobs+=("fr_plain|--mask vardensity --recon|$s")              # plain L1 recon -> segment = SOTA (B2)
  jobs+=("fr_frag|--mask vardensity --recon --frag_recon|$s")  # FRAGILITY-AWARE recon -> segment = OURS
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

echo "==================== FRAGILITY-AWARE RECON RESULTS @ $(date '+%F %T') ===================="
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
print(f"{'pipeline':34}{'tailDice@R8':>15}{'adrenal':>13}{'large':>13}")
ref=None
for lab,key in [("zero-filled -> seg","fr_zf"),("plain recon -> seg  (SOTA B2)","fr_plain"),("FRAGILITY recon -> seg (OURS)","fr_frag")]:
    by=load(f"outputs/results/b1_{key}_s*.json")
    if not by["tail"]: continue
    tv=[v for v in by['tail'] if v==v]; mt=sum(tv)/len(tv)
    d=f"  ({mt-ref:+.3f} vs SOTA)" if ref is not None else ""
    if key=="fr_plain": ref=mt
    print(f"{lab:34}{ms(by['tail']):>15}{ms(by['adr']):>13}{ms(by['large']):>13}{d}")
print("\n-> KEY: does FRAGILITY recon beat plain recon (SOTA) on tail/adrenal at equal budget & seg?")
PY
echo "==================== FRAG-RECON DONE @ $(date '+%F %T') ===================="
