#!/usr/bin/env bash
# DJ-Seg on the RECON pipeline (the orthogonal combination). Per-organ audit showed recon fixes the mid-size tail
# (duodenum/pancreas/gallbladder) while Focal-Tversky fixes the adrenal (recall collapse) -> they are ORTHOGONAL, so
# stacking should beat the recon-only SOTA. SOTA = recon + uniform (fr_plain). OURS = recon + Focal-Tversky (+ mixed-R).
# Same mask (vardensity), budget R8, 3 seeds. Reuses fr_plain_s* as the SOTA reference.
set -uo pipefail
ENVBIN=${ENVBIN:-/home/user/anaconda3/envs/mrifrag/bin}
P=${P:-/media/user/B4864CD4864C98AE/mri_fragility}
cd "$P"; export PYTHONUNBUFFERED=1; PY="$ENVBIN/python"; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

jobs=()
for s in 0 1 2; do
  jobs+=("djr_tv|--recon --loss fragweighted --tversky|$s")               # recon + Focal-Tversky = OURS
  jobs+=("djr_all|--recon --loss fragweighted --tversky --mixed_r|$s")    # recon + Focal-Tversky + mixed-R = full
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

echo "==================== DJ-Seg + RECON RESULTS @ $(date '+%F %T') ===================="
"$PY" - <<'PY'
import json, glob, os, collections, statistics as st
TAIL=["gallbladder","esophagus","pancreas","adrenal_R","adrenal_L","duodenum"]
TAIL5=[o for o in TAIL if o!="esophagus"]           # exclude the data-starved proxy artifact
def tm(po,ks): v=[po[o] for o in ks if o in po]; return sum(v)/len(v) if v else float('nan')
def load(pat):
    by=collections.defaultdict(list)
    for f in glob.glob(pat):
        d=json.load(open(f)); po=d.get("per_organ",{}); t=tm(po,TAIL); lg=d.get("large",float('nan'))
        if t==0.0 and lg==0.0: continue
        by["tail"].append(t); by["tail5"].append(tm(po,TAIL5)); by["adr"].append(tm(po,["adrenal_R","adrenal_L"])); by["large"].append(lg)
    return by
def ms(x):
    x=[v for v in x if v==v]; return f"{sum(x)/len(x):.3f}±{st.pstdev(x):.3f}" if len(x)>1 else (f"{x[0]:.3f}" if x else "-")
print(f"{'pipeline':30}{'tail(6)':>14}{'tail5(no-eso)':>15}{'adrenal':>12}{'large':>12}")
ref=None
for lab,key in [("recon+uniform (SOTA)","fr_plain"),("recon+FocalTversky (OURS)","djr_tv"),("recon+FTv+mixedR (full)","djr_all")]:
    by=load(f"outputs/results/b1_{key}_s*.json")
    if not by["tail"]: print(f"{lab:30}{'(pending)':>14}"); continue
    tv=[v for v in by['tail5'] if v==v]; mt=sum(tv)/len(tv)
    if key=="fr_plain": ref=mt
    d=f"  ({mt-ref:+.3f})" if (ref is not None and key!="fr_plain") else ""
    print(f"{lab:30}{ms(by['tail']):>14}{ms(by['tail5']):>15}{ms(by['adr']):>12}{ms(by['large']):>12}{d}")
print("\n-> KEY: does recon+FocalTversky beat recon-SOTA on tail5(no-eso)/adrenal? (orthogonal levers stacking)")
PY
echo "==================== DJ-RECON DONE @ $(date '+%F %T') ===================="
