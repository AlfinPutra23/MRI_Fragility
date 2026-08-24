#!/usr/bin/env bash
# FG-LOUPE (fragility-guided sampling): lambda sweep x 3 seeds on 2 GPUs. Compared to loupe_uniform_s* (already
# produced by the multiseed) -> same pipeline, ONLY the coverage term differs. Idempotent. Launch AFTER bnathuzgm.
set -uo pipefail
ENVBIN=${ENVBIN:-/home/user/anaconda3/envs/mrifrag/bin}
P=${P:-/media/user/B4864CD4864C98AE/mri_fragility}
cd "$P"; export PYTHONUNBUFFERED=1; PY="$ENVBIN/python"

# lambda weights the cross-entropy coverage penalty (H~5); small = nudge, larger = stronger fragility bias
jobs=(); for s in 0 1 2; do for lam in 0.05 0.15 0.5; do jobs+=("$lam $s"); done; done
i=0
while [ $i -lt ${#jobs[@]} ]; do
  pids=()
  for g in 0 1; do
    [ $i -ge ${#jobs[@]} ] && break
    read lam seed <<< "${jobs[$i]}"; tag="fgloupe_l${lam}_s${seed}"
    if [ ! -f "outputs/results/b1_${tag}.json" ]; then
      echo "[$(date '+%T')] GPU$g: $tag"
      CUDA_VISIBLE_DEVICES=$g "$PY" scripts/b1_joint.py --mask learned --loss uniform --frag_cov_w "$lam" \
          --R 8 --epochs 60 --seed "$seed" --tag "$tag" > "outputs/logs/b1_${tag}.log" 2>&1 &
      pids+=($!)
    fi
    i=$((i+1))
  done
  [ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
done

echo "==================== FG-LOUPE vs LOUPE @ $(date '+%F %T') ===================="
"$PY" - <<'PY'
import json, glob, os, collections, statistics as st
TAIL=["gallbladder","esophagus","pancreas","adrenal_R","adrenal_L","duodenum"]; LEARN=[o for o in TAIL if o!="esophagus"]
def tm(po,ks): v=[po[o] for o in ks if o in po]; return sum(v)/len(v) if v else float('nan')
def grab(pat):
    xs=collections.defaultdict(list)
    for f in glob.glob(f"outputs/results/{pat}"):
        po=json.load(open(f)).get("per_organ",{}); xs["all"].append(tm(po,TAIL)); xs["learn"].append(tm(po,LEARN))
    return xs
def ms(x): return f"{sum(x)/len(x):.3f}"+(f"±{st.pstdev(x):.3f}" if len(x)>1 else "")
lo=grab("b1_loupe_uniform_s*.json")
print(f"{'variant':22}{'tail(no-eso)':>16}{'tail(all6)':>14}")
if lo["learn"]: print(f"{'LOUPE (baseline)':22}{ms(lo['learn']):>16}{ms(lo['all']):>14}")
for lam in ["0.05","0.15","0.5"]:
    fg=grab(f"b1_fgloupe_l{lam}_s*.json")
    if fg["learn"]:
        d=(sum(fg['learn'])/len(fg['learn']))-(sum(lo['learn'])/len(lo['learn'])) if lo['learn'] else float('nan')
        print(f"{'FG-LOUPE lambda='+str(lam):22}{ms(fg['learn']):>16}{ms(fg['all']):>14}   d_vs_LOUPE={d:+.3f}")
PY
echo "==================== FG-LOUPE DONE @ $(date '+%F %T') ===================="
