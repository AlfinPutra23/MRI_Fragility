#!/usr/bin/env bash
# FRAGILITY-GUIDED DISTILLATION (the novel-method shot). A CLEAN-scan teacher (knows the fragile organs, adrenal
# ~0.64) transfers its predictions to the ACCELERATED student (adrenal ~0.44) via a fragility-weighted KL. Different
# mechanism from sampling/recon/loss-reweighting -> could recover fragile organs where those washed. Single GPU (GPU env).
set -uo pipefail
ENVBIN=${ENVBIN:-/home/user/anaconda3/envs/mrifrag/bin}
P=${P:-/media/user/B4864CD4864C98AE/mri_fragility}
cd "$P"; export PYTHONUNBUFFERED=1; PY="$ENVBIN/python"; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPU=${GPU:-1}; mkdir -p models
TEACHER=models/teacher_clean.pth

echo "[$(date '+%T')] GPU$GPU: train clean teacher (R=1) if missing"
if [ ! -f "$TEACHER" ]; then
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/b1_joint.py --mask vardensity --R 1 --epochs 60 --loss uniform --seed 0 \
      --save_net "$TEACHER" --tag distill_teacher > outputs/logs/b1_distill_teacher.log 2>&1
fi
[ -f "$TEACHER" ] || { echo "teacher train FAILED"; tail -5 outputs/logs/b1_distill_teacher.log; exit 1; }

# students @R8: baseline (no distill) vs distilled (two weights), 3 seeds, sequential on one GPU
jobs=()
for s in 0 1 2; do
  jobs+=("dist_base||$s")                                    # R8 baseline (no distill)
  jobs+=("dist_kd05|--teacher $TEACHER --distill_w 0.5|$s")  # + fragility-weighted distillation (0.5)
  jobs+=("dist_kd2|--teacher $TEACHER --distill_w 2.0|$s")   # + distillation (2.0)
done
for j in "${jobs[@]}"; do
  IFS='|' read name flags seed <<< "$j"; tag="${name}_s${seed}"
  [ -f "outputs/results/b1_${tag}.json" ] && continue
  echo "[$(date '+%T')] GPU$GPU: $tag  ($flags)"
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/b1_joint.py --mask vardensity --R 8 --epochs 60 --loss uniform $flags \
      --tag "$tag" --seed "$seed" > "outputs/logs/b1_${tag}.log" 2>&1
done

echo "==================== DISTILLATION RESULTS @ $(date '+%F %T') ===================="
"$PY" - <<'PY'
import json, glob, os, collections, statistics as st
TAIL5=["gallbladder","pancreas","adrenal_R","adrenal_L","duodenum"]
def tm(po,ks): v=[po[o] for o in ks if o in po]; return sum(v)/len(v) if v else float('nan')
def load(pat):
    by=collections.defaultdict(list)
    for f in glob.glob(pat):
        d=json.load(open(f)); po=d.get("per_organ",{})
        if d.get("large",1)==0 and d.get("tail",1)==0: continue
        by["tail5"].append(tm(po,TAIL5)); by["adr"].append(tm(po,["adrenal_R","adrenal_L"])); by["large"].append(d.get("large",float('nan')))
    return by
def ms(x):
    x=[v for v in x if v==v]; return f"{sum(x)/len(x):.3f}±{st.pstdev(x):.3f}" if len(x)>1 else (f"{x[0]:.3f}" if x else "-")
print(f"{'student @R8':28}{'tail5':>14}{'adrenal':>13}{'large':>13}")
ref=None
for lab,key in [("baseline (no distill)","dist_base"),("+ distill w0.5 (ours)","dist_kd05"),("+ distill w2.0 (ours)","dist_kd2")]:
    by=load(f"outputs/results/b1_{key}_s*.json")
    if not by["tail5"]: print(f"{lab:28}{'(pending)':>14}"); continue
    tv=[v for v in by['tail5'] if v==v]; mt=sum(tv)/len(tv)
    if key=="dist_base": ref=mt
    d=f"  ({mt-ref:+.3f})" if (ref is not None and key!="dist_base") else ""
    print(f"{lab:28}{ms(by['tail5']):>14}{ms(by['adr']):>13}{ms(by['large']):>13}{d}")
print("\n-> does clean->accel distillation recover fragile organs beyond the R8 baseline?")
PY
echo "==================== DISTILL DONE @ $(date '+%F %T') ===================="
