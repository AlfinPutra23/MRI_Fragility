#!/usr/bin/env bash
# B1: train all sampling variants @R8 on the 2D joint model, then compare tail Dice.
# Waits for a free GPU (so it doesn't fight the 3D nnU-Net jobs). 2D runs are fast (~20-30 min each).
set -uo pipefail
ENVBIN=${ENVBIN:-/home/user/anaconda3/envs/mrifrag/bin}
P=${P:-/media/user/B4864CD4864C98AE/mri_fragility}
cd "$P"
EP=${EP:-60}

echo "[$(date '+%F %T')] waiting for a free GPU (3D jobs to finish)..."
until ! ps -eo cmd | grep -qE "[n]nUNetv2_train"; do sleep 300; done
export CUDA_VISIBLE_DEVICES=0

run(){ # tag mask loss
  [ -f "outputs/results/b1_$1.json" ] && { echo "SKIP $1"; return 0; }
  echo "==================== B1 $1 ($2/$3) @ $(date '+%F %T') ===================="
  "$ENVBIN/python" scripts/b1_joint.py --mask "$2" --loss "$3" --R 8 --epochs $EP --tag "$1" \
      2>&1 | grep -E "\[B1" | tee -a outputs/logs/b1.log
}

run random_fixed   random      uniform
run equi_fixed     equispaced  uniform
run vd_fixed       vardensity  uniform
run loupe_uniform  learned     uniform
run ours           learned     fragweighted

echo "==================== B1 COMPARE @ $(date '+%F %T') ===================="
"$ENVBIN/python" - <<'EOF'
import json, glob, os
rows = []
for f in sorted(glob.glob("outputs/results/b1_*.json")):
    d = json.load(open(f)); rows.append((d["tag"], d["mask"], d["loss"], d["tail"], d["large"]))
order = {"random_fixed":0,"equi_fixed":1,"vd_fixed":2,"loupe_uniform":3,"ours":4}
rows.sort(key=lambda r: order.get(r[0], 9))
print(f"\n{'variant':16}{'mask':12}{'loss':14}{'TAIL Dice':>10}{'LARGE':>8}")
for t, m, l, ta, la in rows:
    print(f"{t:16}{m:12}{l:14}{ta:10.3f}{la:8.3f}")
d = dict(rows);
EOF
echo "==================== B1 DONE @ $(date '+%F %T') ===================="
