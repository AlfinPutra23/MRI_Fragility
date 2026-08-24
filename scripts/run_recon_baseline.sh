#!/usr/bin/env bash
# (b) RECON-THEN-SEGMENT baseline in the REAL pipeline: segment the CS-reconstructed R8 images with the existing
# Uniform_s42 nnU-Net, vs the matched zero-filled R8. Isolates the reconstruction effect (same mask for both).
# Single GPU (GPU env). Needs recon_cs.py to have produced imagesTs_R8zf + imagesTs_R8cs.
set -uo pipefail
ENVBIN=${ENVBIN:-/home/user/anaconda3/envs/mrifrag/bin}
P=${P:-/media/user/B4864CD4864C98AE/mri_fragility}; GPU=${GPU:-0}
export nnUNet_raw=$P/nnUNet_raw nnUNet_preprocessed=$P/nnUNet_preprocessed nnUNet_results=$P/nnUNet_results
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=$nnUNet_raw/Dataset501_MRIfrag; TR=nnUNetTrainer_Uniform_s42; cd "$P"

for tag in R8zf R8cs; do
  o="$D/predsRECON_$tag"
  if [ "$(ls "$o"/*.nii.gz 2>/dev/null|wc -l)" -ne 240 ]; then
    echo "[$(date '+%T')] predict $tag (GPU$GPU)"
    CUDA_VISIBLE_DEVICES=$GPU "$ENVBIN/nnUNetv2_predict" -i "$D/imagesTs_$tag" -o "$o" -d 501 -c 3d_fullres -f 0 -tr "$TR" --disable_tta >> "$P/outputs/logs/recon_predict.log" 2>&1
  fi
done

echo "==================== RECON-THEN-SEGMENT COMPARE @ $(date '+%F %T') ===================="
cd "$P/scripts"; "$ENVBIN/python" - <<'PY'
import glob, os, numpy as np, nibabel as nib, sys, collections
from scipy.stats import wilcoxon
sys.path.insert(0,"."); import labels as L
D="../nnUNet_raw/Dataset501_MRIfrag"; lab=f"{D}/labelsTs"
TAIL={o:n for o,n in L.ABDO.items() if o in L.TAIL}
def dice(a,b): i=np.logical_and(a,b).sum(); s=a.sum()+b.sum(); return 1.0 if s==0 else 2*i/s
def per_case(pred):
    d={}
    for gp in sorted(glob.glob(f"{lab}/*.nii.gz")):
        c=os.path.basename(gp); pp=f"{pred}/{c}"
        if not os.path.exists(pp): continue
        g=np.asanyarray(nib.load(gp).dataobj); p=np.asanyarray(nib.load(pp).dataobj)
        d[c]=np.mean([dice(g==o,p==o) for o in TAIL])
    return d
zf=per_case(f"{D}/predsRECON_R8zf"); cs=per_case(f"{D}/predsRECON_R8cs")
com=[c for c in zf if c in cs]
mz=np.mean([zf[c] for c in com]); mc=np.mean([cs[c] for c in com])
print(f"cases compared: {len(com)}")
print(f"  zero-filled -> segment : tail Dice {mz:.4f}")
print(f"  CS-recon    -> segment : tail Dice {mc:.4f}   ({mc-mz:+.4f} vs zero-filled)")
if com: print(f"  per-case Wilcoxon p = {wilcoxon([cs[c] for c in com],[zf[c] for c in com]).pvalue:.2e}")
def per_organ(pred):
    acc=collections.defaultdict(list)
    for gp in sorted(glob.glob(f"{lab}/*.nii.gz")):
        c=os.path.basename(gp); pp=f"{pred}/{c}"
        if not os.path.exists(pp): continue
        g=np.asanyarray(nib.load(gp).dataobj); p=np.asanyarray(nib.load(pp).dataobj)
        for o,n in TAIL.items(): acc[n].append(dice(g==o,p==o))
    return {n:np.mean(v) for n,v in acc.items()}
z=per_organ(f"{D}/predsRECON_R8zf"); c=per_organ(f"{D}/predsRECON_R8cs")
print("per-organ (zero-filled -> CS-recon):")
for n in TAIL.values():
    if n in z and n in c: print(f"  {n:14}{z[n]:.3f} -> {c[n]:.3f}  ({c[n]-z[n]:+.3f})")
PY
echo "==================== RECON BASELINE DONE @ $(date '+%F %T') ===================="
