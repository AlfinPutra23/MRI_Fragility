#!/usr/bin/env bash
# nnU-Net Focal-Tversky (the proxy-winning recall loss) in the REAL 3D pipeline. Seed-matched (s42) to the existing
# Uniform_s42 baseline (0.647) and FragW4_s42 (0.670, +2.2). Train fold_0 -> predict clean+R8 -> compare per-organ.
# Q: does Focal-Tversky beat FragW4 on the fragile organs in the real pipeline? Single GPU (GPU env var).
set -uo pipefail
ENVBIN=${ENVBIN:-/home/user/anaconda3/envs/mrifrag/bin}
P=${P:-/media/user/B4864CD4864C98AE/mri_fragility}
VAR=${VAR:-/home/user/anaconda3/envs/mrifrag/lib/python3.11/site-packages/nnunetv2/training/nnUNetTrainer/variants}
GPU=${GPU:-0}
export nnUNet_raw=$P/nnUNet_raw nnUNet_preprocessed=$P/nnUNet_preprocessed nnUNet_results=$P/nnUNet_results
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=$nnUNet_raw/Dataset501_MRIfrag; RES=$nnUNet_results/Dataset501_MRIfrag; LOGD=$P/outputs/logs
TR=nnUNetTrainer_FocalTversky_s42
cp "$P/scripts/nnunet_focaltversky_trainer.py" "$VAR/"

echo "[$(date '+%F %T')] TRAIN $TR (GPU$GPU, ~4h)"
mdir="$RES/${TR}__nnUNetPlans__3d_fullres/fold_0"
if [ ! -f "$mdir/checkpoint_final.pth" ]; then
  CUDA_VISIBLE_DEVICES=$GPU "$ENVBIN/nnUNetv2_train" 501 3d_fullres 0 -tr "$TR" > "$LOGD/ft_train.log" 2>&1
fi
echo "[$(date '+%F %T')] PREDICT clean + R8"
for tag in clean R8; do
  o="$D/predsSW_${TR}_$tag"
  if [ "$(ls "$o"/*.nii.gz 2>/dev/null | wc -l)" -ne 240 ]; then
    CUDA_VISIBLE_DEVICES=$GPU "$ENVBIN/nnUNetv2_predict" -i "$D/imagesTs_$tag" -o "$o" \
        -d 501 -c 3d_fullres -f 0 -tr "$TR" --disable_tta >> "$LOGD/ft_predict.log" 2>&1
  fi
done

echo "==================== FOCAL-TVERSKY nnU-NET COMPARE @ $(date '+%F %T') ===================="
cd "$P/scripts"; "$ENVBIN/python" - <<'PY'
import glob, os, numpy as np, nibabel as nib, sys, collections
from scipy.stats import wilcoxon
sys.path.insert(0,"."); import labels as L
D="../nnUNet_raw/Dataset501_MRIfrag"; lab_dir=f"{D}/labelsTs"
TAIL={o:n for o,n in L.ABDO.items() if o in L.TAIL}
def dice(a,b): i=np.logical_and(a,b).sum(); return 1.0 if a.sum()+b.sum()==0 else 2*i/(a.sum()+b.sum())
def per_case_tail(preds):
    out={}
    for gp in sorted(glob.glob(f"{lab_dir}/*.nii.gz")):
        c=os.path.basename(gp); pp=f"{preds}/{c}"
        if not os.path.exists(pp): continue
        g=np.asanyarray(nib.load(gp).dataobj); p=np.asanyarray(nib.load(pp).dataobj)
        out[c]=np.mean([dice(g==o,p==o) for o in TAIL])
    return out
runs={"Uniform_s42(base)":"predsSW_nnUNetTrainer_Uniform_s42_R8",
      "FragW4_s42":"predsSW_nnUNetTrainer_FragW4_s42_R8",
      "FocalTversky_s42":"predsSW_nnUNetTrainer_FocalTversky_s42_R8"}
tc={k:per_case_tail(f"{D}/{v}") for k,v in runs.items()}
print(f"{'model @R8':22}{'tail Dice':>11}{'vs base':>10}{'Wilcoxon p':>13}")
base=tc["Uniform_s42(base)"]; bm=np.mean(list(base.values()))
for k,d in tc.items():
    if not d: print(f"{k:22}{'(pending)':>11}"); continue
    m=np.mean(list(d.values())); com=[c for c in d if c in base]
    p=wilcoxon([d[c] for c in com],[base[c] for c in com]).pvalue if k!="Uniform_s42(base)" and com else float('nan')
    print(f"{k:22}{m:>11.4f}{m-bm:>+10.4f}{p:>13.2e}")
# per-organ for FocalTversky vs base
print("\nper-organ @R8 (Focal-Tversky vs Uniform base):")
def per_organ(preds):
    acc=collections.defaultdict(list)
    for gp in sorted(glob.glob(f"{lab_dir}/*.nii.gz")):
        c=os.path.basename(gp); pp=f"{preds}/{c}"
        if not os.path.exists(pp): continue
        g=np.asanyarray(nib.load(gp).dataobj); p=np.asanyarray(nib.load(pp).dataobj)
        for o,n in TAIL.items(): acc[n].append(dice(g==o,p==o))
    return {n:np.mean(v) for n,v in acc.items()}
ub=per_organ(f"{D}/{runs['Uniform_s42(base)']}"); ftp=per_organ(f"{D}/{runs['FocalTversky_s42']}")
for n in TAIL.values():
    if n in ub and n in ftp: print(f"  {n:14}{ub[n]:.3f} -> {ftp[n]:.3f}  ({ftp[n]-ub[n]:+.3f})")
PY
echo "==================== FT DONE @ $(date '+%F %T') ===================="
