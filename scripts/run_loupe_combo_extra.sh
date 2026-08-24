#!/bin/bash
cd /media/user/B4864CD4864C98AE/mri_fragility
LOG=outputs/logs/loupe_combo.log; PY=/home/user/anaconda3/envs/magicnet/bin/python
while true; do FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits|awk -F', *' '$2<3000{print $1;exit}'); [ -n "$FREE" ]&&break; sleep 120; done
echo "[$(date '+%F %T')] combo EXTRA seeds 3-6 on GPU$FREE" >>"$LOG"
for s in 3 4 5 6; do
  [ -f outputs/results/b1_cLOUPE_s$s.json ]   || CUDA_VISIBLE_DEVICES=$FREE PYTHONNOUSERSITE=1 $PY -u scripts/b1_joint.py --mask learned --loss fragweighted --R 8 --epochs 60 --seed $s --tag cLOUPE_s$s >>"$LOG" 2>&1
  [ -f outputs/results/b1_cLOUPEtv_s$s.json ] || CUDA_VISIBLE_DEVICES=$FREE PYTHONNOUSERSITE=1 $PY -u scripts/b1_joint.py --mask learned --tversky        --R 8 --epochs 60 --seed $s --tag cLOUPEtv_s$s >>"$LOG" 2>&1
done
$PY - <<'PY' >>"$LOG" 2>&1
import json,glob,numpy as np
from scipy.stats import wilcoxon
def load(p): return {f.split('_s')[1][0]:json.load(open(f))['tail'] for f in sorted(glob.glob(f'outputs/results/{p}'))}
g=load('b1_cLOUPE_s*.json'); c=load('b1_cLOUPEtv_s*.json'); ss=sorted(set(g)&set(c))
G=np.array([g[s] for s in ss]); C=np.array([c[s] for s in ss]); d=C-G
o={'n':len(ss),'generic_loupe':round(float(G.mean()),4),'loupe_tversky':round(float(C.mean()),4),'delta':round(float(d.mean()),4),
   'delta_std':round(float(d.std(ddof=1)),4),'wilcoxon_p':(float(wilcoxon(C,G).pvalue) if (d!=0).any() else None),'wins':int((d>0).sum())}
json.dump(o,open('outputs/results/loupe_combo_7seed.json','w'),indent=2); print('COMBO 7-SEED:',o)
PY
echo "[$(date '+%F %T')] combo extra DONE" >>"$LOG"
