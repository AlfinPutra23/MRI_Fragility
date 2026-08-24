#!/bin/bash
# BETTER METHOD: combine the INDEPENDENT winners. LOUPE + Focal-Tversky vs generic LOUPE (the +0.043 winner).
# Orthogonal mechanisms (task-driven sampling + recall loss vs tail 0-Dice collapse); untested together. GPU-gated.
cd /media/user/B4864CD4864C98AE/mri_fragility
LOG=outputs/logs/loupe_combo.log
PY=/home/user/anaconda3/envs/magicnet/bin/python
[ -f outputs/results/loupe_combo.json ] && { echo "[$(date '+%F %T')] exists->skip" >>"$LOG"; exit 0; }
echo "[$(date '+%F %T')] LOUPE-combo queued, waiting for idle GPU" >>"$LOG"
while true; do FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits|awk -F', *' '$2<3000{print $1;exit}'); [ -n "$FREE" ]&&break; sleep 120; done
echo "[$(date '+%F %T')] launch on GPU$FREE (3 seeds x 2 arms)" >>"$LOG"
for s in 0 1 2; do
  [ -f outputs/results/b1_cLOUPE_s$s.json ]   || CUDA_VISIBLE_DEVICES=$FREE PYTHONNOUSERSITE=1 $PY -u scripts/b1_joint.py --mask learned --loss fragweighted --R 8 --epochs 60 --seed $s --tag cLOUPE_s$s >>"$LOG" 2>&1
  [ -f outputs/results/b1_cLOUPEtv_s$s.json ] || CUDA_VISIBLE_DEVICES=$FREE PYTHONNOUSERSITE=1 $PY -u scripts/b1_joint.py --mask learned --tversky        --R 8 --epochs 60 --seed $s --tag cLOUPEtv_s$s >>"$LOG" 2>&1
done
$PY - <<'PY' >>"$LOG" 2>&1
import json,glob,numpy as np
from scipy.stats import wilcoxon
def m(p): return np.array([json.load(open(f))['tail'] for f in sorted(glob.glob(f'outputs/results/{p}'))])
g=m('b1_cLOUPE_s*.json'); c=m('b1_cLOUPEtv_s*.json'); n=min(len(g),len(c)); g,c=g[:n],c[:n]; d=c-g
o={'generic_loupe':round(float(g.mean()),4),'loupe_tversky':round(float(c.mean()),4),'delta':round(float(d.mean()),4),
   'wilcoxon_p':(float(wilcoxon(c,g).pvalue) if (d!=0).any() else None),'wins':int((d>0).sum()),'n':int(n)}
json.dump(o,open('outputs/results/loupe_combo.json','w'),indent=2)
print('LOUPE-COMBO:',o)
print('VERDICT:', 'COMBO WINS' if (o['delta']>0.005 and o['wins']>=2) else 'no gain over generic LOUPE')
PY
echo "[$(date '+%F %T')] LOUPE-combo DONE" >>"$LOG"
