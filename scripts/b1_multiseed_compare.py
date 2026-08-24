"""Aggregate B1 multi-seed runs: mean +- std tail Dice per variant, + the two key gaps with significance."""
import json, glob, re, numpy as np
from scipy.stats import ttest_rel
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ORDER = [("random_fixed","random (fixed)"),("equi_fixed","equispaced (fixed)"),("vd_fixed","var-density (fixed)"),
         ("loupe_uniform","LOUPE (learned)"),("ours","OURS (learned+frag)")]
by = {}
for f in glob.glob("outputs/results/b1_*_s*.json"):
    d = json.load(open(f)); m = re.match(r"(.+)_s(\d+)", d["tag"])
    if not m: continue
    by.setdefault(m.group(1), {})[int(m.group(2))] = d["tail"]

print(f"\n=== B1 multi-seed: tail Dice @R8 (mean +- std over seeds) ===")
print(f"{'variant':22}{'tail (mean+-std)':>20}{'n':>4}")
means = {}
for base, lab in ORDER:
    if base not in by: print(f"{lab:22}{'(missing)':>20}"); continue
    v = np.array(list(by[base].values())); means[base] = v
    print(f"{lab:22}{v.mean():>12.3f} +-{v.std():.3f}{len(v):>5}")

def gap(a, b):
    if a not in means or b not in means: return None
    common = sorted(set(by[a]) & set(by[b]))
    va = np.array([by[a][s] for s in common]); vb = np.array([by[b][s] for s in common])
    p = ttest_rel(va, vb).pvalue if len(common) > 1 else np.nan
    return va.mean()-vb.mean(), (va-vb).std(), p, len(common)

print("\n=== key gaps (paired across seeds) ===")
for lab, a, b in [("learned sampling (LOUPE - fixed vd)","loupe_uniform","vd_fixed"),
                  ("fragility prior (ours - LOUPE)","ours","loupe_uniform"),
                  ("TOTAL (ours - fixed vd)","ours","vd_fixed")]:
    g = gap(a, b)
    if g: print(f"  {lab:36} +{g[0]:.3f} +-{g[1]:.3f}  (paired-t p={g[2]:.3f}, n={g[3]} seeds)")

# figure with error bars
labs=[l for b,l in ORDER if b in means]; mu=[means[b].mean() for b,_ in ORDER if b in means]; sd=[means[b].std() for b,_ in ORDER if b in means]
cols=["#999","#bbb","#5b8def","#f4a259","#d93025"][:len(labs)]
fig,ax=plt.subplots(figsize=(8.5,5.2)); ax.bar(range(len(labs)),mu,yerr=sd,capsize=5,color=cols)
for i,(m,s) in enumerate(zip(mu,sd)): ax.text(i,m+s+0.004,f"{m:.3f}",ha="center",fontweight="bold",fontsize=9)
ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs,fontsize=8.5,rotation=10); ax.set_ylabel("tail Dice @R8 (mean±std, 3 seeds)")
ax.set_title("B1 (multi-seed): fragility-guided task-aware sampling > learned > fixed",fontweight="bold")
fig.tight_layout(); fig.savefig("outputs/plots/b1_gate_multiseed.png",dpi=145); print("\nwrote outputs/plots/b1_gate_multiseed.png")
json.dump({b:list(map(float,means[b])) for b in means}, open("outputs/results/b1_multiseed.json","w"), indent=2)
