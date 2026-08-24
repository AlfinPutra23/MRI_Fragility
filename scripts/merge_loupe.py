"""Merge TC-LOUPE seed batches into a single verdict over ALL seeds (settles the n=3 ambiguity honestly).
  python merge_loupe.py f1.json f2.json ... -> outputs/results/conditional_loupe_merged.json"""
import sys, json, numpy as np
from scipy.stats import wilcoxon

files = sys.argv[1:]
gen, cond, div = {}, {}, {}
for f in files:
    d = json.load(open(f))
    gen.update(d.get("generic", {})); cond.update(d.get("conditional", {})); div.update(d.get("mask_divergence", {}))
seeds = sorted(set(gen) & set(cond), key=int)
gt = np.array([gen[s]["tail"] for s in seeds]); ct = np.array([cond[s]["tail"] for s in seeds]); dl = ct - gt
out = {"generic": gen, "conditional": cond, "mask_divergence": div, "seeds_used": seeds,
       "SUMMARY": {"n_seeds": len(seeds),
                   "generic_tail_mean": round(float(gt.mean()), 4), "conditional_tail_mean": round(float(ct.mean()), 4),
                   "delta": round(float(dl.mean()), 4), "delta_std": round(float(dl.std(ddof=1)), 4),
                   "wilcoxon_p": float(wilcoxon(ct, gt).pvalue) if (dl != 0).any() else None,
                   "conditional_wins_seeds": int((dl > 0).sum()),
                   "mean_mask_divergence": round(float(np.mean(list(div.values()))), 4) if div else None}}
json.dump(out, open("outputs/results/conditional_loupe_merged.json", "w"), indent=2)
s = out["SUMMARY"]
print(f"=== TC-LOUPE MERGED VERDICT (n={s['n_seeds']} seeds) ===")
print(f"  generic {s['generic_tail_mean']} vs conditional {s['conditional_tail_mean']}")
print(f"  Δ = {s['delta']:+.4f} ± {s['delta_std']:.4f}  (Wilcoxon p={s['wilcoxon_p']}, wins {s['conditional_wins_seeds']}/{s['n_seeds']})")
sig = s['wilcoxon_p'] is not None and s['wilcoxon_p'] < 0.05
print("  VERDICT:", "REAL WIN (Δ>0, significant)" if (s['delta'] > 0.005 and sig)
      else "PROVEN NULL/TIE (settle it, drop the method)" if abs(s['delta']) < 0.005 or s['conditional_wins_seeds'] <= s['n_seeds'] / 2
      else "STILL SUGGESTIVE (positive but n.s. — honest 'promising', not a claim)")
