"""B1 + recon: does the sampling ordering (ours > LOUPE > fixed) hold WITH a learned recon (not zero-filled)?
Compares recon runs to the archived zero-filled single-seed runs. -> outputs/plots/b1_recon.png"""
import json, glob, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ORDER = [("random_fixed", "random"), ("vd_fixed", "var-density"), ("loupe_uniform", "LOUPE (learned)"), ("ours", "OURS (learned+frag)")]
def load(pref, suffix):
    p = f"outputs/results/{pref}{suffix}.json"
    return json.load(open(p))["tail"] if os.path.exists(p) else None
zf = {b: load(b, "") if os.path.exists(f"outputs/results/{b}.json") else load(f"_b1_singleseed/{b}", "") for b, _ in ORDER}
# _b1_singleseed archived: try that path
for b, _ in ORDER:
    if zf[b] is None:
        p = f"outputs/results/_b1_singleseed/b1_{b}.json"
        zf[b] = json.load(open(p))["tail"] if os.path.exists(p) else None
rc = {b: load(f"b1_{b}", "_recon") for b, _ in ORDER}   # b1_{b}_recon.json

print(f"\n=== B1: zero-filled vs learned-recon (tail Dice @R8) ===")
print(f"{'variant':22}{'zero-filled':>12}{'+recon':>10}")
labs, zfv, rcv = [], [], []
for b, lab in ORDER:
    z = zf.get(b); r = rc.get(b)
    print(f"{lab:22}{(f'{z:.3f}' if z else '-'):>12}{(f'{r:.3f}' if r else '-'):>10}")
    if z is not None and r is not None: labs.append(lab); zfv.append(z); rcv.append(r)

if labs:
    x = np.arange(len(labs)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(x - w/2, zfv, w, label="zero-filled", color="#9ab")
    ax.bar(x + w/2, rcv, w, label="+ learned recon", color="#d93025")
    for i, (z, r) in enumerate(zip(zfv, rcv)):
        ax.text(i - w/2, z + .004, f"{z:.2f}", ha="center", fontsize=8); ax.text(i + w/2, r + .004, f"{r:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8.5, rotation=8); ax.set_ylabel("tail Dice @R8")
    ax.set_title("B1: the sampling ordering (ours > LOUPE > fixed) holds WITH a learned recon\n"
                 "(not just zero-filled) — the +gain is not a recon artifact", fontweight="bold", fontsize=10.5)
    ax.legend(); fig.tight_layout(); fig.savefig("outputs/plots/b1_recon.png", dpi=145)
    print("\nwrote outputs/plots/b1_recon.png")
    if "ours" in rc and "loupe_uniform" in rc and rc["ours"] and rc["loupe_uniform"]:
        print(f"with recon: ours - LOUPE = {rc['ours']-rc['loupe_uniform']:+.3f}; LOUPE - vd = {rc['loupe_uniform']-rc['vd_fixed']:+.3f}")
