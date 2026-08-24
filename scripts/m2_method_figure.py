"""Method-result figure: per-organ effect of fragility-weighting @R8 + the acceleration-specific crossover.
Uses the seed-matched sweep preds (Uniform_s42 vs FragW4_s42). -> outputs/plots/m2_method.png"""
import glob, os, numpy as np, nibabel as nib, sys
from scipy.stats import wilcoxon
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "scripts" if os.path.isdir("scripts") else "."); import labels as L
D = "nnUNet_raw/Dataset501_MRIfrag" if os.path.isdir("nnUNet_raw") else "../nnUNet_raw/Dataset501_MRIfrag"
U = "predsSW_nnUNetTrainer_Uniform_s42"; W = "predsSW_nnUNetTrainer_FragW4_s42"


def dice(a, b):
    s = a.sum() + b.sum(); return 2 * np.logical_and(a, b).sum() / s if s else np.nan


def collect(tag):
    """per-organ per-case dice for U and W at tag."""
    du = {o: {} for o in L.ABDO}; dw = {o: {} for o in L.ABDO}
    for gp in sorted(glob.glob(f"{D}/labelsTs/*.nii.gz")):
        c = os.path.basename(gp)[:-7]
        pu, pw = f"{D}/{U}_{tag}/{c}.nii.gz", f"{D}/{W}_{tag}/{c}.nii.gz"
        if not (os.path.exists(pu) and os.path.exists(pw)):
            continue
        g = np.asanyarray(nib.load(gp).dataobj).astype(np.int16)
        au = np.asanyarray(nib.load(pu).dataobj).astype(np.int16)
        aw = np.asanyarray(nib.load(pw).dataobj).astype(np.int16)
        for o in L.ABDO:
            if (g == o).sum() >= 30:
                du[o][c] = dice(au == o, g == o); dw[o][c] = dice(aw == o, g == o)
    return du, dw


cu8, cw8 = collect("R8"); cu1, cw1 = collect("clean")

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.4), gridspec_kw={"width_ratios": [1.7, 1]})

# Panel A: per-organ Δ @R8, sorted, tail red / large blue
rows = []
for o, nm in L.ABDO.items():
    cc = sorted(set(cu8[o]) & set(cw8[o]))
    if len(cc) < 5: continue
    a = np.array([cu8[o][c] for c in cc]); b = np.array([cw8[o][c] for c in cc])
    p = wilcoxon(a, b).pvalue if np.any(a != b) else 1.0
    rows.append((nm, o in L.TAIL, b.mean() - a.mean(), p))
rows.sort(key=lambda r: r[2])
y = np.arange(len(rows))
axA.barh(y, [r[2] for r in rows], color=["#d93025" if r[1] else "#2c5fb0" for r in rows])
for i, (nm, t, d, p) in enumerate(rows):
    star = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else ""
    axA.text(d + (0.001 if d >= 0 else -0.001), i, f"{nm} {star}", va="center",
             ha="left" if d >= 0 else "right", fontsize=8.5)
axA.axvline(0, color="k", lw=0.8); axA.set_yticks([]); axA.set_xlim(-0.018, 0.052)
axA.set_xlabel("Δ Dice @R8  (fragility-weighted − uniform)")
axA.set_title("Per-organ effect @R8: fragile organs (red) gain,\nsolved organs (blue) ~unchanged",
              fontsize=10.5, fontweight="bold")

# Panel B: acceleration-specific crossover — mean TAIL/LARGE Δ at clean vs R8
def grp_delta(cu, cw, ids):
    cc = sorted(set.intersection(*[set(cu[o]) & set(cw[o]) for o in ids]))
    per = [np.mean([cw[o][c] - cu[o][c] for o in ids if c in cu[o]]) for c in cc]
    return np.mean(per), np.std(per) / np.sqrt(len(per))
tail = list(L.TAIL); large = [o for o in L.ABDO if o not in L.TAIL]
t1, t1e = grp_delta(cu1, cw1, tail); t8, t8e = grp_delta(cu8, cw8, tail)
l1, l1e = grp_delta(cu1, cw1, large); l8, l8e = grp_delta(cu8, cw8, large)
axB.errorbar([1, 8], [t1, t8], yerr=[t1e, t8e], fmt="-o", color="#d93025", lw=2.5, capsize=4, label="tail organs")
axB.errorbar([1, 8], [l1, l8], yerr=[l1e, l8e], fmt="-o", color="#2c5fb0", lw=2.5, capsize=4, label="large organs")
axB.axhline(0, color="k", lw=0.8, ls=":")
axB.set_xticks([1, 8]); axB.set_xticklabels(["R=1\n(clean)", "R=8"]); axB.set_xlim(0.3, 8.7)
axB.set_ylabel("Δ Dice (weighted − uniform)")
axB.set_title(f"Acceleration-specific:\nhurts at clean ({t1:+.3f}), helps at R8 ({t8:+.3f})",
              fontsize=10.5, fontweight="bold")
axB.legend(fontsize=9); axB.grid(alpha=0.3)

fig.suptitle("Anatomy-prior fragility-weighting recovers fragile organs, specifically under acceleration",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = "outputs/plots/m2_method.png" if os.path.isdir("outputs") else "../outputs/plots/m2_method.png"
fig.savefig(out, dpi=140); print(f"wrote {out}")
print(f"tail Δ: clean {t1:+.4f}  R8 {t8:+.4f}  |  large Δ: clean {l1:+.4f}  R8 {l8:+.4f}")
