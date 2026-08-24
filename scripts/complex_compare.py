"""Does the abdominal fragility SURVIVE a realistic complex multicoil forward model? Compares per-organ Dice-drop under
the complex forward (cx_fragility_dice.json) vs the magnitude-FFT benchmark (m0_law_v2.json rows), and re-tests the
centroid law on the complex-forward drops. -> outputs/results/complex_compare.json + outputs/plots/complex_forward.png"""
import json, numpy as np
from scipy.stats import spearmanr, pearsonr
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt

rows = json.load(open("outputs/results/m0_law_v2.json"))["rows"]
mag = {r["organ"]: {"drop": r["drop"], "centroid": r["centroid"], "tail": r["tail"]} for r in rows}
cx = json.load(open("outputs/results/cx_fragility_dice.json"))

organs = [o for o in mag if o in cx]
mag_drop = np.array([mag[o]["drop"] for o in organs])
cx_drop = np.array([cx[o]["R1"] - cx[o]["R8"] for o in organs])   # clean(=R1) -> complex-R8
cent = np.array([mag[o]["centroid"] for o in organs])
tail = np.array([mag[o]["tail"] for o in organs])

order_sp = float(spearmanr(mag_drop, cx_drop).correlation)         # is the fragility ORDERING preserved?
law_mag = float(pearsonr(cent, mag_drop)[0])                        # centroid -> drop (magnitude benchmark)
law_cx_p = float(pearsonr(cent, cx_drop)[0])                        # centroid -> drop (complex forward)
law_cx_sp = float(spearmanr(cent, cx_drop).correlation)
out = {"n_organs": len(organs), "ordering_mag_vs_cx_spearman": order_sp,
       "centroid_law_magnitude_pearson": law_mag, "centroid_law_complex_pearson": law_cx_p,
       "centroid_law_complex_spearman": law_cx_sp,
       "per_organ": {o: {"mag_drop": float(mag[o]["drop"]), "cx_drop": float(cx[o]["R1"] - cx[o]["R8"]),
                         "centroid": float(mag[o]["centroid"])} for o in organs}}
json.dump(out, open("outputs/results/complex_compare.json", "w"), indent=2)
print(json.dumps({k: v for k, v in out.items() if k != "per_organ"}, indent=2))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
ax[0].scatter(mag_drop[~tail], cx_drop[~tail], c="#377eb8", label="large", s=60)
ax[0].scatter(mag_drop[tail], cx_drop[tail], c="#d73027", label="tail", s=60)
lim = [0, max(mag_drop.max(), cx_drop.max()) * 1.1]; ax[0].plot(lim, lim, "k--", alpha=.4)
for o in organs: ax[0].annotate(o, (mag[o]["drop"], cx[o]["R1"] - cx[o]["R8"]), fontsize=6)
ax[0].set_xlabel("magnitude-FFT Dice drop"); ax[0].set_ylabel("complex multicoil Dice drop")
ax[0].set_title(f"Fragility ordering survives\nSpearman {order_sp:.2f}", fontweight="bold"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].scatter(cent[~tail], cx_drop[~tail], c="#377eb8", s=60); ax[1].scatter(cent[tail], cx_drop[tail], c="#d73027", s=60)
for o in organs: ax[1].annotate(o, (mag[o]["centroid"], cx[o]["R1"] - cx[o]["R8"]), fontsize=6)
ax[1].set_xlabel("spectral centroid (anatomy)"); ax[1].set_ylabel("complex multicoil Dice drop")
ax[1].set_title(f"Centroid law holds under complex forward\nPearson {law_cx_p:.2f} (magnitude was {law_mag:.2f})", fontweight="bold"); ax[1].grid(alpha=.3)
fig.suptitle("Abdominal fragility under a REALISTIC complex multicoil forward model (A=M·𝓕·S: phase + coils + noise + RSS)", fontweight="bold")
fig.tight_layout(); fig.savefig("outputs/plots/complex_forward.png", dpi=140, bbox_inches="tight"); plt.close(fig)
print("wrote outputs/plots/complex_forward.png")
