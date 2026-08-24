"""Re-plot the W2 knee-law figure (knee_seg.py skipped it: magicnet has no matplotlib). Panels 1+2 come straight from
knee_law.json; panel (b) recomputes the model-INDEPENDENT |energy removed| on the SAME 12 test cases (pure k-space, no
retrain) with Dice-drops from the json. Run with base-anaconda python (h5py 3.8 reads SKM-TEA + has mpl/nibabel/scipy)."""
import os, glob, json, numpy as np, h5py, nibabel as nib
from scipy.stats import spearmanr
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"]})

RS = [2, 4, 6, 8]; LAB = {1: "patellar", 2: "femoral", 3: "tibial-med", 4: "tibial-lat", 5: "menisc-med", 6: "menisc-lat"}
d = json.load(open("outputs/results/knee_law.json"))
per, cent, drop = d["per_structure_dice"], d["centroid"], d["dice_drop_clean_to_R8"]

# reproduce knee_seg's test split (seed 0, ntest 12) EXACTLY
h5s = sorted(glob.glob("data/skmtea/kspace/**/*.h5", recursive=True))
cases = [(h, f"data/skmtea/seg/{os.path.basename(h)[:-3]}_raw-data-track.nii.gz") for h in h5s]
cases = [(h, s) for h, s in cases if os.path.exists(s)]
test_i = set(np.random.RandomState(0).permutation(len(cases))[:12].tolist())


def vd_mask(W, R, acs=0.08, seed=0):
    r = np.random.RandomState(seed); m = np.zeros(W, bool); c = W // 2; na = max(1, int(acs * W)); m[c - na // 2:c + na // 2 + 1] = True
    fr = np.abs(np.arange(W) - c); p = 1.0 / (fr + 1); p[m] = 0; p /= p.sum(); m[r.choice(W, W // R - m.sum(), False, p=p)] = True
    return m


MASKS = {R: vd_mask(512, R) for R in RS}


def energy_removed(cimg, M, mask):
    Fo = np.fft.fftshift(np.fft.fft2(cimg * M)); Fo[:, mask] = 0
    ef = np.abs(np.fft.ifft2(np.fft.ifftshift(Fo))); return float(np.sqrt((ef[M] ** 2).sum()))


ener = {k: {R: [] for R in RS} for k in LAB}
for j, (h, s) in enumerate(cases):
    if j not in test_i: continue
    seg = np.asanyarray(nib.load(s).dataobj).astype(np.int16)
    with h5py.File(h, "r") as f:
        TGT = f["target"]; Z = TGT.shape[2]
        for z in [z for z in range(Z) if (seg[:, :, z] > 0).sum() > 200]:
            c = TGT[:, :, z, 0, 0].astype(np.complex64); lz = seg[:, :, z]
            for k in LAB:
                M = lz == k
                if M.sum() < 60: continue
                for R in RS: ener[k][R].append(energy_removed(c, M, MASKS[R]))

EX, EY = [], []
for k in LAB:
    for R in RS:
        if ener[k][R]:
            EX.append(np.mean(ener[k][R])); EY.append(per[LAB[k]]["clean"] - per[LAB[k]][f"R{R}"])
EX, EY = np.array(EX), np.array(EY)
print("recomputed energy->drop spearman:", round(spearmanr(EX, EY).correlation, 3),
      "(json:", round(d["law_b_energy_to_dicedrop"]["spearman"], 3), ")")

conds = ["clean"] + [f"R{R}" for R in RS]; xR = [0] + RS
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
for k in LAB: ax[0].plot(xR, [per[LAB[k]][c] for c in conds], "-o", label=LAB[k])
ax[0].set_xlabel("acceleration R"); ax[0].set_ylabel("Dice"); ax[0].set_title("Per-structure Dice vs R (real qDESS)", fontweight="bold"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
cx = np.array([cent[LAB[k]] for k in LAB]); dy = np.array([drop[LAB[k]] for k in LAB])
ax[1].scatter(cx, dy, s=80, color="#d73027")
for k in LAB: ax[1].annotate(LAB[k], (cent[LAB[k]], drop[LAB[k]]), fontsize=7)
ax[1].set_xlabel("spectral centroid (anatomy)"); ax[1].set_ylabel("Dice drop clean→R8")
ax[1].set_title(f"(a) centroid → Dice drop\nSpearman {d['law_a_centroid_to_dicedrop']['spearman']:.2f}  (INVERTS on knee)", fontweight="bold"); ax[1].grid(alpha=.3)
ax[2].scatter(EX, EY, s=45, color="#1a9850", alpha=.8)
ax[2].set_xscale("log"); ax[2].set_xlabel("|energy removed| (in-region, log)"); ax[2].set_ylabel("Dice drop clean→R")
ax[2].set_title(f"(b) energy → Dice drop\nSpearman {d['law_b_energy_to_dicedrop']['spearman']:.2f}  (mechanism holds)", fontweight="bold"); ax[2].grid(alpha=.3, which="both")
fig.suptitle(f"SKM-TEA real-k-space law test (n={d['n_cases']} cases, {d['n_test']} test): the MECHANISM→Dice holds; the centroid predictor is abdomen-specific",
             fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig("outputs/plots/knee_law.png", dpi=140, bbox_inches="tight"); plt.close(fig)
print("wrote outputs/plots/knee_law.png")
