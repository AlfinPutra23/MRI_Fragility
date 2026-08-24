"""Cluster-aware stats (fixes the pseudoreplication weakness). The knee energy->Dice has n=24 points but only 6
INDEPENDENT structures (each x4 R) -> effective n is smaller than 24. We report the naive pooled r AND a CLUSTER
bootstrap CI (resample STRUCTURES, not points) + per-R correlations. We also add organ-level bootstrap CIs to the
abdominal centroid law and the complex-forward law (13 independent organs). Run with base-anaconda (h5py). -> cluster_stats.json"""
import os, glob, json, numpy as np, h5py, nibabel as nib
from scipy.stats import spearmanr, pearsonr

RS = [2, 4, 6, 8]; LAB = {1: "patellar", 2: "femoral", 3: "tibial-med", 4: "tibial-lat", 5: "menisc-med", 6: "menisc-lat"}


def vd(W, R, acs=0.08, seed=0):
    r = np.random.RandomState(seed); m = np.zeros(W, bool); c = W // 2; na = max(1, int(acs * W)); m[c - na // 2:c + na // 2 + 1] = True
    fr = np.abs(np.arange(W) - c); p = 1.0 / (fr + 1); p[m] = 0; p /= p.sum(); m[r.choice(W, W // R - m.sum(), False, p=p)] = True
    return m


MASKS = {R: vd(512, R) for R in RS}


def energy_removed(c, M, mask):
    Fo = np.fft.fftshift(np.fft.fft2(c * M)); Fo[:, mask] = 0
    return float(np.sqrt((np.abs(np.fft.ifft2(np.fft.ifftshift(Fo)))[M] ** 2).sum()))


def boot_ci(x, y, n=3000, method="pearson"):
    rng = np.random.RandomState(0); b = []
    f = (lambda a, c: pearsonr(a, c)[0]) if method == "pearson" else (lambda a, c: spearmanr(a, c).correlation)
    for _ in range(n):
        i = rng.choice(len(x), len(x), replace=True)
        if len(set(y[i])) > 1: b.append(f(x[i], y[i]))
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


# ---- recompute knee energy per (structure,R) on the 12 test cases, keep structure labels ----
h5s = sorted(glob.glob("data/skmtea/kspace/**/*.h5", recursive=True))
cases = [(h, f"data/skmtea/seg/{os.path.basename(h)[:-3]}_raw-data-track.nii.gz") for h in h5s]
cases = [(h, s) for h, s in cases if os.path.exists(s)]
test_i = set(np.random.RandomState(0).permutation(len(cases))[:12].tolist())
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

d = json.load(open("outputs/results/knee_law.json")); per = d["per_structure_dice"]
struct, X, Y = [], [], []
for k in LAB:
    for R in RS:
        if ener[k][R]:
            struct.append(k); X.append(np.log(np.mean(ener[k][R]))); Y.append(per[LAB[k]]["clean"] - per[LAB[k]][f"R{R}"])
struct, X, Y = np.array(struct), np.array(X), np.array(Y)
pooled = float(spearmanr(X, Y).correlation)

# cluster bootstrap over the 6 structures (the honest effective unit)
rng = np.random.RandomState(0); boot = []
for _ in range(3000):
    samp = rng.choice(list(LAB), len(LAB), replace=True)
    xi = np.concatenate([X[struct == k] for k in samp]); yi = np.concatenate([Y[struct == k] for k in samp])
    if len(set(yi)) > 1: boot.append(spearmanr(xi, yi).correlation)
cl_ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
perR = {f"R{R}": float(spearmanr(np.array([np.log(np.mean(ener[k][R])) for k in LAB if ener[k][R]]),
                                 np.array([per[LAB[k]]["clean"] - per[LAB[k]][f"R{R}"] for k in LAB if ener[k][R]])).correlation)
        for R in RS}

out = {"knee_energy_to_dice": {"pooled_spearman_n24": pooled, "n_independent_structures": 6,
                               "cluster_bootstrap_95CI_over_structures": cl_ci, "per_R_spearman_n6": perR}}

# ---- organ-level bootstrap CIs for the headline laws (13 independent organs) ----
rows = json.load(open("outputs/results/m0_law_v2.json"))["rows"]
cent = np.array([r["centroid"] for r in rows]); drop = np.array([r["drop"] for r in rows])
out["abdominal_law_centroid"] = {"pearson_n13": float(pearsonr(cent, drop)[0]), "bootstrap_95CI": boot_ci(cent, drop)}
try:
    cx = json.load(open("outputs/results/complex_compare.json"))["per_organ"]
    orgs = [r["organ"] for r in rows if r["organ"] in cx]
    c2 = np.array([r["centroid"] for r in rows if r["organ"] in cx]); dd = np.array([cx[o]["cx_drop"] for o in orgs])
    out["complex_forward_law"] = {"pearson": float(pearsonr(c2, dd)[0]), "bootstrap_95CI": boot_ci(c2, dd)}
except Exception as e:
    out["complex_forward_law"] = f"skip: {e}"

json.dump(out, open("outputs/results/cluster_stats.json", "w"), indent=2)
print(json.dumps(out, indent=2))
