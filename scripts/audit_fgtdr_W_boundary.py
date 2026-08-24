"""AUDIT-2 (CPU): can a BETTER-POINTED fragility prior be discriminative on knee?
Compare the structure energy-share Pf/Pa across radial frequency for two priors:
  (a) REGION mask  sm = (seg>0)            -- what condseg_knee_fgtdr.py actually uses (we saw it peaks LOW)
  (b) BOUNDARY mask (structure edges)      -- edges are high-frequency by construction
If (b)'s share RISES with frequency (high/low ratio >> 1), the fragility idea is sound and it's the *prior
formulation* that was wrong on knee, not the idea. Pure numpy/scipy, no GPU."""
import glob, os, numpy as np, h5py, nibabel as nib
from scipy.ndimage import binary_erosion
cases = sorted(glob.glob("data/skmtea/kspace/**/*.h5", recursive=True))[:6]
n = 512; c = n // 2
yy, xx = np.mgrid[:n, :n]; rho = np.sqrt((yy - c) ** 2 + (xx - c) ** 2) / c
Pa = np.zeros((n, n)); Preg = np.zeros((n, n)); Pbnd = np.zeros((n, n)); k = 0
for h in cases:
    sp = f"data/skmtea/seg/{os.path.basename(h)[:-3]}_raw-data-track.nii.gz"
    if not os.path.exists(sp): continue
    seg = np.asanyarray(nib.load(sp).dataobj).astype(np.int16)
    with h5py.File(h, "r") as f:
        T = f["target"]
        for z in [z for z in range(T.shape[2]) if (seg[:, :, z] > 0).sum() > 250][::4]:
            mag = np.abs(T[:, :, z, 0, 0]).astype(np.float32)
            reg = (seg[:, :, z] > 0)
            bnd = reg & ~binary_erosion(reg, iterations=2)          # structure edges (thin structs stay whole)
            Pa += np.abs(np.fft.fftshift(np.fft.fft2(mag))) ** 2
            Preg += np.abs(np.fft.fftshift(np.fft.fft2(mag * reg))) ** 2
            Pbnd += np.abs(np.fft.fftshift(np.fft.fft2(mag * bnd))) ** 2; k += 1
fr_reg = Preg / (Pa + 1e-9); fr_bnd = Pbnd / (Pa + 1e-9)
def profile(fr, tag):
    edges = np.linspace(0, 1.0, 9); prof = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (rho >= a) & (rho < b); prof.append(fr[m].mean())
    lo = fr[rho < 0.15].mean(); hi = fr[(rho >= 0.15) & (rho < 1.0)].mean()
    print(f"\n{tag}")
    print("  band:  " + "  ".join(f"{a:.2f}" for a in edges[:-1]))
    print("  share: " + "  ".join(f"{p:.3f}" for p in prof))
    print(f"  low(<0.15)={lo:.3f}  high(0.15-1)={hi:.3f}  ->  HIGH/LOW ratio = {hi/lo:.2f}")
    return hi / lo
print(f"n_slices={k}")
r1 = profile(fr_reg, "(a) REGION mask  [current method]")
r2 = profile(fr_bnd, "(b) BOUNDARY mask  [proposed fix]")
print(f"\n=> region high/low={r1:.2f}   boundary high/low={r2:.2f}   improvement x{r2/r1:.1f}")
print("VERDICT:", "boundary prior IS discriminative toward high-freq -> idea works with better prior" if r2 > 1.2
      else "boundary prior also flat -> knee genuinely lacks high-freq-fragile structure")
import json as _json                                        # reproducibility: dump the numbers the site/plot cite (were hardcoded)
_json.dump({"region_aim_ratio": round(float(r1), 3), "boundary_aim_ratio": round(float(r2), 3),
            "improvement_x": round(float(r2 / r1), 2), "n_slices": int(k)},
           open("outputs/results/W_audit.json", "w"), indent=2)
print("wrote outputs/results/W_audit.json")
