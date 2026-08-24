"""AUDIT: is the FG-TDR fragility spectrum W(k) actually DISCRIMINATIVE on the knee test data, or ~uniform?
If structures' energy share (Pf/Pa) does NOT rise with radial frequency, the '√W freq loss' is just a generic
image-quality regularizer — i.e. the *fragility* part isn't the active ingredient on this data."""
import glob, os, numpy as np, h5py, nibabel as nib
cases = sorted(glob.glob("data/skmtea/kspace/**/*.h5", recursive=True))[:6]
n = 512; c = n // 2
yy, xx = np.mgrid[:n, :n]; rho = np.sqrt((yy - c) ** 2 + (xx - c) ** 2) / c        # 0=DC .. ~1.4 corner
Pf = np.zeros((n, n)); Pa = np.zeros((n, n)); k = 0
for h in cases:
    seg_p = f"data/skmtea/seg/{os.path.basename(h)[:-3]}_raw-data-track.nii.gz"
    if not os.path.exists(seg_p): continue
    seg = np.asanyarray(nib.load(seg_p).dataobj).astype(np.int16)
    with h5py.File(h, "r") as f:
        T = f["target"]; Z = T.shape[2]
        zs = [z for z in range(Z) if (seg[:, :, z] > 0).sum() > 250][::4]
        for z in zs:
            mag = np.abs(T[:, :, z, 0, 0]).astype(np.float32); sm = (seg[:, :, z] > 0).astype(np.float32)
            Pf += np.abs(np.fft.fftshift(np.fft.fft2(mag * sm))) ** 2
            Pa += np.abs(np.fft.fftshift(np.fft.fft2(mag))) ** 2; k += 1
frac = Pf / (Pa + 1e-9)                                    # structure's SHARE of energy at each frequency (the core of W)
W = 1 + 4.0 * frac; Wn = W / W.mean()
print(f"n_slices={k}   W(k) after norm:  min={Wn.min():.3f}  mean={Wn.mean():.3f}  max={Wn.max():.3f}  "
      f"p95/p50={np.percentile(Wn,95)/np.percentile(Wn,50):.2f}")
print("\nRadial profile — does the STRUCTURE energy-share rise with frequency? (the whole premise)")
edges = np.linspace(0, 1.0, 9)
print(f"  {'radial freq band':>18} | {'struct share Pf/Pa':>18} | {'mean W(norm)':>12}")
for a, b in zip(edges[:-1], edges[1:]):
    m = (rho >= a) & (rho < b)
    print(f"  {f'{a:.2f}-{b:.2f}':>18} | {frac[m].mean():>18.4f} | {Wn[m].mean():>12.3f}")
lo = (rho < 0.15); hi = (rho >= 0.15) & (rho < 1.0)
print(f"\nlow-freq (<0.15) struct-share={frac[lo].mean():.3f}   high-freq (0.15-1) struct-share={frac[hi].mean():.3f}")
print(f"=> high/low share ratio = {frac[hi].mean()/frac[lo].mean():.2f}   (>>1 = fragility signal; ~1 = W is ~generic)")
