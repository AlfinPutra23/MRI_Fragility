"""W1-residual fix: a REALISTIC COMPLEX MULTICOIL forward model for the abdominal benchmark. The current benchmark
undersamples FFT(|magnitude|) -- no phase, coils, or noise. Here we simulate the real acquisition A = M·F·S:
  x (magnitude) -> x·e^{iφ} (smooth image phase) -> per-coil x·e^{iφ}·S_c (Nc simulated sensitivities)
  -> FFT -> + complex k-space noise -> apply VD mask M -> per-coil zero-filled IFFT -> root-sum-of-squares combine.
Produces imagesTs_R{R}_cx (same layout as imagesTs_R{R}) to be segmented by the EXISTING nnU-Net (inference only):
does the per-organ fragility ordering + centroid law survive a proper forward model?  numpy+nibabel (base-anaconda/mrifrag)."""
import os, glob, sys, argparse, numpy as np, nibabel as nib
sys.path.insert(0, "scripts"); from kspace import vd_cartesian_mask

D = "nnUNet_raw/Dataset501_MRIfrag"; RS = [2, 4, 6, 8]; NC = 8


def coil_maps(H, W, nc=NC, seed=0):
    """Nc smooth complex sensitivities on a ring around the FOV, RSS-normalized (sum|S|^2 ~ 1)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32); cy, cx = H / 2, W / 2
    rad = 0.6 * min(H, W); sig = 0.5 * min(H, W)
    S = np.empty((nc, H, W), np.complex64)
    for c in range(nc):
        th = 2 * np.pi * c / nc; y0 = cy + rad * np.sin(th); x0 = cx + rad * np.cos(th)
        mag = np.exp(-(((yy - y0) ** 2 + (xx - x0) ** 2) / (2 * sig ** 2)))
        ph = 0.6 * np.sin(2 * np.pi * yy / H + th) + 0.6 * np.cos(2 * np.pi * xx / W + 2 * th)
        S[c] = (mag * np.exp(1j * ph)).astype(np.complex64)
    return (S / (np.sqrt((np.abs(S) ** 2).sum(0)) + 1e-6)).astype(np.complex64)


def smooth_phase(H, W, rng, amp=1.2):
    """low-order smooth background phase (B0-like): quadratic + a few low-freq sinusoids."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32); yn = (yy - H / 2) / H; xn = (xx - W / 2) / W
    ph = amp * (rng.uniform(-1, 1) * yn + rng.uniform(-1, 1) * xn
                + rng.uniform(-1, 1) * (yn ** 2 - xn ** 2) + rng.uniform(-1, 1) * yn * xn)
    for _ in range(2):
        ph += 0.4 * rng.uniform(-1, 1) * np.sin(2 * np.pi * (rng.integers(1, 4) * yn + rng.integers(1, 4) * xn))
    return ph.astype(np.float32)


def forward_rss(x, S, mask1d, rng, noise_frac):
    """one slice: complex multicoil forward + VD mask + zero-filled per-coil IFFT + RSS combine -> magnitude."""
    xc = (x * np.exp(1j * smooth_phase(*x.shape, rng))).astype(np.complex64)
    kc = np.fft.fftshift(np.fft.fft2(xc[None] * S, axes=(-2, -1)), axes=(-2, -1))          # (nc,H,W)
    if noise_frac > 0:
        s = noise_frac * np.abs(kc).mean()
        kc = kc + s * (rng.standard_normal(kc.shape) + 1j * rng.standard_normal(kc.shape)).astype(np.complex64)
    kc[:, ~mask1d, :] = 0                                                                   # PE = rows (axis0)
    imgc = np.fft.ifft2(np.fft.ifftshift(kc, axes=(-2, -1)), axes=(-2, -1))
    return np.sqrt((np.abs(imgc) ** 2).sum(0)).astype(np.float32)                           # RSS combine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_cases", type=int, default=0, help="cap cases (0=all 240)")
    ap.add_argument("--noise_frac", type=float, default=0.004)
    args = ap.parse_args()
    for R in RS: os.makedirs(f"{D}/imagesTs_R{R}_cx", exist_ok=True)
    cases = sorted(glob.glob(f"{D}/imagesTs_clean/*_0000.nii.gz"))
    if args.n_cases: cases = cases[:args.n_cases]
    print(f"complex multicoil forward on {len(cases)} cases, Nc={NC}, noise_frac={args.noise_frac}", flush=True)
    for idx, cp in enumerate(cases):
        tag = os.path.basename(cp)
        if all(os.path.exists(f"{D}/imagesTs_R{R}_cx/{tag}") for R in RS):   # resume
            continue
        im = nib.load(cp); vol = np.asanyarray(im.dataobj).astype(np.float32)
        v = np.moveaxis(vol, 2, 0)                                            # slices first (H=rows=PE)
        H, W = v.shape[1], v.shape[2]
        S = coil_maps(H, W, seed=idx)
        for R in RS:
            mask = vd_cartesian_mask(H, R, seed=idx)                          # one mask/volume (single acquisition)
            rng = np.random.default_rng(1000 + idx)                          # deterministic per case
            out = np.empty_like(v)
            for i in range(v.shape[0]):
                out[i] = forward_rss(v[i], S, mask, rng, args.noise_frac)
            out = np.moveaxis(out, 0, 2).astype(np.float32)
            nib.save(nib.Nifti1Image(out, im.affine, im.header), f"{D}/imagesTs_R{R}_cx/{tag}")
        if idx % 20 == 0: print(f"  {idx+1}/{len(cases)}  {tag}", flush=True)
    print("complex forward DONE")


if __name__ == "__main__":
    main()
