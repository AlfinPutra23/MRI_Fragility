"""Classical compressed-sensing (POCS + total-variation) reconstruction baseline for the REAL nnU-Net pipeline.
The reconstruction field's 'recon-then-segment' SOTA: reconstruct the undersampled scan, THEN segment. We build a
fair CS recon (no training data needed) from the clean test images, undersampled @R8 with a fixed per-case mask.
Outputs BOTH the zero-filled and the CS recon from the SAME mask, so segmentation Dice isolates the recon effect.
-> imagesTs_R8zf/ (zero-filled) and imagesTs_R8cs/ (CS-TV recon)."""
import os, glob, sys, numpy as np, nibabel as nib
sys.path.insert(0, "scripts" if os.path.isdir("scripts") else ".")
from kspace import vd_cartesian_mask
try:
    from skimage.restoration import denoise_tv_chambolle as tvd; HAVE_TV = True
except Exception:
    from scipy.ndimage import gaussian_filter; HAVE_TV = False

D = "nnUNet_raw/Dataset501_MRIfrag"; R = 8; NITER = 12; TVW = 0.03
os.makedirs(f"{D}/imagesTs_R8zf", exist_ok=True); os.makedirs(f"{D}/imagesTs_R8cs", exist_ok=True)

def denoise(x):
    return tvd(x, weight=TVW) if HAVE_TV else gaussian_filter(x, 0.7)

def pocs_slice(meas, mask):                       # meas: complex k-space (rows zeroed where not sampled); mask: (H,) bool
    x = np.abs(np.fft.ifft2(np.fft.ifftshift(meas)))
    for _ in range(NITER):
        x = denoise(x)
        K = np.fft.fftshift(np.fft.fft2(x)); K[mask] = meas[mask]      # data consistency: restore measured rows
        x = np.abs(np.fft.ifft2(np.fft.ifftshift(K)))
    return x

cases = sorted(glob.glob(f"{D}/imagesTs_clean/*_0000.nii.gz"))
print(f"CS recon ({'TV-POCS' if HAVE_TV else 'gaussian-POCS'}) on {len(cases)} cases, R={R}, {NITER} iters")
for idx, cp in enumerate(cases):
    tag = os.path.basename(cp)
    if os.path.exists(f"{D}/imagesTs_R8cs/{tag}"):        # resume
        continue
    im = nib.load(cp); vol = np.asanyarray(im.dataobj).astype(np.float32)
    v = np.moveaxis(vol, 2, 0)                            # slices-first (match undersample_volume slice_axis=2)
    H = v.shape[1]                                        # pe axis (=axis1 here, undersample_slice pe_axis=0 on the slice)
    mask = vd_cartesian_mask(H, R, seed=idx).astype(bool)  # fixed per-case mask
    zf = np.empty_like(v); cs = np.empty_like(v)
    for i in range(v.shape[0]):
        sl = v[i]; K = np.fft.fftshift(np.fft.fft2(sl)); meas = np.zeros_like(K); meas[mask] = K[mask]
        zf[i] = np.abs(np.fft.ifft2(np.fft.ifftshift(meas)))
        cs[i] = pocs_slice(meas, mask)
    zf = np.moveaxis(zf, 0, 2).astype(np.float32); cs = np.moveaxis(cs, 0, 2).astype(np.float32)
    nib.save(nib.Nifti1Image(zf, im.affine, im.header), f"{D}/imagesTs_R8zf/{tag}")
    nib.save(nib.Nifti1Image(cs, im.affine, im.header), f"{D}/imagesTs_R8cs/{tag}")
    if idx % 20 == 0: print(f"  {idx+1}/{len(cases)}  {tag}", flush=True)
print("CS recon DONE")
