"""#2 domain-matched recon baseline (fair W3 opponent): reconstruct the TRAINING set to CS-recon (imagesTr ->
imagesTr_R8cs) so we can train a segmenter ON the recon distribution (not clean-trained). Same TV-POCS recon as
recon_cs.py (R8, per-case fixed VD mask). CPU, resumable. Run with mrifrag (skimage+nibabel)."""
import os, glob, sys, numpy as np, nibabel as nib
sys.path.insert(0, "scripts"); from kspace import vd_cartesian_mask
from skimage.restoration import denoise_tv_chambolle as tvd

D = "nnUNet_raw/Dataset501_MRIfrag"; R = 8; NITER = 12; TVW = 0.03
os.makedirs(f"{D}/imagesTr_R8cs", exist_ok=True)


def pocs_slice(meas, mask):
    x = np.abs(np.fft.ifft2(np.fft.ifftshift(meas)))
    for _ in range(NITER):
        x = tvd(x, weight=TVW)
        K = np.fft.fftshift(np.fft.fft2(x)); K[mask] = meas[mask]        # data consistency
        x = np.abs(np.fft.ifft2(np.fft.ifftshift(K)))
    return x


cases = sorted(glob.glob(f"{D}/imagesTr/*_0000.nii.gz"))
print(f"CS recon on {len(cases)} TRAIN cases (R={R}, {NITER} iters)", flush=True)
for idx, cp in enumerate(cases):
    tag = os.path.basename(cp)
    if os.path.exists(f"{D}/imagesTr_R8cs/{tag}"): continue          # resume
    im = nib.load(cp); vol = np.asanyarray(im.dataobj).astype(np.float32)
    v = np.moveaxis(vol, 2, 0); H = v.shape[1]
    mask = vd_cartesian_mask(H, R, seed=idx).astype(bool)
    cs = np.empty_like(v)
    for i in range(v.shape[0]):
        sl = v[i]; K = np.fft.fftshift(np.fft.fft2(sl)); meas = np.zeros_like(K); meas[mask] = K[mask]
        cs[i] = pocs_slice(meas, mask)
    cs = np.moveaxis(cs, 0, 2).astype(np.float32)
    nib.save(nib.Nifti1Image(cs, im.affine, im.header), f"{D}/imagesTr_R8cs/{tag}")
    if idx % 20 == 0: print(f"  {idx+1}/{len(cases)} {tag}", flush=True)
print("TRAIN CS recon DONE", flush=True)
