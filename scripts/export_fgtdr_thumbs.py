"""Export the individual real-knee thumbnails used in the FG-TDR architecture figure, as standalone PNGs, so a design
tool can place them directly. -> outputs/plots/fgtdr_thumbs/*.png"""
import glob, os, numpy as np, h5py, nibabel as nib
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
import sys; sys.path.insert(0, "scripts"); from kspace import vd_cartesian_mask
OUT = "outputs/plots/fgtdr_thumbs"; os.makedirs(OUT, exist_ok=True)

h5 = glob.glob("data/skmtea/kspace/**/MTR_020.h5", recursive=True)[0]
seg = np.asanyarray(nib.load("data/skmtea/seg/MTR_020_raw-data-track.nii.gz").dataobj).astype(np.int16)
with h5py.File(h5, "r") as f:
    TGT = f["target"]; Z = TGT.shape[2]
    zs = [z for z in range(Z) if (seg[:, :, z] > 0).sum() > 250]
    z = max(zs, key=lambda z: len(np.unique(seg[:, :, z])))
    clean = np.abs(TGT[:, :, z, 0, 0]).astype(np.float32)
    Pf = np.zeros((512, 512)); Pa = np.zeros((512, 512)); n = 0
    for zz in zs[::3]:
        c = np.abs(TGT[:, :, zz, 0, 0]).astype(np.float32); sm = (seg[:, :, zz] > 0).astype(np.float32)
        Pf += np.abs(np.fft.fftshift(np.fft.fft2(c * sm))) ** 2; Pa += np.abs(np.fft.fftshift(np.fft.fft2(c))) ** 2; n += 1
Wk = 1 + 5 * (Pf / n) / (Pa / n + 1e-6); Wk = np.log(Wk); Wk = (Wk - Wk.min()) / (Wk.ptp() + 1e-9)
gtz = seg[:, :, z]; p98 = np.percentile(clean, 98.0) + 1e-6; norm = lambda x: np.clip(x / p98, 0, 1)
clean_n = norm(clean)
mask = vd_cartesian_mask(512, 8); K = np.fft.fftshift(np.fft.fft2(clean)); Km = K.copy(); Km[~mask, :] = 0
zf = norm(np.abs(np.fft.ifft2(np.fft.ifftshift(Km)))); Kdisp = np.log(np.abs(Km) + 1); Kdisp = Kdisp / Kdisp.max()
ys, xs = np.where(gtz > 0); m = 55; sy = slice(max(0, ys.min() - m), ys.max() + m); sx = slice(max(0, xs.min() - m), xs.max() + m)
PAL = plt.cm.tab10(np.linspace(0, 1, 10))
def segrgba(s):
    r = np.zeros((*s.shape, 4))
    for lab in np.unique(s):
        if lab == 0: continue
        c = PAL[int(lab) % 10]; r[s == lab] = [c[0], c[1], c[2], 0.85]
    return r

def save(name, img, seg=None, crop=True):
    a = img[sy, sx] if crop else img; fig, ax = plt.subplots(figsize=(3, 3.2)); ax.imshow(np.rot90(a), cmap="gray")
    if seg is not None: ax.imshow(np.rot90(segrgba(seg[sy, sx])))
    ax.axis("off"); fig.savefig(f"{OUT}/{name}.png", dpi=150, bbox_inches="tight", pad_inches=0, transparent=True); plt.close(fig)
    print(f"  {name}.png")

save("1_undersampled_input", zf)
save("2_kspace_mask", Kdisp, crop=False)
save("3_reconstruction", clean_n)
save("4_pred_segmentation", zf, gtz)
save("5_ground_truth", clean_n, gtz)
save("6_fragility_Wk", Wk, crop=False)
print(f"wrote 6 thumbnails -> {OUT}/")
