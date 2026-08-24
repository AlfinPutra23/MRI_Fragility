"""M0 groundwork (no GPU): retrospective k-space undersampling simulator + visual/metric sanity.

Magnitude MRI is valid for retrospective Cartesian undersampling:
    image -> 2D FFT -> variable-density 1D phase-encode mask (+ fully-sampled ACS center) -> IFFT -> |recon|
We sweep R in {2,4,6,8}, render a small-organ slice, and report global SSIM/PSNR vs R.
This sets up the *metric-blindness* test: image metrics stay high while (later) small-organ Dice will crater.

Run: <python-with-skimage> code/m0_kspace_sim.py
"""
import os, glob
import numpy as np
import nibabel as nib
from skimage.metrics import structural_similarity as ssim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import MRISEG_REL as REL, PLOTS as OUT_P, RESULTS as OUT_R
RS = [2, 4, 6, 8]
PANCREAS = 11


def vd_cartesian_mask(n, R, acs_frac=0.08, poly=8, seed=0):
    """1D variable-density phase-encode mask with fully-sampled ACS center."""
    rng = np.random.default_rng(seed)
    c = n // 2
    n_acs = max(int(round(n * acs_frac)), 4)
    mask = np.zeros(n, bool)
    mask[c - n_acs // 2: c + (n_acs - n_acs // 2)] = True
    n_keep = int(round(n / R))
    remaining = max(n_keep - int(mask.sum()), 0)
    x = (np.arange(n) - c) / c
    pdf = (1 - np.abs(x)) ** poly
    pdf[mask] = 0
    if pdf.sum() > 0 and remaining > 0:
        pdf = pdf / pdf.sum()
        idx = rng.choice(n, size=min(remaining, int((pdf > 0).sum())), replace=False, p=pdf)
        mask[idx] = True
    return mask


def undersample(img2d, mask1d, pe_axis=0):
    k = np.fft.fftshift(np.fft.fft2(img2d))
    m = mask1d[:, None] if pe_axis == 0 else mask1d[None, :]
    rec = np.fft.ifft2(np.fft.ifftshift(k * m))
    return np.abs(rec)


def psnr(ref, x):
    mse = np.mean((ref - x) ** 2)
    return 99.0 if mse == 0 else 20 * np.log10(ref.max() / np.sqrt(mse))


def pick_demo():
    """volume + axial slice index with the largest pancreas cross-section."""
    for lp in sorted(glob.glob(f"{REL}/labelsTr/train_0*_VEN.nii.gz"))[:25]:
        la = np.asanyarray(nib.load(lp).dataobj).astype(np.int16)
        area = (la == PANCREAS).sum(axis=(0, 1))
        if area.max() > 200:
            ip = lp.replace("labelsTr", "ImageTr").replace("_VEN.nii.gz", "_VEN_0000.nii.gz")
            return ip, lp, int(area.argmax())
    raise RuntimeError("no pancreas slice found")


def main():
    os.makedirs(OUT_P, exist_ok=True); os.makedirs(OUT_R, exist_ok=True)
    ip, lp, z = pick_demo()
    img = np.asanyarray(nib.load(ip).dataobj).astype(np.float32)
    lab = np.asanyarray(nib.load(lp).dataobj).astype(np.int16)
    sl = np.rot90(img[:, :, z]); gt = np.rot90(lab[:, :, z] == PANCREAS)
    n = img.shape[0]  # undersampling axis on the UN-rotated slice (rot90 is display-only)

    # ---- metric curve averaged over many slices/volumes ----
    print("computing SSIM/PSNR vs R over 30 slices...")
    curve = {R: {"ssim": [], "psnr": []} for R in RS}
    vols = sorted(glob.glob(f"{REL}/ImageTr/train_0*_VEN_0000.nii.gz"))[:6]
    for vp in vols:
        v = np.asanyarray(nib.load(vp).dataobj).astype(np.float32)
        zc = v.shape[2] // 2
        for zz in range(zc - 3, zc + 3):
            ref = v[:, :, zz]
            if ref.max() <= 0:
                continue
            for R in RS:
                m = vd_cartesian_mask(ref.shape[0], R, seed=zz)
                rec = undersample(ref, m)
                dr = ref.max()
                curve[R]["ssim"].append(ssim(ref, rec, data_range=dr))
                curve[R]["psnr"].append(psnr(ref, rec))
    summ = {R: dict(ssim=float(np.mean(curve[R]["ssim"])), psnr=float(np.mean(curve[R]["psnr"])),
                    eff_R=float(n / vd_cartesian_mask(n, R).sum())) for R in RS}
    print("\n=== global image metrics vs R (mean over 30 slices) ===")
    print(f"{'R':>3} {'eff_R':>6} {'SSIM':>7} {'PSNR(dB)':>9}")
    for R in RS:
        print(f"{R:>3} {summ[R]['eff_R']:6.2f} {summ[R]['ssim']:7.3f} {summ[R]['psnr']:9.2f}")
    import json
    json.dump(summ, open(f"{OUT_R}/m0_kspace_metrics.json", "w"), indent=2)
    print(f"\nwrote {OUT_R}/m0_kspace_metrics.json")
    print("NOTE: SSIM stays high even at R=8 -> sets up the metric-blindness claim (Dice comes next).")

    # ---- demo figure: full + each R, full-FOV (top) and pancreas zoom (bottom) ----
    ys, xs = np.where(gt); m = 45
    y0, y1 = max(ys.min() - m, 0), min(ys.max() + m, sl.shape[0])
    x0, x1 = max(xs.min() - m, 0), min(xs.max() + m, sl.shape[1])
    cols = ["fully sampled"] + [f"R={R}" for R in RS]
    recs = [sl] + [np.rot90(undersample(img[:, :, z], vd_cartesian_mask(n, R, seed=z))) for R in RS]
    vmax = np.percentile(sl, 99.5)
    fig, ax = plt.subplots(2, 5, figsize=(15.5, 6.6))
    for c, (name, rec) in enumerate(zip(cols, recs)):
        ax[0, c].imshow(rec, cmap="gray", vmax=vmax); ax[0, c].axis("off")
        t = name if c == 0 else f"{name}  SSIM {summ[RS[c-1]]['ssim']:.3f}  {summ[RS[c-1]]['psnr']:.1f}dB"
        ax[0, c].set_title(t, fontsize=10, fontweight="bold")
        crop = rec[y0:y1, x0:x1]
        ax[1, c].imshow(crop, cmap="gray", vmax=vmax); ax[1, c].axis("off")
        ax[1, c].contour(gt[y0:y1, x0:x1], colors="#1a9850", linewidths=1.3)
    ax[0, 0].text(-0.08, 0.5, "full FOV", transform=ax[0, 0].transAxes, rotation=90,
                  va="center", ha="right", fontsize=10, fontweight="bold")
    ax[1, 0].text(-0.08, 0.5, "pancreas zoom\n(green = GT)", transform=ax[1, 0].transAxes, rotation=90,
                  va="center", ha="right", fontsize=10, fontweight="bold")
    fig.suptitle("Retrospective k-space undersampling (variable-density Cartesian + ACS) — "
                 "global metrics stay high while small-organ detail blurs", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0.02, 0, 1, 0.96])
    fig.savefig(f"{OUT_P}/m0_kspace_sim.png", dpi=140); plt.close(fig)
    print(f"wrote {OUT_P}/m0_kspace_sim.png  (demo: {os.path.basename(ip)}, slice {z})")


if __name__ == "__main__":
    main()
