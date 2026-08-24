"""Canonical retrospective k-space undersampling for magnitude MRI (validated 2026-06-28).

Model: 2D multi-slice Cartesian acquisition. For each axial slice:
    slice -> 2D FFT -> variable-density 1D phase-encode mask (+ fully-sampled ACS center) -> IFFT -> |recon|
One fixed mask per volume (a single acquisition), applied to every slice (realistic for 2D multi-slice).

Validated curve (mean over 30 slices): R2 SSIM .970/38dB, R4 .855/31dB, R6 .777/28dB, R8 .745/27dB.
"""
import numpy as np


def vd_cartesian_mask(n, R, acs_frac=0.08, poly=8, seed=0):
    """1D variable-density phase-encode mask (bool, length n) with fully-sampled ACS center.
    R<=1 returns all-True (fully sampled)."""
    if R is None or R <= 1:
        return np.ones(n, bool)
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


def undersample_slice(img2d, mask1d, pe_axis=0):
    """Retrospective zero-filled reconstruction of one 2D magnitude slice."""
    k = np.fft.fftshift(np.fft.fft2(img2d))
    m = mask1d[:, None] if pe_axis == 0 else mask1d[None, :]
    return np.abs(np.fft.ifft2(np.fft.ifftshift(k * m))).astype(img2d.dtype)


def undersample_volume(vol, R, seed=0, slice_axis=2, pe_axis=0):
    """Apply per-slice undersampling along slice_axis with ONE fixed mask (single acquisition).
    vol: 3D float array. Returns recon volume (same shape/dtype). R<=1 -> copy."""
    if R is None or R <= 1:
        return vol.copy()
    v = np.moveaxis(vol, slice_axis, 0)           # slices first
    n_pe = v.shape[1 + pe_axis]
    mask = vd_cartesian_mask(n_pe, R, seed=seed)
    out = np.empty_like(v)
    for i in range(v.shape[0]):
        out[i] = undersample_slice(v[i], mask, pe_axis=pe_axis)
    return np.moveaxis(out, 0, slice_axis)


def effective_R(n, R):
    return n / int(vd_cartesian_mask(n, R).sum())
