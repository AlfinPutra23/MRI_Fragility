"""Export clean, high-res, chrome-free panel images so the Figure 1 layout can be composed by hand in a design tool
(Figma / Illustrator / PowerPoint / Keynote). No titles, no borders, no arrows -- just the images. -> outputs/plots/figure1_panels/"""
import os, sys, numpy as np
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt, matplotlib.colors as mcolors
sys.path.insert(0, "scripts" if os.path.isdir("scripts") else "."); import labels as L
from make_schematics_visual import pick_slice, norm, kspace_log, ORG_COL
from kspace import vd_cartesian_mask, undersample_slice

OUT = "outputs/plots/figure1_panels"; os.makedirs(OUT, exist_ok=True)
img, lab = pick_slice(); n = img.shape[0]; R = 8
m = vd_cartesian_mask(n, R); ali = undersample_slice(img, m, pe_axis=0)
Km = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(img)) * m[:, None]))

def save(arr, name, cmap=None, rgb=False, interp="bilinear"):
    fig = plt.figure(figsize=(6, 6)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.imshow(arr, cmap=(None if rgb else cmap), interpolation=interp)
    fig.savefig(f"{OUT}/{name}.png", dpi=300, pad_inches=0); plt.close(fig)

save(norm(img), "01_clean_mri", "gray")
save(kspace_log(img), "02_kspace", "magma")
save(Km, "03_undersampled_kspace", "magma")
save(norm(ali), "04_fast_image_R8", "gray")

# 05: segmentation composite (colored organs blended on the MRI)
base = norm(img); rgb = np.stack([base] * 3, -1); alpha = 0.55
for o in L.ABDO:
    mk = lab == o
    if mk.sum() < 8: continue
    rgb[mk] = (1 - alpha) * rgb[mk] + alpha * np.array(mcolors.to_rgb(ORG_COL.get(o, "#ff0000")))
save(np.clip(rgb, 0, 1), "05_segmentation", rgb=True, interp="nearest")

# 05b: masks ONLY on a transparent background (overlay it yourself, at any opacity)
rgba = np.zeros((n, n, 4))
for o in L.ABDO:
    mk = lab == o
    if mk.sum() < 8: continue
    rgba[mk] = (*mcolors.to_rgb(ORG_COL.get(o, "#ff0000")), 0.9)
plt.imsave(f"{OUT}/05b_masks_transparent.png", rgba)

# 06: the 1-D sampling mask (which k-space lines are kept) as a strip
plt.imsave(f"{OUT}/06_sampling_mask.png", np.tile(m[:, None].astype(float), (1, 40)), cmap="gray")

print("wrote clean panels to", OUT)
for f in sorted(os.listdir(OUT)):
    print("  ", f)
