"""FG-TDR architecture figure (polished network-diagram style) — REAL knee qDESS thumbnails, the REAL fragility spectrum
W(k), 3-D perspective conv blocks, and a ×T unrolled loop. -> outputs/plots/fgtdr_architecture.png"""
import glob, os, numpy as np, h5py, nibabel as nib, sys
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch, Polygon
sys.path.insert(0, "scripts"); from kspace import vd_cartesian_mask
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"]})
LAB = {1: "patellar", 2: "femoral", 3: "tibial-med", 4: "tibial-lat", 5: "menisc-med", 6: "menisc-lat"}

# ---- real knee data (SKM-TEA) ----
h5 = glob.glob("data/skmtea/kspace/**/MTR_020.h5", recursive=True)[0]
seg = np.asanyarray(nib.load("data/skmtea/seg/MTR_020_raw-data-track.nii.gz").dataobj).astype(np.int16)
with h5py.File(h5, "r") as f:
    TGT = f["target"]; Z = TGT.shape[2]
    zs = [z for z in range(Z) if (seg[:, :, z] > 0).sum() > 250]
    z = max(zs, key=lambda z: len(np.unique(seg[:, :, z])))
    clean = np.abs(TGT[:, :, z, 0, 0]).astype(np.float32)
    Pf = np.zeros((512, 512)); Pa = np.zeros((512, 512)); n = 0
    for zz in zs[::2]:                                            # real W(k): structure vs total power spectrum
        c = np.abs(TGT[:, :, zz, 0, 0]).astype(np.float32); sm = (seg[:, :, zz] > 0).astype(np.float32)
        Pf += np.abs(np.fft.fftshift(np.fft.fft2(c * sm))) ** 2; Pa += np.abs(np.fft.fftshift(np.fft.fft2(c))) ** 2; n += 1
Wk = 1 + 5 * (Pf / n) / (Pa / n + 1e-6); Wk = np.log(Wk); Wk = (Wk - Wk.min()) / (Wk.ptp() + 1e-9)
gtz = seg[:, :, z]; p98 = np.percentile(clean, 98.0) + 1e-6; norm = lambda x: np.clip(x / p98, 0, 1)   # shared window
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

# ---- figure ----
FW, FH = 17.5, 8.4; ASP = FW / FH
RED, DCp, BLU, YEL, GRN, PUR, GOLD = "#c0392b", "#f0c5bd", "#2c7fb8", "#f4c430", "#2ca25f", "#8c6bb1", "#d99000"
fig = plt.figure(figsize=(FW, FH)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
def shade(c, f): c = np.array(mpl.colors.to_rgb(c)); return tuple(np.clip(c * f, 0, 1))
def block3d(x, y, w, h, color, d=0.008, label=None):                        # cooler 3-D bar: shaded top+side + white top-edge highlight
    dy = d * ASP
    ax.add_patch(Polygon([(x, y + h / 2), (x + d, y + h / 2 + dy), (x + w + d, y + h / 2 + dy), (x + w, y + h / 2)], closed=True, fc=shade(color, 1.32), ec="#1f1f1f", lw=.6))
    ax.add_patch(Polygon([(x + w, y - h / 2), (x + w + d, y - h / 2 + dy), (x + w + d, y + h / 2 + dy), (x + w, y + h / 2)], closed=True, fc=shade(color, .58), ec="#1f1f1f", lw=.6))
    ax.add_patch(Rectangle((x, y - h / 2), w, h, fc=color, ec="#1f1f1f", lw=.9))
    ax.plot([x, x + w], [y + h / 2, y + h / 2], color="w", lw=.7, alpha=.45)
    if label: ax.text(x + w / 2 + d / 2, y - h / 2 - 0.013, label, ha="center", va="top", fontsize=6.8, color="#555")
def blocks(x0, y, specs, d=0.008, gap=0.006):
    x = x0
    for s in specs: block3d(x, y, s[1], s[2], s[0], d, s[3] if len(s) > 3 else None); x += s[1] + gap
    return x - gap + d
def thumb(x, y, w, h, img, seg=None, title="", tc="#222", box="#333", crop=True):
    a = fig.add_axes([x, y, w, h]); a.imshow(np.rot90(img[sy, sx] if crop else img), cmap="gray")
    if seg is not None: a.imshow(np.rot90(segrgba(seg[sy, sx])))
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_edgecolor(box); s.set_linewidth(1.6)
    if title: a.set_title(title, fontsize=9.5, fontweight="bold", color=tc, pad=3)
def arrow(x1, y1, x2, y2, col="#2b2b2b", lw=2.4, ls="-", rad=None, ms=16):    # neat, consistent arrows
    kw = dict(arrowstyle="-|>", mutation_scale=ms, lw=lw, color=col, ls=ls, shrinkA=0, shrinkB=0)
    if rad is not None: kw["connectionstyle"] = f"arc3,rad={rad}"
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), **kw))

yM = 0.50; Tw, Th = 0.105, 0.225; ty = yM - Th / 2                            # all main-row thumbnails centred on yM
RB_W, RB_G, N_RB, DEP = 0.015, 0.006, 5, 0.008; DC_W, DC_GAP = 0.032, 0.010
UB_W, UB_G, N_UB = 0.015, 0.006, 7
recon_w = N_RB * RB_W + (N_RB - 1) * RB_G + DEP; reconDC_w = recon_w + DC_GAP + DC_W
unet_w = N_UB * UB_W + (N_UB - 1) * UB_G + DEP
CHAIN_L, CHAIN_R, PAD = 0.018, 0.80, 0.008                                    # solve for ONE uniform gap -> equal picture↔arrow spacing everywhere
W5 = [Tw, reconDC_w, Tw, unet_w, Tw]; GAP = (CHAIN_R - CHAIN_L - sum(W5)) / 4
L = []; _x = CHAIN_L
for w in W5: L.append(_x); _x += w + GAP
def hjoin(xr_, xl_): arrow(xr_ + PAD, yM, xl_ - PAD, yM)                       # arrow filling a gap with equal pad on both sides

# 1  undersampled input (+ k-space beneath)
thumb(L[0], ty, Tw, Th, zf, title="Undersampled input")
thumb(L[0] + (Tw - 0.083) / 2, 0.145, 0.083, 0.18, Kdisp, title="k-space (×R mask)", crop=False)
hjoin(L[0] + Tw, L[1])
# 2  unrolled reconstruction (red 3-D stack) + DC + ×T recurrence
_x = L[1]
for hh in [.30, .25, .30, .25, .30]: block3d(_x, yM, RB_W, hh, RED); _x += RB_W + RB_G
dcx = L[1] + recon_w + DC_GAP
ax.add_patch(FancyBboxPatch((dcx, yM - 0.055), DC_W, 0.11, boxstyle="round,pad=0.002", fc=DCp, ec=RED, lw=1.5))
ax.text(dcx + DC_W / 2, yM, "DC", ha="center", va="center", fontsize=9, fontweight="bold", color=RED)
rcx = L[1] + recon_w / 2
arrow(dcx + DC_W / 2, yM + 0.057, L[1] + 0.008, yM + 0.17, col="#8a8a8a", lw=1.7, rad=-0.42, ms=13)
ax.text(rcx, yM + 0.205, "× T iterations", ha="center", fontsize=9, style="italic", color="#8a8a8a")
ax.text(rcx, yM + 0.255, "Unrolled Reconstruction  R$_\\theta$", ha="center", fontsize=11.5, fontweight="bold")
ax.text(rcx, ty - 0.055, "CNN denoiser  (residual, shared)", ha="center", fontsize=7.8, color="#555")
hjoin(L[1] + reconDC_w, L[2])
# 3  reconstruction x̂
thumb(L[2], ty, Tw, Th, clean_n, title="Reconstruction  $\\hat{x}$")
hjoin(L[2] + Tw, L[3])
# 4  segmentation U-Net (blue enc / yellow bottleneck / green dec, channel dims)
_x = L[3]
for c, h, lab in [(BLU, .30, "32"), (BLU, .24, "64"), (BLU, .18, "128"), (YEL, .12, "256"), (GRN, .18, "128"), (GRN, .24, "64"), (GRN, .30, "32")]:
    block3d(_x, yM, UB_W, h, c, DEP, lab); _x += UB_W + UB_G
ax.text(L[3] + unet_w / 2, yM + 0.255, "Segmentation U-Net  S$_\\phi$", ha="center", fontsize=11.5, fontweight="bold")
hjoin(L[3] + unet_w, L[4])
# 5  predicted segmentation ŝ (+ ground truth above)
thumb(L[4], ty, Tw, Th, zf, gtz, title="Predicted segmentation  $\\hat{s}$")
thumb(L[4], 0.735, Tw, 0.205, clean_n, gtz, title="Ground truth  $s^*$")
# loss box — same uniform gap after ŝ — with s* and W(k) supervision
LB_L = L[4] + Tw + GAP; LB_W = 0.985 - LB_L
ax.add_patch(FancyBboxPatch((LB_L, 0.45), LB_W, 0.17, boxstyle="round,pad=0.004", fc="#fdf3d8", ec="#b8860b", lw=1.8))
ax.text(LB_L + LB_W / 2, 0.535, "LOSS\n$\\lambda_{task}$ Dice-CE($\\hat{s},s^*$)\n$+\\ \\lambda_{freq}\\,\\|\\sqrt{W}(\\mathcal{F}\\hat{x}-\\mathcal{F}x^*)\\|^2$", ha="center", va="center", fontsize=9.3, fontweight="bold")
hjoin(L[4] + Tw, LB_L)                                                        # ŝ -> loss (identical gap to the chain)
arrow(L[4] + Tw * 0.7, 0.735, LB_L + 0.012, 0.62, rad=-0.28, lw=2.0)          # s* -> loss
wkx = 0.42
thumb(wkx, 0.08, 0.10, 0.20, Wk, title="Fragility prior  W(k)  [real]", tc=PUR, box=PUR, crop=False)
arrow(wkx + 0.105, 0.17, LB_L + 0.03, 0.45, col=PUR, lw=2.0, ls="--", rad=-0.16)   # W(k) -> loss (the novel term)
ax.text((wkx + LB_L) / 2 + 0.05, 0.125, "up-weight the frequencies\nfragile organs need", ha="center", fontsize=8.3, style="italic", color=PUR)
# ---- legend key (dashed box, mini 3-D swatches) ----
leg = [(RED, "Unrolled recon  R$_\\theta$"), (DCp, "Data consistency (DC)"), (BLU, "U-Net encoder"), (YEL, "Bottleneck"), (GRN, "U-Net decoder"), (PUR, "Fragility prior  W(k)"), (GOLD, "Task + frequency loss")]
ax.add_patch(FancyBboxPatch((0.03, 0.006), 0.945, 0.052, boxstyle="round,pad=0.004", fc="#fafafa", ec="#888", lw=1.1, ls="--"))
for i, (c, lab) in enumerate(leg):
    xi = 0.05 + i * 0.133; block3d(xi, 0.032, 0.018, 0.021, c, d=0.004); ax.text(xi + 0.026, 0.032, lab, va="center", fontsize=8.3, color="#333")
fig.suptitle("Fragility-Guided Task-Driven Reconstruction (FG-TDR): reconstruct FOR segmentation, steered by the fragility spectrum", fontsize=13.5, fontweight="bold", y=0.99)
for ext in ("png", "svg"): fig.savefig(f"outputs/plots/fgtdr_architecture.{ext}", dpi=150, bbox_inches="tight")   # SVG = editable vector for a design tool
print("wrote outputs/plots/fgtdr_architecture.png + .svg")
