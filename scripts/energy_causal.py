"""Causal/mechanism test: is fragility GOVERNED by how much of an organ's energy the acquisition removes?
For each organ compute its per-phase-encode-line energy; at each R the variable-density mask discards some lines ->
E_lost(organ,R) = fraction of that organ's energy thrown away. If Dice-drop(organ,R) collapses onto ONE curve of
E_lost across ALL organs AND accelerations, the drop is a function of energy-removed (the mechanism), not organ
identity or R per se. CPU-only, uses existing drops. -> outputs/{results/energy_causal.json, plots/energy_causal.png}
  python energy_causal.py --root <Dataset501> [--max_cases 40]"""
import os, sys, glob, json, argparse, numpy as np, nibabel as nib
from scipy.stats import pearsonr, spearmanr
sys.path.insert(0, "scripts" if os.path.isdir("scripts") else "."); import labels as L
from kspace import vd_cartesian_mask
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from paths import RESULTS as RES, PLOTS as PLT

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--max_cases", type=int, default=40)
ap.add_argument("--patch", type=int, default=96); a = ap.parse_args()
Rs = [2, 4, 6, 8]
frag = json.load(open(f"{RES}/m0_fragility_dice.json"))

def pe_energy(patch):                      # energy per phase-encode line (fftshift-centered), DC removed
    F = np.fft.fftshift(np.fft.fft2(patch)); P = np.abs(F) ** 2; P[patch.shape[0]//2, patch.shape[1]//2] = 0
    return P.sum(axis=1)                    # sum over frequency-encode axis -> length H (PE lines)

cases = sorted(glob.glob(f"{a.root}/labelsTs/*.nii.gz"))
cases = cases[::max(len(cases)//a.max_cases, 1)]
El = {o: {r: [] for r in Rs} for o in L.ABDO}
print(f"energy_causal: {len(cases)} cases")
for gp in cases:
    c = os.path.basename(gp)[:-7]
    img = np.asanyarray(nib.load(f"{a.root}/imagesTs_clean/{c}_0000.nii.gz").dataobj).astype(np.float32)
    lab = np.asanyarray(nib.load(gp).dataobj).astype(np.int16)
    for o in L.ABDO:
        m = lab == o
        if m.sum() < 60: continue
        z = int(np.argmax([(lab[:, :, k] == o).sum() for k in range(lab.shape[2])]))
        m2 = m[:, :, z]
        if m2.sum() < 30: continue
        ys, xs = np.where(m2); cy, cx = (ys.min()+ys.max())//2, (xs.min()+xs.max())//2; h = a.patch//2
        patch = (img[:, :, z] * m2)[max(cy-h, 0):cy+h, max(cx-h, 0):cx+h]
        sz = max(patch.shape); pp = np.zeros((sz, sz), np.float32); pp[:patch.shape[0], :patch.shape[1]] = patch
        e = pe_energy(pp); tot = e.sum() + 1e-9
        for r in Rs:
            keep = vd_cartesian_mask(sz, r)                       # kept PE lines (ACS-centered, matches fftshift)
            El[o][r].append(float(e[~keep].sum() / tot))          # fraction of the organ's energy DISCARDED

# pool all (organ, R) conditions
rows = []
for o, nm in L.ABDO.items():
    if nm not in frag: continue
    for r in Rs:
        if El[o][r]:
            rows.append(dict(organ=nm, tail=o in L.TAIL, R=r,
                             elost=float(np.mean(El[o][r])), drop=frag[nm]["R1"] - frag[nm][f"R{r}"]))
X = np.array([d["elost"] for d in rows]); Y = np.array([d["drop"] for d in rows])
pr, pp = pearsonr(X, Y); sr, sp = spearmanr(X, Y)
# per-R (does the SAME curve hold at each acceleration?)
perR = {r: pearsonr([d["elost"] for d in rows if d["R"] == r], [d["drop"] for d in rows if d["R"] == r])[0] for r in Rs}
# baseline: does R alone (ignoring which organ) explain the drop? (it can't rank organs)
r_Ronly = pearsonr([d["R"] for d in rows], Y)[0]

print(f"\n=== do all {len(rows)} (organ × acceleration) conditions collapse onto drop = f(energy removed)? ===")
print(f"  pooled:  Pearson r = {pr:+.2f} (p={pp:.1e}) ,  Spearman r = {sr:+.2f}")
print(f"  per-R Pearson:  " + " ".join(f"R{r}={perR[r]:+.2f}" for r in Rs) + "   (same relationship at every R = mechanism)")
print(f"  baseline 'acceleration R alone' r = {r_Ronly:+.2f}  (can't separate organs)")
json.dump(dict(pooled_pearson=float(pr), pooled_spearman=float(sr), perR={str(r): float(perR[r]) for r in Rs},
               r_R_only=float(r_Ronly), n=len(rows)), open(f"{RES}/energy_causal.json", "w"), indent=2)

import matplotlib as mpl
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"]})
NICE = {"liver": "liver", "kidney_L": "kidney", "spleen": "spleen", "pancreas": "pancreas",
        "gallbladder": "gallbladder", "adrenal_L": "adrenal", "esophagus": "esophagus"}
r8 = {d["organ"]: d for d in rows if d["R"] == 8}
ERASE, LOSE = "#c0392b", "#2c7fb8"
fig, ax = plt.subplots(1, 2, figsize=(15.5, 6.4), gridspec_kw={"width_ratios": [1, 1.2]})

# --- Panel A: concrete example (a few organs at 8x) ---
ex = [o for o in ["liver", "gallbladder", "adrenal_L"] if o in r8]   # robust / middle / fragile (clean contrast)
ex.sort(key=lambda o: r8[o]["elost"])
yy = np.arange(len(ex)); h = 0.38
ax[0].barh(yy + h/2, [r8[o]["elost"]*100 for o in ex], h, color=ERASE, label="signal the fast scan ERASED (%)")
ax[0].barh(yy - h/2, [r8[o]["drop"]*100 for o in ex], h, color=LOSE, label="accuracy the segmenter LOST (Dice points)")
ax[0].set_yticks(yy); ax[0].set_yticklabels([NICE.get(o, o) for o in ex], fontsize=11)
ax[0].set_xlabel("percent", fontsize=11)
ax[0].set_title("A.  Concrete (at 8× fast):\nerase more of an organ  →  lose more accuracy", fontweight="bold", fontsize=12)
ax[0].legend(fontsize=9.5, loc="lower right"); ax[0].grid(axis="x", alpha=.3)

# --- Panel B: the same rule for ALL organs and ALL speeds ---
cmap = {2: "#fdcc8a", 4: "#fc8d59", 6: "#e34a33", 8: "#a50026"}
for r in Rs:
    xs = [d["elost"]*100 for d in rows if d["R"] == r]; ys = [d["drop"] for d in rows if d["R"] == r]
    ax[1].scatter(xs, ys, s=72, color=cmap[r], edgecolor="white", lw=.7, label=f"{r}× faster", zorder=3)
b, aa = np.polyfit(X*100, Y, 1); xx = np.linspace(0, X.max()*100*1.02, 50)
ax[1].plot(xx, aa + b*xx, "--", color="#555", lw=2, zorder=2)
for o in ("pancreas", "gallbladder"):                  # liver & adrenal are labelled by their thumbnails
    if o in r8:
        dx, ha = ((-8, "right") if o == "gallbladder" else (6, "left"))   # gallbladder label LEFT, clears the thumbnail
        ax[1].annotate(NICE[o], (r8[o]["elost"]*100, r8[o]["drop"]), fontsize=9.5, fontweight="bold",
                       xytext=(dx, -3), textcoords="offset points", color="#222", ha=ha)
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from make_schematics_visual import pick_slice, norm as _n
from skimage.transform import resize as _resize
from scipy.ndimage import binary_erosion as _erode
import matplotlib.colors as _mc
_si, _sl = pick_slice()
def _thumb(oid, col, pad=6, sz=90):                     # fixed-size RGB thumbnail with the segmented organ HIGHLIGHTED
    m0 = _sl == oid
    if m0.sum() < 20: return None
    ys, xs = np.where(m0); y0, y1 = max(ys.min()-pad, 0), ys.max()+pad; x0, x1 = max(xs.min()-pad, 0), xs.max()+pad
    c = _n(_si)[y0:y1, x0:x1]; mc = m0[y0:y1, x0:x1]; s = max(c.shape)
    def _sq(a):
        o = np.zeros((s, s), np.float32); oy, ox = (s-a.shape[0])//2, (s-a.shape[1])//2
        o[oy:oy+a.shape[0], ox:ox+a.shape[1]] = a; return o
    cS = _resize(_sq(c.astype(np.float32)), (sz, sz), preserve_range=True, anti_aliasing=True)
    mS = _resize(_sq(mc.astype(np.float32)), (sz, sz), preserve_range=True) > 0.5
    rgb = np.stack([cS/(cS.max()+1e-6)]*3, -1); cc = np.array(_mc.to_rgb(col))
    rgb[mS] = 0.55*rgb[mS] + 0.45*cc                     # tint the organ region
    rgb[mS ^ _erode(mS)] = cc                            # + a solid outline so viewers see WHICH structure it is
    return np.clip(rgb, 0, 1)
def _place(pt, box, oid, col, cap, cap_dy=-50):
    th = _thumb(oid, col)
    if th is None: return
    ab = AnnotationBbox(OffsetImage(th, zoom=0.8), pt, xybox=box, xycoords="data", boxcoords="data",
                        pad=0.2, bboxprops=dict(edgecolor=col, lw=2.5),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2, shrinkB=3))
    ax[1].add_artist(ab)
    ax[1].annotate(cap, box, xytext=(0, cap_dy), textcoords="offset points", ha="center",
                   va=("top" if cap_dy < 0 else "bottom"), fontsize=10, fontweight="bold", color=col)
liv = r8.get("liver"); adrn = r8.get("adrenal_L") or r8.get("adrenal_R")
if adrn: _place((adrn["elost"]*100, adrn["drop"]), (26, 0.188),      # adrenal thumbnail UP in the clear top area
                12 if (_sl == 12).sum() >= (_sl == 13).sum() else 13, ERASE, "adrenal: lots\nerased → breaks", cap_dy=-46)
if liv: _place((liv["elost"]*100, liv["drop"]), (31, 0.04), 5, LOSE, "liver: little\nerased → stays sharp", cap_dy=46)  # liver DOWN, caption above
ax[1].set_xlabel("how much of the organ's fine detail the fast scan ERASED  (%)", fontsize=11.5)
ax[1].set_ylabel("how much the segmentation got WORSE  (Dice drop)", fontsize=11.5)
ax[1].set_title(f"B.  Same rule for every organ AND every speed\n"
                f"(each dot = one organ at one speed; they all fall on one line, r = {pr:+.2f})", fontweight="bold", fontsize=12)
ax[1].legend(title="scan speed", fontsize=9.5, loc="upper left"); ax[1].grid(alpha=.3); ax[1].set_ylim(-0.01, Y.max()*1.12)

fig.suptitle("Organs break because the fast scan erases their signal — the more it erases, the more they break",
             fontsize=14, fontweight="bold", y=1.005)
fig.tight_layout(); fig.savefig(f"{PLT}/energy_causal.png", dpi=150, bbox_inches="tight")
print(f"\nwrote {RES}/energy_causal.json , {PLT}/energy_causal.png")
