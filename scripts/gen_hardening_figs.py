"""Reviewer-hardening visualizations (matplotlib, CVD-safe, value-labelled). Each strengthens a specific claim added
in the 2026-07-27 pass. -> site/assets/{energy_task_link,w1_dissociation,matched_baseline,law_two_cis,
complex_ordering,metric_blindness_psnr,qualitative_samples}.png . Run under mrifrag (nibabel/skimage/scipy/mpl)."""
import json, glob, os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
sys.path.insert(0, "scripts"); import labels as L

FRAG, ROB = "#E1701A", "#2E6FDB"          # fragile / robust (CVD-safe)
WIN, LOSE, MECH = "#0E8A5F", "#C4362B", "#7A4FCF"
INK, MUTE, GRID = "#12151a", "#5b6572", "#e3e7ec"
R = "nnUNet_raw/Dataset501_MRIfrag"; A = "site/assets"
plt.rcParams.update({"font.size": 12, "savefig.bbox": "tight", "savefig.facecolor": "white"})


def style(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTE, length=0); ax.grid(True, color=GRID, lw=0.8, alpha=0.7); ax.set_axisbelow(True)


def nm(o):
    try: return L.SHORT.get(o, L.LABELS.get(o, str(o)))
    except Exception: return str(o)


# ---- FIG 1: energy -> TASK(Dice) (escapes the Parseval tautology) ----
def fig_energy_task():
    rows = json.load(open("outputs/results/skmtea_law_v2.json"))["rows"]
    Rs = [2, 4, 6, 8]; cols = {2: "#BBD0F2", 4: "#7FA8E8", 6: "#3A63C0", 8: "#16306F"}
    fig, ax = plt.subplots(figsize=(7.4, 5.4), dpi=140); style(ax)
    E, G = [], []
    for Rv in Rs:
        e = [r[f"Elost_R{Rv}"] for r in rows]; g = [r[f"degr_R{Rv}"] for r in rows]
        ax.scatter(e, g, s=110, c=cols[Rv], edgecolor="white", lw=1.4, zorder=3, label=f"R{Rv} (within-R ρ={spearmanr(e,g).correlation:.2f})")
        E += e; G += g
    b = np.polyfit(E, G, 1); xs = np.linspace(min(E), max(E), 50)
    ax.plot(xs, np.polyval(b, xs), color=MUTE, lw=2, ls="--", zorder=2)
    rho = spearmanr(E, G).correlation
    ax.set_xlabel("fraction of k-space ENERGY removed", color=INK)
    ax.set_ylabel("Dice degradation  (task loss)", color=INK)
    ax.set_title(f"Removed energy predicts the TASK, not just image error\npooled ρ={rho:.3f} · within a FIXED R still ρ≈0.94 at R6/R8 (real knee k-space)", color=INK, fontsize=12.5)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    fig.savefig(f"{A}/energy_task_link.png"); plt.close(fig); return "energy_task_link.png"


# ---- FIG 2: W1 whole-body dissociation (over baseline difficulty) + preprocessing sensitivity ----
def fig_w1():
    d = json.load(open("outputs/results/totalseg_law_fullres.json"))["rows"]
    w = json.load(open("outputs/results/totalseg_w1_partial.json"))
    cen = np.array([r["centroid"] for r in d]); drop = np.array([r["drop"] for r in d]); dif = np.array([r["dice_R1"] for r in d])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.2), dpi=140, gridspec_kw={"width_ratios": [1.5, 1]})
    style(ax1); style(ax2)
    sc = ax1.scatter(cen, drop, c=dif, cmap="viridis", s=90, edgecolor="white", lw=1.1, zorder=3)
    cb = fig.colorbar(sc, ax=ax1); cb.set_label("baseline difficulty  (clean R1 Dice)", color=INK); cb.ax.tick_params(colors=MUTE)
    b = np.polyfit(cen, drop, 1); xs = np.linspace(cen.min(), cen.max(), 40)
    ax1.plot(xs, np.polyval(b, xs), color=MUTE, lw=2, ls="--", zorder=2)
    a = w["fullres"]["all"]
    ax1.set_xlabel("spectral centroid", color=INK); ax1.set_ylabel("Dice drop  R1→R8", color=INK)
    ax1.set_title(f"41 whole-body structures (TotalSeg-MRI, full-res)\ncentroid predicts drop OVER difficulty: partial ρ={a['partial_centroid_drop_given_difficulty']} (perm p={a['perm_p']})", color=INK, fontsize=12)
    # panel 2: full-res vs fast partial (preprocessing sensitivity)
    labels = ["all (41)", "gradual (30)"]; fr = [w["fullres"]["all"]["partial_centroid_drop_given_difficulty"], w["fullres"]["gradual_only"]["partial_centroid_drop_given_difficulty"]]
    fa = [w["fast"]["all"]["partial_centroid_drop_given_difficulty"], w["fast"]["gradual_only"]["partial_centroid_drop_given_difficulty"]]
    x = np.arange(2); wd = 0.36
    ax2.bar(x - wd/2, fr, wd, color=WIN, label="full-res", zorder=3)
    ax2.bar(x + wd/2, fa, wd, color=MUTE, label="fast recon", zorder=3)
    for xi, v in zip(x - wd/2, fr): ax2.text(xi, v + 0.02, f"{v}", ha="center", fontsize=10, color=INK)
    for xi, v in zip(x + wd/2, fa): ax2.text(xi, v + 0.02, f"{v}", ha="center", fontsize=10, color=MUTE)
    ax2.axhline(0, color=GRID, lw=1); ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_ylabel("partial ρ (centroid→drop | difficulty)", color=INK); ax2.set_ylim(-0.05, 0.8)
    ax2.set_title("holds on full-res, collapses on fast\n(preprocessing-sensitive — disclosed)", color=INK, fontsize=12)
    ax2.legend(frameon=False, fontsize=10)
    fig.savefig(f"{A}/w1_dissociation.png"); plt.close(fig); return "w1_dissociation.png"


# ---- FIG 3: mixed-R vs a MATCHED baseline (strawman deflation) ----
def fig_matched():
    def tails(tag): return np.array([json.load(open(f))["tail"] for f in sorted(glob.glob(f"outputs/results/b1_mit_{tag}_s*.json"))])
    base, mix, tv = tails("baseline"), tails("mixedr"), tails("tversky")
    means = [base.mean(), mix.mean(), tv.mean()]; names = ["R8 baseline\n(uniform)", "mixed-R", "Focal-Tversky"]
    cols = [MUTE, ROB, WIN]
    fig, ax = plt.subplots(figsize=(7.6, 5.4), dpi=140); style(ax)
    x = np.arange(3)
    ax.bar(x, means, 0.6, color=cols, zorder=3, alpha=0.9)
    for xi, arr in zip(x, [base, mix, tv]): ax.scatter([xi]*len(arr), arr, c=INK, s=26, zorder=4, alpha=0.7)
    for xi, m in zip(x, means): ax.text(xi, m + 0.004, f"{m:.3f}", ha="center", fontsize=11, color=INK, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("tail-organ Dice @ R8 (3 seeds)", color=INK)
    ax.set_ylim(0.45, 0.56)
    ax.annotate(f"matched gain = +{mix.mean()-base.mean():.3f}\n(vs the +0.088 OOD-strawman headline)", xy=(1, mix.mean()),
                xytext=(0.35, 0.545), fontsize=10.5, color=LOSE, ha="left")
    ax.annotate("Focal-Tversky edges\nmixed-R out here", xy=(2, tv.mean()), xytext=(1.55, 0.552), fontsize=10, color=WIN)
    ax.set_title("Against a MATCHED R8-trained baseline the mixed-R gain is modest\n(direction bulletproof: 239/240, p=4·10⁻⁴¹ — magnitude honest)", color=INK, fontsize=11.5)
    fig.savefig(f"{A}/matched_baseline.png"); plt.close(fig); return "matched_baseline.png"


# ---- FIG 4: two CIs on the law (report both; lead with the wide one) ----
def fig_two_cis():
    h = json.load(open("outputs/results/harden_stats.json"))["MRISegmentator"]["honest_n"]
    c = json.load(open("outputs/results/cpu_audit_extras.json"))["law_caseclustered"]
    obs = 0.863
    fig, ax = plt.subplots(figsize=(8.2, 3.6), dpi=140); style(ax); ax.grid(True, axis="x", color=GRID, lw=0.8)
    rows = [("structure-level bootstrap (n=13)\nTHE honest interval", h["boot_CI95"], WIN),
            ("case-clustered (n=240 cases)\nsubject-sampling noise only", c["boot95_CI"], MUTE)]
    for i, (lab, ci, col) in enumerate(rows):
        ax.plot(ci, [i, i], color=col, lw=4, zorder=3, solid_capstyle="round")
        ax.scatter([obs], [i], color=col, s=80, zorder=4, edgecolor="white")
        ax.text(ci[0]-0.01, i, f"{ci[0]}", va="center", ha="right", fontsize=10, color=col)
        ax.text(ci[1]+0.01, i, f"{ci[1]}", va="center", ha="left", fontsize=10, color=col)
    ax.axvline(obs, color=INK, ls=":", lw=1, alpha=0.6); ax.text(obs, 1.6, f"ρ={obs}", ha="center", fontsize=10, color=INK)
    ax.set_yticks([0, 1]); ax.set_yticklabels([r[0] for r in rows]); ax.set_ylim(-0.6, 1.8)
    ax.set_xlabel("spectral-centroid law  ρ (MRISeg)", color=INK); ax.set_xlim(0.45, 1.0)
    ax.set_title("Report BOTH confidence intervals — the wide one is the honest one", color=INK, fontsize=12)
    fig.savefig(f"{A}/law_two_cis.png"); plt.close(fig); return "law_two_cis.png"


# ---- FIG 5: complex multicoil forward preserves per-organ ordering ----
def fig_complex():
    po = json.load(open("outputs/results/complex_compare.json"))["per_organ"]
    mag = np.array([v["mag_drop"] for v in po.values()]); cx = np.array([v["cx_drop"] for v in po.values()])
    rho = spearmanr(mag, cx).correlation
    fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=140); style(ax)
    lim = max(mag.max(), cx.max()) * 1.1
    ax.plot([0, lim], [0, lim], color=MUTE, ls="--", lw=1.5, zorder=1)
    ax.scatter(mag, cx, s=100, c=MECH, edgecolor="white", lw=1.3, zorder=3)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("per-organ Dice drop — magnitude-only sim", color=INK)
    ax.set_ylabel("per-organ Dice drop — full complex forward\n(phase + coils + noise)", color=INK)
    ax.set_title(f"Fragility ordering survives a realistic acquisition\nmagnitude vs complex multicoil: Spearman ρ={rho:.3f}", color=INK, fontsize=12)
    fig.savefig(f"{A}/complex_ordering.png"); plt.close(fig); return "complex_ordering.png"


# ---- FIG 6: metric blindness generalizes to PSNR (needs percase_R8) ----
def fig_metric_blindness():
    d = json.load(open("outputs/results/cpu_audit_extras.json"))
    if "percase_R8" not in d: return None
    P = np.array(d["percase_R8"]["psnr"]); S = np.array(d["percase_R8"]["ssim"]); T = np.array(d["percase_R8"]["tail_dice"])
    m = d["metric_blindness_R8"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.8), dpi=140); style(a1); style(a2)
    a1.scatter(P, T, s=34, c=LOSE, alpha=0.55, edgecolor="none")
    a1.set_xlabel("PSNR of the R8 recon (dB)", color=INK); a1.set_ylabel("tail-organ Dice @ R8", color=INK)
    a1.set_title(f"PSNR → Dice:  ρ = {m['psnr_vs_tailDice_spearman']}", color=INK, fontsize=12.5)
    a2.scatter(S, T, s=34, c=ROB, alpha=0.55, edgecolor="none")
    a2.set_xlabel("SSIM of the R8 recon", color=INK); a2.set_ylabel("tail-organ Dice @ R8", color=INK)
    a2.set_title(f"SSIM → Dice:  ρ = {m['ssim_vs_tailDice_spearman']}", color=INK, fontsize=12.5)
    fig.suptitle("At a FIXED acceleration, recon-quality metrics are blind to segmentation — for BOTH metrics", color=INK, fontsize=13, y=1.02)
    fig.savefig(f"{A}/metric_blindness_psnr.png"); plt.close(fig); return "metric_blindness_psnr.png"


# ---- FIG 7: QUALITATIVE SAMPLES — fragile (adrenal) vanishes, robust (liver) persists ----
def _dice(a, b):
    s = a.sum() + b.sum(); return 2.0 * np.logical_and(a, b).sum() / s if s else float("nan")


def _best_slice(mask):
    return int(np.argmax(mask.sum(axis=(0, 1))))


def _crop(cy, cx, h, w, margin, shape):
    y0 = max(0, cy - margin); y1 = min(shape[0], cy + margin)
    x0 = max(0, cx - margin); x1 = min(shape[1], cx + margin)
    return y0, y1, x0, x1


def fig_samples(case="test_002_ART"):
    imc = np.asanyarray(nib := __import__("nibabel").load(f"{R}/imagesTs_clean/{case}_0000.nii.gz").dataobj).astype(np.float32)
    im8 = np.asanyarray(__import__("nibabel").load(f"{R}/imagesTs_R8/{case}_0000.nii.gz").dataobj).astype(np.float32)
    gt = np.asanyarray(__import__("nibabel").load(f"{R}/labelsTs/{case}.nii.gz").dataobj).astype(np.int16)
    pc = np.asanyarray(__import__("nibabel").load(f"{R}/predsTs_clean/{case}.nii.gz").dataobj).astype(np.int16)
    p8 = np.asanyarray(__import__("nibabel").load(f"{R}/predsTs_R8/{case}.nii.gz").dataobj).astype(np.int16)
    organs = [(12, "adrenal R (fragile)", FRAG, 48), (5, "liver (robust)", ROB, 150)]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.2), dpi=140)
    col_titles = ["clean scan + ground truth", "clean recon → prediction", "R8 recon → prediction"]
    for r, (o, label, col, margin) in enumerate(organs):
        gm = (gt == o)
        if gm.sum() < 50: continue
        z = _best_slice(gm)
        gs, cs, p8s = gm[:, :, z], (pc[:, :, z] == o), (p8[:, :, z] == o)
        ys, xs = np.where(gs)
        cy, cx = int(ys.mean()), int(xs.mean())
        y0, y1, x0, x1 = _crop(cy, cx, *gs.shape, margin, gs.shape)
        d_c, d_8 = _dice(pc == o, gt == o), _dice(p8 == o, gt == o)
        panels = [(imc[:, :, z], gs, "#00E5A0", None), (imc[:, :, z], cs, col, d_c), (im8[:, :, z], p8s, col, d_8)]
        for c, (img, mask, mc, dsc) in enumerate(panels):
            ax = axes[r, c]
            sub = np.rot90(img[y0:y1, x0:x1]); msk = np.rot90(mask[y0:y1, x0:x1])
            ax.imshow(sub, cmap="gray", vmax=np.percentile(sub, 99.5))
            if msk.any(): ax.contour(msk.astype(float), levels=[0.5], colors=[mc], linewidths=1.8)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0: ax.set_title(col_titles[c], fontsize=11, color=INK)
            if c == 0: ax.set_ylabel(label, fontsize=12, color=col, fontweight="bold")
            if dsc is not None: ax.text(0.04, 0.06, f"Dice={dsc:.2f}", transform=ax.transAxes, fontsize=11,
                                        color="white", fontweight="bold", bbox=dict(fc=mc, ec="none", alpha=0.85, pad=2))
    fig.suptitle("Under R8 acceleration the fragile organ's prediction degrades sharply; the robust organ is untouched",
                 fontsize=12.5, color=INK, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{A}/qualitative_samples.png"); plt.close(fig); return "qualitative_samples.png"


# ---- FIG 8: THEORY — the derived scalar Φ_R beats the empirical centroid; multi-factor index loses (Occam) ----
def fig_theory():
    tv = json.load(open("outputs/results/theory_validate.json"))
    law = {r["organ"]: r for r in json.load(open("outputs/results/m0_law_v2.json"))["rows"]}
    dice = json.load(open("outputs/results/m0_fragility_dice.json"))
    orgs = [o for o in tv["Phi_R_per_organ"] if o in law and o in dice]
    phi8 = np.array([tv["Phi_R_per_organ"][o]["R8"] for o in orgs])
    drop8 = np.array([dice[o]["R1"] - dice[o]["R8"] for o in orgs])
    frag = np.array([law[o]["tail"] for o in orgs])
    t2, t3 = tv["T2_Phi_beats_centroid"], tv["T3_occam_gate"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.2, 5.2), dpi=140, gridspec_kw={"width_ratios": [1.35, 1]})
    style(a1); style(a2)
    # Panel A: Φ_R8 vs drop (beats centroid)
    for x, y, f in zip(phi8, drop8, frag):
        a1.scatter(x, y, s=130, c=FRAG if f else ROB, edgecolor="white", lw=1.5, zorder=3)
    b = np.polyfit(phi8, drop8, 1); xs = np.linspace(phi8.min(), phi8.max(), 40)
    a1.plot(xs, np.polyval(b, xs), color=MUTE, lw=2, ls="--", zorder=2)
    for lab in ["adrenal_R", "gallbladder", "esophagus", "liver", "kidney_L"]:
        if lab in orgs:
            i = orgs.index(lab); a1.annotate(lab.replace("_", " "), (phi8[i], drop8[i]), textcoords="offset points", xytext=(7, 5), fontsize=9.5, color=INK)
    a1.scatter([], [], s=110, c=FRAG, edgecolor="white", label="fragile organ"); a1.scatter([], [], s=110, c=ROB, edgecolor="white", label="robust organ")
    a1.legend(frameon=False, fontsize=10, loc="upper left")
    a1.set_xlabel("Φ_R  =  fraction of organ k-space energy the R8 mask discards  (a-priori, derived)", color=INK, fontsize=10.5)
    a1.set_ylabel("Dice drop  R1→R8", color=INK)
    a1.set_title(f"The DERIVED scalar Φ_R predicts fragility — and beats the proxy\nΦ_R → drop ρ={t2['spearman_Phi8_vs_drop']}  >  centroid ρ={t2['spearman_centroid_vs_drop']}", color=INK, fontsize=12)
    # Panel B: Occam gate
    names = ["Φ_R alone\n(single scalar)", "multi-factor index\n(SA/V·√Φ/contrast)"]
    vals = [t3["spearman_Phi_alone"], t3["spearman_FI"]]
    a2.bar([0, 1], vals, 0.6, color=[WIN, LOSE], zorder=3, alpha=0.9)
    for x, v in zip([0, 1], vals): a2.text(x, v + 0.015, f"{v}", ha="center", fontsize=12, color=INK, fontweight="bold")
    a2.set_xticks([0, 1]); a2.set_xticklabels(names); a2.set_ylim(0, 1.0)
    a2.set_ylabel("Spearman → Dice drop (pooled organ×R)", color=INK)
    a2.text(0.5, 0.30, f"partial corr, controlling Φ_R:\nSA/V | Φ = {t3['partial_SAV_given_Phi']}\n1/contrast | Φ = {t3['partial_invContrast_given_Phi']}\n→ they add NOTHING over Φ",
            ha="center", va="center", transform=a2.transData if False else a2.transAxes, fontsize=10, color=INK,
            bbox=dict(fc="#f6f8fb", ec=GRID, pad=6))
    a2.set_title("Occam: the single scalar wins;\nthe multi-factor index does NOT", color=INK, fontsize=12)
    fig.savefig(f"{A}/theory_phi.png"); plt.close(fig); return "theory_phi.png"


if __name__ == "__main__":
    only = sys.argv[1:] or None
    figs = {"energy": fig_energy_task, "w1": fig_w1, "matched": fig_matched, "cis": fig_two_cis,
            "complex": fig_complex, "blindness": fig_metric_blindness, "samples": fig_samples, "theory": fig_theory}
    for k, fn in figs.items():
        if only and k not in only: continue
        try:
            out = fn(); print(f"  [{'ok ' if out else 'skip'}] {k}: {out}")
        except Exception as e:
            print(f"  [ERR] {k}: {type(e).__name__}: {e}")
    print("done")
