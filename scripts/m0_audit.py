"""M0 groundwork (no GPU): dataset audit + gradient-imbalance de-risk.

Reads every patient's PRE-phase label (MRISegmentator-Abdomen) and computes:
  - per-organ physical volume (cm^3) and presence frequency,
  - the large-vs-small VOLUME ratio == the implicit per-organ GRADIENT ratio
    (CE/Dice gradient mass scales with voxel count) -> tests the v2 method premise,
  - image spacing / shape distribution (preprocessing facts).

The 15 G1 organs (paper Sec 6.4 order = nnU-Net label index, 1-indexed). We VALIDATE
the mapping empirically: liver must dominate, adrenals/gallbladder must be tiny.

Run: <python-with-nibabel> code/m0_audit.py
"""
import os, glob, json
import numpy as np
import nibabel as nib
from paths import MRISEG_REL as REL, PLOTS as OUT_P, RESULTS as OUT_R

# AUTHORITATIVE map from the repo README / ITK label file (verified 2026-06-28)
ORGANS = {1: "spleen", 2: "kidney_R", 3: "kidney_L", 4: "gallbladder", 5: "liver",
          6: "esophagus", 7: "stomach", 11: "pancreas", 12: "adrenal_R", 13: "adrenal_L",
          16: "small_bowel", 17: "duodenum", 18: "colon"}  # 13 abdominal organs we benchmark
# (8=aorta 9=IVC 10=portal_vein  14/15=lungs[partial FOV]  19+=vessels/muscles/bones)
# the small/hard "tail" organs the whole premise is about
TAIL = {4, 6, 11, 12, 13, 17}  # gallbladder, esophagus, pancreas, adrenal_R/L, duodenum
MIN_VOX = 30  # presence threshold


def patient_labels():
    """one representative phase (PRE) per patient, train+test."""
    out = []
    for split, idir in (("train", "labelsTr"), ("test", "labelsTs")):
        for lp in sorted(glob.glob(f"{REL}/{idir}/*_PRE.nii.gz")):
            pid = os.path.basename(lp).replace("_PRE.nii.gz", "")
            img = lp.replace(idir, "ImageTr" if split == "train" else "ImageTs").replace("_PRE.nii.gz", "_PRE_0000.nii.gz")
            out.append((split, pid, img, lp))
    return out


def main():
    pats = patient_labels()
    print(f"auditing {len(pats)} patients (PRE phase)\n")
    vols = {k: [] for k in ORGANS}       # cm^3 per case where present
    present = {k: 0 for k in ORGANS}
    spac_inplane, spac_slice, shapes, intens = [], [], [], []
    n = 0
    for split, pid, ip, lp in pats:
        lb = nib.load(lp)
        zoom = lb.header.get_zooms()[:3]
        vox_cm3 = (zoom[0] * zoom[1] * zoom[2]) / 1000.0
        la = np.asanyarray(lb.dataobj).astype(np.int16)
        for k in ORGANS:
            c = int((la == k).sum())
            if c >= MIN_VOX:
                vols[k].append(c * vox_cm3)
                present[k] += 1
        spac_inplane.append(float(zoom[0])); spac_slice.append(float(zoom[2]))
        shapes.append(tuple(int(s) for s in la.shape))
        if n < 12:  # sample image intensity/spacing from headers only (cheap) for a few
            im = nib.load(ip); intens.append((float(np.asanyarray(im.dataobj).min()),
                                              float(np.asanyarray(im.dataobj).max())))
        n += 1
        if n % 40 == 0:
            print(f"  ...{n}/{len(pats)}")

    # ---- aggregate ----
    rows = []
    for k, nm in ORGANS.items():
        v = np.array(vols[k]) if vols[k] else np.array([np.nan])
        rows.append(dict(id=k, organ=nm, tail=(k in TAIL), present=present[k],
                         present_pct=100 * present[k] / len(pats),
                         mean_cm3=float(np.nanmean(v)), median_cm3=float(np.nanmedian(v))))
    rows.sort(key=lambda r: (np.nan_to_num(r["mean_cm3"], nan=-1)), reverse=True)

    mv = {r["organ"]: r["mean_cm3"] for r in rows}
    liver = mv.get("liver", np.nan)
    small = min((r["mean_cm3"] for r in rows if r["present"] > 0), default=np.nan)
    small_nm = [r["organ"] for r in rows if r["mean_cm3"] == small][0]
    ratio = liver / small if small else np.nan
    # adrenal-specific
    adr = np.nanmean([mv.get("adrenal_R", np.nan), mv.get("adrenal_L", np.nan)])

    print("\n=== per-organ volume (cm^3) + presence — sorted large->small ===")
    print(f"{'organ':12} {'tail':5} {'present%':8} {'mean_cm3':>10} {'median_cm3':>11}")
    for r in rows:
        print(f"{r['organ']:12} {'*' if r['tail'] else ' ':5} {r['present_pct']:7.1f}% "
              f"{r['mean_cm3']:10.1f} {r['median_cm3']:11.1f}")

    print("\n=== GRADIENT-IMBALANCE DE-RISK (the v2 method premise) ===")
    print(f"  largest organ : liver  ~{liver:.0f} cm^3")
    print(f"  smallest organ: {small_nm}  ~{small:.1f} cm^3")
    print(f"  VOLUME(=gradient-mass) RATIO liver/{small_nm} = {ratio:.0f}x")
    print(f"  liver / adrenal(mean) = {liver/adr:.0f}x")
    verdict = "PREMISE HOLDS (>=50x imbalance -> rebalancing has room)" if ratio >= 50 \
              else "WEAK (<50x -> method premise shaky, lean on benchmark)"
    print(f"  -> {verdict}")

    # mapping sanity
    map_ok = (rows[0]["organ"] in ("liver", "small_bowel", "colon", "stomach")) and \
             (mv.get("adrenal_R", 1e9) < 30 and mv.get("gallbladder", 1e9) < 80)
    print(f"\n  label-map sanity: largest={rows[0]['organ']}, adrenal_R={mv.get('adrenal_R'):.1f}cm^3, "
          f"gallbladder={mv.get('gallbladder'):.1f}cm^3 -> {'CONSISTENT with anatomy' if map_ok else 'CHECK MAP!'}")

    print("\n=== preprocessing facts ===")
    print(f"  in-plane spacing: median {np.median(spac_inplane):.2f} mm  range [{min(spac_inplane):.2f},{max(spac_inplane):.2f}]")
    print(f"  slice thickness : median {np.median(spac_slice):.2f} mm  range [{min(spac_slice):.2f},{max(spac_slice):.2f}]")
    from collections import Counter
    sh = Counter(shapes)
    print(f"  shapes: {len(sh)} unique; top -> {sh.most_common(3)}")
    print(f"  intensity (12 imgs): min {min(i[0] for i in intens):.0f}, max {max(i[1] for i in intens):.0f} (>=0 => magnitude)")

    # ---- save results ----
    os.makedirs(OUT_R, exist_ok=True); os.makedirs(OUT_P, exist_ok=True)
    with open(f"{OUT_R}/m0_dataset_audit.json", "w") as f:
        json.dump(dict(n=len(pats), organs=rows, liver_cm3=liver, smallest=small_nm,
                       smallest_cm3=small, imbalance_ratio=ratio,
                       median_inplane_mm=float(np.median(spac_inplane)),
                       median_slice_mm=float(np.median(spac_slice)),
                       map_ok=bool(map_ok), verdict=verdict), f, indent=2)
    print(f"\nwrote {OUT_R}/m0_dataset_audit.json")

    # ---- figure: per-organ volume (log) = implicit gradient weight ----
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    r2 = [r for r in rows if r["present"] > 0]
    names = [r["organ"] for r in r2]; means = [r["mean_cm3"] for r in r2]
    cols = ["#d93025" if r["tail"] else "#5b8def" for r in r2]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar(range(len(r2)), means, color=cols)
    ax.set_yscale("log"); ax.set_ylabel("mean organ volume  (cm$^3$, log)")
    ax.set_xticks(range(len(r2))); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_title(f"Per-organ volume = implicit per-organ loss-gradient weight\n"
                 f"liver/{small_nm} = {ratio:.0f}x imbalance  (red = small 'tail' organs)",
                 fontweight="bold")
    for i, r in enumerate(r2):
        ax.text(i, means[i] * 1.12, f"{means[i]:.0f}", ha="center", fontsize=8)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#5b8def", label="large organ"),
                       Patch(color="#d93025", label="small / 'tail' organ")], loc="upper right")
    fig.tight_layout(); fig.savefig(f"{OUT_P}/m0_organ_volume_imbalance.png", dpi=140)
    print(f"wrote {OUT_P}/m0_organ_volume_imbalance.png")


if __name__ == "__main__":
    main()
