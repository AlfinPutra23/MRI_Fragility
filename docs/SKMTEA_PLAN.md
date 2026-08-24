# Plan: validate the fragility law on SKM-TEA (real multicoil k-space) — the W1 fix

*2026-07-03. Goal: show the fragility phenomenon + the spectral-centroid law hold on GENUINE acquired multicoil
k-space (not magnitude-only simulation), removing the paper's biggest weakness (W1 in `docs/WEAKNESSES.md`).*

## Why SKM-TEA
It is the **only** public benchmark with **real multicoil raw k-space + tissue segmentations** (K2S has k-space+seg
too but no per-structure variety). SKM-TEA: 155 quantitative 3D qDESS **knee** scans, multicoil raw k-space, DICOM,
and segmentations of **femoral / tibial / patellar cartilage + meniscus**. Its structures span the fragility axis:
thin cartilage sheets (high SA/V, high centroid → predict fragile) vs the bulkier meniscus/bone (→ predict robust).
If our law predicts *which knee tissue breaks first* on real k-space, it is not a simulation artifact.

## The claim we want
> "The per-structure fragility law — thin, high-frequency structures collapse first under acceleration, predictable
> from anatomy (spectral centroid) — holds on **real acquired multicoil k-space** in a **different anatomy (knee)**,
> not just retrospectively-simulated abdominal MRI."

That is a strong generalization + realism result: **3 datasets, 2 anatomies, real + simulated k-space, 1 law.**

## Pipeline (mirrors our abdominal pipeline, but on real k-space)
1. **Data.** Register for SKM-TEA (Stanford AIMI); download via the `skm-tea` / DOSMA toolkit. **Disk warning:** the
   full raw k-space is ~1–1.6 TB — **download a SUBSET first (~20–30 cases)** for proof-of-concept; watch disk.
2. **Undersample the REAL k-space** at R ∈ {1,2,4,6,8} using SKM-TEA's provided Poisson-disc masks (or our vd mask
   for consistency) — this time masking *genuine* multicoil k-space, with coil sensitivities and real noise.
3. **Reconstruct** each R: start with **RSS / zero-filled** (fast), optionally the provided **E2E-VarNet** baseline.
4. **Segment.** Train a segmenter on SKM-TEA tissues (they provide splits + a baseline), or use their pretrained
   model; predict on each R.
5. **Fragility.** Per-tissue Dice vs R → the R1→R8 drop (reuse `fragility_eval.py` with a knee label map).
6. **Law.** Per-tissue (and **per-case**) SA/V + spectral centroid (`law_v2.py`, new labels module); correlate with
   the drop; predict R* (`predict_rstar.py`). Compare centroid vs SA/V as before.

## Handling the small-structure-count problem
Only ~4–5 tissue classes → n≈4 for a per-structure correlation (too small). **Fix = per-case analysis** (also our
W4 fix): 155 cases × ~5 tissues ≈ **~700 tissue instances**; correlate per-instance centroid vs per-instance drop →
large n, tight CI, and it doubles as the per-patient R\* demonstration. Report both per-structure (mean) and per-case.

## Milestones
- **SKM0 (½–1 day):** access + download ~20 cases; load one case's raw k-space + coils + seg; sanity RSS recon +
  undersample @R8; confirm the toolkit works. Watch disk.
- **SKM1 (1–2 days):** per-tissue Dice vs R on the subset (baseline recon + a quick segmenter) → do the fragility
  curves show thin cartilage dropping faster than meniscus? (the phenomenon on real k-space).
- **SKM2 (1 day):** per-case centroid/SA/V → law correlation + R\* on real k-space (the headline).
- **SKM3 (optional):** scale to more cases / add E2E-VarNet recon for robustness.

## Effort & timing (honest)
- **Effort: medium–high** — data access + large download + toolkit + a knee segmenter + adapting the analysis.
  Realistically **~4–7 focused days**, gated by download/segmenter.
- **WACV (Sept):** a *stretch* but high value — even SKM1+SKM2 on 20–30 cases gives a real-k-space validation that
  neutralizes W1 in the rebuttal. Prioritize the phenomenon (SKM1) + per-case law (SKM2); skip full-scale.
- **MICCAI (later):** SKM-TEA becomes a core result (full 155 cases + learned recon), reused from this work.

## Risks
- **Access/size:** SKM-TEA needs registration; raw k-space is huge — subset-first is essential (disk).
- **n / anatomy:** knee tissues are all thin-ish; the SA/V *range* is narrower than the abdomen → the law may be
  weaker per-structure. Mitigate with per-case (~700 instances) and by including meniscus/bone as the robust anchor.
- **Segmenter quality on real k-space** must be reasonable at R1 or the "drop" is dominated by baseline error — use
  their validated baseline.
- If the law is *weak* on knee: report honestly as a boundary of the law (different anatomy/tissue) — still informative.

## Decision
SKM-TEA is the right and only way to answer "is this real?" without inventing data. Recommend: **start SKM0 (subset
download + sanity) once a GPU frees**, decide go/no-go for WACV after SKM1 shows whether the phenomenon replicates on
real k-space. Reuses `kspace.py`, `fragility_eval.py`, `law_v2.py`, `predict_rstar.py` with a knee label module.
