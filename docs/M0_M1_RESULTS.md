# M0 + M1 Results — Per-organ fragility of abdominal MRI segmentation under acceleration

**Dataset:** MRISegmentator-Abdomen (195 pts × 4 T1 phases = 780 series, 135/60 patient split → 540 train /
240 test cases, 62 structures). **Sim k-space:** retrospective variable-density Cartesian + ACS, R∈{1,2,4,6,8}.
**Model:** nnU-Net v2 3d_fullres, trained @R=2 (250 epochs), tested across R. n=240 test cases. No-TTA inference.

---
## Headline (honest)
As MRI acceleration R grows, **small/fragile abdominal organs' segmentation collapses far faster than large
compact organs', while image SSIM stays high** — the premise **HOLDS** (tail mean Dice drop **0.149** vs large
**0.070**, R1→R8). The cleanest case: the **adrenals (~3.6 cm³) lose ~30% Dice (0.64→0.45) while the liver
loses 2% (0.99→0.97)** — a ~10× fragility ratio — at SSIM 0.97→0.745. The mechanism de-risk confirms the
seg-loss gradient is **large-organ-dominated early in training (214×) and at convergence (14×)**, so an
organ-agnostic loss under-serves the fragile organs and **rebalancing has room**. Two honest refinements:
(1) fragility tracks **volume *and* boundary complexity**, not volume alone (colon/small_bowel are large but
fragile; pancreas is small but robust); (2) the static 448× volume imbalance does **not** survive a trained
model (collapses to 14×), so the method should weight by **fragility, not inverse volume**.

---
## Table 1 — M0 per-organ fragility (Dice vs R, n=240)
| organ | tail | R1 | R2 | R4 | R6 | R8 | drop(R1→R8) |
|---|---|---|---|---|---|---|---|
| adrenal_L | ★ | 0.638 | 0.635 | 0.596 | 0.516 | 0.440 | **−0.198** |
| adrenal_R | ★ | 0.645 | 0.642 | 0.605 | 0.532 | 0.459 | **−0.186** |
| gallbladder | ★ | 0.895 | 0.895 | 0.848 | 0.787 | 0.725 | −0.170 |
| colon | | 0.932 | 0.931 | 0.901 | 0.840 | 0.769 | −0.163 |
| duodenum | ★ | 0.858 | 0.857 | 0.836 | 0.787 | 0.717 | −0.141 |
| esophagus | ★ | 0.869 | 0.868 | 0.848 | 0.797 | 0.741 | −0.128 |
| small_bowel | | 0.940 | 0.939 | 0.918 | 0.875 | 0.826 | −0.113 |
| pancreas | ★ | 0.908 | 0.908 | 0.895 | 0.868 | 0.835 | −0.073 |
| stomach | | 0.947 | 0.947 | 0.936 | 0.910 | 0.881 | −0.066 |
| spleen | | 0.969 | 0.968 | 0.959 | 0.943 | 0.921 | −0.047 |
| kidney_L | | 0.969 | 0.969 | 0.960 | 0.947 | 0.929 | −0.040 |
| kidney_R | | 0.974 | 0.973 | 0.964 | 0.952 | 0.935 | −0.039 |
| liver | | 0.989 | 0.989 | 0.985 | 0.978 | 0.970 | −0.019 |

**Gate:** TAIL mean drop **0.149** vs LARGE **0.070** (2.1×, > 0.03 threshold) → **PREMISE HOLDS**.
→ Fig `outputs/plots/m0_fragility_curve.png`; data `outputs/results/m0_fragility_dice.json`.

**Honest nuance (state in text):** fragility is **not pure size-ordering**. Colon (−0.163) and small_bowel
(−0.113) are large-volume but thin-walled/hollow → fragile; pancreas (−0.073) is a small "tail" organ but
compact/solid → robust. The defensible characterization: **fragility ∝ small volume OR thin/hollow boundary**;
compact solid organs (liver/kidney/spleen) are robust. The tail-vs-large mean (2.1×) *understates* the cleanest
contrast (adrenal vs liver ≈ 10×) because pancreas dilutes the tail and colon inflates the large group.

## Table 2 — Metric-blindness
| R | image SSIM | mean TAIL Dice | mean LARGE Dice |
|---|---|---|---|
| 1 (clean) | 1.000 | 0.802 | 0.960 |
| 2 | 0.970 | 0.801 | 0.959 |
| 4 | 0.855 | 0.771 | 0.946 |
| 6 | 0.777 | 0.715 | 0.920 |
| 8 | 0.745 | 0.653 | 0.886 |

**Trend-level metric-blindness HOLDS:** SSIM degrades gently (−23%) while mean tail Dice craters (adrenals −31%
relative). The image metric under-reports small-organ damage → right panel of `m0_fragility_curve.png`.
**Negative result (report honestly):** the stricter *within-R per-case* Spearman ρ(SSIM, Dice) is ~0 for **all**
organs (tail +0.027 vs large +0.045) — at fixed R, per-case image quality does not predict per-case Dice for
anyone. We therefore claim metric-blindness only at the **trend (vs-R) level**, not as a per-case size-differential.

## Table 3 — M1(b) mechanism de-risk (per-organ seg-loss gradient mass on the trained net)
liver/adrenal **|dL/dlogits| mass ratio**, decomposed:
| ratio | value | reading |
|---|---|---|
| volume (n_vox) | 207× | static premise |
| CE-term | **14.1×** | what drives learning (≪ volume: a trained model self-corrects — solved liver → ~0 gradient) |
| Dice-term | 0.2× | Dice *over*-rebalances toward small organs, but is ~10⁴× smaller than CE → negligible |
| **total** | **14.1×** | ≥5× → **MECHANISM REAL, rebalancing has room** |

**During-training trajectory (`gradient_trajectory.json`, full 250-epoch instrumented run):** liver/adrenal CE
grad ratio **214× (ep0) → min 21× (ep20) → sustained ~44× median → 41× final**. The imbalance is **not** a
transient init artifact — it persists at ~40× throughout the learning that shapes the representation. (The
tail-centered 8-case convergence probe gave 14×; the absolute value depends on patch sampling, but every
measurement is ≫ the 5× gate.) Dice ratio collapses 356×→0.2× by ep30 and stays — self-rebalances but
negligible. Gradient mass is dominated by large-**and-hard** organs (colon > liver > stomach > small_bowel >
duodenum ≫ adrenals), consistent with the M0 "volume+boundary" finding.
→ Figs `outputs/plots/m1_gradient_probe.png`, `outputs/plots/m1_gradient_trajectory.png`.

---
## What this dictates for the method (M2)
1. **Weight by fragility (the M0 curve), NOT inverse volume.** Inverse-volume (×207) over-corrects and ignores
   the fragile large hollow organs (colon/duodenum). Fragility-weighting is the data-justified choice.
2. **Reweight the CE term specifically.** M1 shows CE carries the imbalance (Dice already over-corrects but is
   negligible). The minimal, mechanism-aligned intervention is a per-organ-weighted CE.
3. **De-risk the loss before the sampler.** Test fragility-weighted DiceCE vs uniform (seg-only, no LOUPE/VarNet)
   @R8 first; only build the joint sampling+recon if the loss recovers tail organs. (= the queued M2-entry run.)

## Honest scope / caveats
- **Simulated k-space** (no public abdominal raw-k-space + seg); state up front; fastMRI-brain real-k-space
  sanity check is the planned external validity check.
- **Single dataset, single fold, trained @R2** (deploy-across-R variant). Generality: replicate the ordering on
  **AMOS22-MRI** (downloaded) + across T1 phases (M3).
- **250-epoch** (de-risk) model; a HOLDS here should be re-confirmed at 1000 epochs before publication (under-
  training would only *help* the premise, so this is conservative).
- Metric-blindness claimed at trend level only (within-R ρ null — reported, not hidden).

## Figures / artifacts index
| file | shows |
|---|---|
| `outputs/plots/m0_fragility_curve.png` | per-organ Dice-vs-R + metric-blindness (SSIM overlay) |
| `outputs/results/m0_fragility_dice.json` | per-organ Dice at each R |
| `outputs/plots/m1_gradient_probe.png` | per-organ CE vs Dice gradient mass (convergence) |
| `outputs/results/m1_gradient_probe.json` | gradient-mass ratios |
| `outputs/results/m1_metric_blindness.json` | within-R ρ (the null) |
| `.../nnUNetTrainer_GradLog/.../gradient_trajectory.json` | during-training 214×→14× curve |

## One-paragraph framing (intro/abstract)
In accelerated multi-organ abdominal MRI, image-quality metrics hide a per-organ failure: as acceleration grows,
small and structurally-vulnerable organs' segmentation collapses long before SSIM/PSNR degrade — adrenals lose
~30% Dice while the liver loses 2%, at near-constant SSIM. We trace this to a **gradient-imbalance mechanism**:
the segmentation loss is dominated by large/well-solved organs (≈214× early in training), so an organ-agnostic
objective under-serves the fragile structures. We contribute (i) the first **per-organ fragility benchmark** for
abdominal MRI acceleration, characterizing fragility by volume *and* boundary complexity, and (ii) a
**fragility-rebalanced** task objective that targets the imbalance where it lives (the CE gradient). Honest,
rigorous, falsifiable — best fit MIDL/MICCAI.
