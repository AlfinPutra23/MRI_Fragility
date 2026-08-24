# Law v2 — a physically-grounded fragility predictor (spectral centroid)

*2026-07-02. Motivation: colon/small_bowel are fragile despite low SA/V — the old law's outliers. A literature
deep-dive (below) pointed to the fix. Code `scripts/law_v2.py`, result `outputs/plots/m0_law_v2.png`.*

## The question
SA/V (surface/volume) predicts fragility at Spearman r=0.85, but **large hollow organs (colon, small_bowel) drop
hard despite low SA/V** — they sit above the line. Why, and can we do better?

## Literature deep-dive (what pointed the way)
- **Spectral bias / frequency principle** — networks trained by gradient descent fit **low frequencies first and
  high frequencies worst** ([Xu et al. 2022, arXiv:2201.07395](https://arxiv.org/abs/2201.07395); Rahaman 2019).
  ⇒ an organ whose signal is high-frequency is *inherently* under-learned.
- **K-space undersampling removes high-frequency energy first** (fast-MRI recon literature; HFGN, HF-refinement).
  ⇒ acceleration attacks exactly the high-frequency band. **Same band as spectral bias → double jeopardy.**
- **Fractal dimension** of organ boundaries quantifies tortuosity/complexity (Belviso 2026, *Clinical Anatomy*;
  bowel/vessel FD studies) — a candidate for the hollow-organ complexity SA/V misses.
- **Radiomics shape features** (SA/V, sphericity, compactness) are legitimate, scanner-robust descriptors — but
  they are *geometry*, blind to image contrast/frequency content.

**Synthesis:** the *unifying* driver isn't geometry — it's **how much of an organ's signal lives in high spatial
frequencies.** SA/V is only a crude proxy for it. The right predictor measures the frequency content directly.

## The predictor: spectral centroid
For each organ, take the clean image restricted to the organ (per max-area axial slice, fixed 96² patch),
2D-FFT, drop DC, and compute the **energy-weighted mean radial frequency**:

$$ \text{centroid} \;=\; \frac{\sum_{\rho} \rho\,|F(\rho)|^2}{\sum_{\rho} |F(\rho)|^2}, \qquad \rho=\frac{|k|}{k_{\text{Nyq}}}\in[0,1] $$

High centroid = signal concentrated at fine scales = fragile. It is cutoff-free (unlike an HF-fraction@R threshold).

## Result — replicated on BOTH datasets
**MRISegmentator (n=13 organs):**
| predictor | Spearman r | **leave-one-out CV R²** |
|---|---|---|
| SA/V (old law) | +0.85 | +0.45 |
| **spectral centroid** | **+0.86** | **+0.70** |
| HF-fraction @R8 (hard cutoff) | +0.78 | +0.58 |
| fractal dimension | +0.74 | +0.10 (overfits) |
| boundary contrast | −0.56 | — |
| SA/V + centroid | — | +0.66 |
| SA/V + fractal dim | — | +0.10 |

**AMOS-MRI (n=11 organs) — independent replication:**
| predictor | Spearman r | **leave-one-out CV R²** |
|---|---|---|
| SA/V (old law) | +0.83 | +0.46 |
| **spectral centroid** | **+0.86** | **+0.60** |
| HF-fraction @R8 | +0.86 | +0.52 |
| fractal dimension | +0.79 | +0.37 |
| SA/V + centroid | — | +0.41 (overfits) |

**Consistent across both:** centroid ≥ SA/V on rank, and clearly beats SA/V on predictive R² (0.70/0.60 vs 0.45/0.46);
centroid *alone* is best (every combo overfits at small n); FD and contrast underperform. Not a one-dataset fluke.

- **Centroid ties SA/V on ranking but predicts the drop *magnitude* far better** (LOO-CV R² 0.45→0.70, +55%).
- **Centroid *alone* is the best model** — every 2-feature combo is worse (overfitting at n=13). Occam wins; drop FD.
- The smooth centroid beats the hard HF-fraction@R8 — no arbitrary cutoff to defend.
- Visually (`m0_law_v2.png`), `colon` moves toward the trend under centroid vs being a clear SA/V outlier.

## Why this is the better law (for the paper)
1. **Mechanistic, not descriptive.** Fragility = spectral centroid ties *directly* to the two cited mechanisms
   (k-space HF removal × network spectral bias). SA/V was a geometric stand-in; this is the real variable.
2. **Better magnitude prediction** (LOO-CV R² 0.70) = a law you can *use* to predict an organ's safe R\*, not just rank.
3. **One clean, cutoff-free feature.** Defensible and simple.

## Honest caveats / next tests
- Rank correlation improvement is marginal (0.86 vs 0.85); the win is in **predictive R²** and **grounding**, not rank.
- n=13 organs; 5 features were screened (mild multiple-comparisons risk) — centroid was the *a priori* physical
  favorite, which mitigates it.
- **Must replicate on AMOS** (`--out_prefix amos --root <AMOS>`) and across the **multifold** folds (error bars).
- colon/small_bowel improve but aren't perfectly on the line — hollowness is partly, not fully, absorbed.

## Recommendation
Adopt **spectral centroid as the primary predictor**, present **SA/V as the intuitive geometric proxy**, and frame
the law around the **frequency-content mechanism** (spectral bias × k-space). Validate on AMOS + multifold, then this
becomes the paper's headline: *fragility is predicted by an organ's spatial-frequency content — the exact thing fast
MRI destroys and networks learn last.*
