# Methods, Math & Theory — complete reference

*2026-07-03. The full formalism behind the project: the pipeline, the fragility measurement, every predictor
(with equations + the code that computes it), the mechanism/theory, the statistics, and the LOUPE / FG-LOUPE method
(including exactly why FG-LOUPE fails). Pairs with `LAW_V2.md`, `LAW_ATTENTION.md`, `METHOD_FG_LOUPE.md`, `WEAKNESSES.md`.*

---

## 0. Notation

| symbol | meaning |
|---|---|
| $x\in\mathbb{R}^{H\times W}$ | a magnitude MRI slice (image domain) |
| $\mathcal{F}$ | 2-D discrete Fourier transform; $X=\mathcal{F}(x)$ is **k-space** |
| $y$ | ground-truth segmentation; $m_o=\mathbb{1}[y=o]$ is the binary mask of organ $o$ |
| $M\in\{0,1\}^{H}$ | phase-encode (PE) undersampling mask (which k-space lines are acquired) |
| $R$ | acceleration factor $=H/\lVert M\rVert_1$ (fraction of lines kept $=1/R$) |
| $\tilde x_R$ | reconstructed image at acceleration $R$ |
| $S$ | segmentation network; $\hat y = S(\tilde x_R)$ |
| $\rho=\lvert k\rvert/k_{\text{Nyq}}\in[0,1]$ | normalized radial spatial frequency (0 = DC/center, 1 = Nyquist/edge) |

---

## 1. The forward pipeline (acquire → reconstruct → segment)

**Acquisition + retrospective undersampling.** Real fast MRI skips k-space lines. We simulate it: transform to
k-space, keep only the lines in $M$, and reconstruct (zero-filled):

$$ \tilde x_R \;=\; \big\lvert\, \mathcal{F}^{-1}\!\big( M \odot \mathcal{F}(x) \big)\,\big\rvert . $$

Code: `scripts/kspace.py::undersample_slice` (and the differentiable `scripts/loupe.py::undersample`).

**Variable-density mask** (the fixed baseline). Central "auto-calibration" (ACS) lines are always kept; the rest are
sampled with probability falling polynomially toward the periphery:

$$ p(k) \;\propto\; \Big(1-\tfrac{|k-c|}{H/2}\Big)^{\text{poly}},\qquad
   \text{ACS: } p(k)=1 \ \text{for } |k-c|\le \tfrac{\text{acs\_frac}\cdot H}{2}, $$

normalized so $\mathbb{E}\lVert M\rVert_1 = H/R$. Code: `scripts/kspace.py::vd_cartesian_mask` (`poly=8`, `acs_frac=0.08`).

**Segmentation + metric.** $\hat y = S(\tilde x_R)$; per-organ Dice
$\text{DSC}_o(R)=\dfrac{2\lvert \hat y_o\cap y_o\rvert}{\lvert\hat y_o\rvert+\lvert y_o\rvert}$. We sweep $R\in\{1,2,4,6,8\}$.
$S$ = nnU-Net (3D, 5-fold CV) for the benchmark; a 2-D U-Net for the sampling ablations. Code: `fragility_eval.py`.

---

## 2. Fragility (the target variable)

The **fragility** of organ $o$ is its Dice loss from full-scan to 8× fast:

$$ \boxed{\;\Delta_o \;=\; \text{DSC}_o(1) - \text{DSC}_o(8)\;}$$

Cross-validated benchmark result (5 folds, MRISegmentator): $\overline{\Delta}_{\text{tail}}=0.150$ vs
$\overline{\Delta}_{\text{large}}=0.071$ (tight per-fold std). Code: `fragility_eval.py` → `multifold_aggregate.py`.

---

## 3. The predictors (compute fragility from anatomy, a priori)

Each is a scalar per organ, computed from the clean image + GT **without any acceleration experiment**.

### 3.1 Surface-to-volume ratio (SA/V) — the geometric proxy
Fraction of an organ's voxels lying on its 1-voxel boundary shell:

$$ \text{SA/V}(o)=\frac{\lVert\, m_o \oplus \operatorname{erode}(m_o)\,\rVert_1}{\lVert m_o\rVert_1}
   \;=\;\frac{|\text{boundary voxels}|}{|\text{all voxels}|}. $$

Blob (liver) → low (mostly interior); thin/small (adrenal) → high (mostly boundary). Code: `h_a_predict.py`,
`law_v2.py` (`surf = mc ^ binary_erosion(mc)`).

### 3.2 Spectral centroid — the physical quantity (the best law)
Energy-weighted **mean radial frequency** of the organ's image content (DC removed):

$$ \boxed{\;\text{centroid}(o)=\frac{\sum_k \rho(k)\,\lvert \mathcal{F}(x\odot m_o)\rvert^2(k)}{\sum_k \lvert \mathcal{F}(x\odot m_o)\rvert^2(k)}\;}$$

It is literally "where the organ's signal lives in frequency." Low = smooth/coarse (energy at the kept center);
high = fine detail (energy in the discarded periphery). Code: `law_v2.py::radial_spec` (returns `centroid`),
visual `make_centroid_explainer.py`.

### 3.3 Other predictors (tested, weaker)
- **HF-fraction @R**: energy above the $R$-cutoff, $\text{HF}_R(o)=\sum_{\rho>1/R}P(k)\big/\sum_k P(k)$ with
  $P=\lvert\mathcal{F}(x\odot m_o)\rvert^2$. Physical but cutoff-dependent. Code: `law_v2.py::radial_spec` (`hf8`).
- **Fractal dimension** (boundary tortuosity), box-counting slope $D=-\,\mathrm{d}\log N(s)/\mathrm{d}\log s$.
  Code: `law_v2.py::box_fd`. Overfits at small $n$ → dropped.

**Leaderboard (5-fold, `make_law_leaderboard.py`):** centroid LOO-R² $0.69\pm0.05$ (rank $0.84$) ≫ HF $0.58$ >
SA/V $0.43$ ≈ fractal $0.41$. **Centroid wins.**

---

## 4. Theory — WHY thin/high-frequency organs break (the mechanism)

Two independent facts hit the **same** frequency band, so high-centroid organs are doubly disadvantaged.

### 4.1 Acquisition removes high frequencies
Natural MR images have approximately power-law spectra, $\lvert X(\rho)\rvert^2\sim \rho^{-\alpha}$ — energy
concentrated at low $\rho$. A variable-density $R$-fold mask keeps the central $\sim\!1/R$ of k-space and discards the
high-$\rho$ periphery. So the **fine-detail (high-frequency) content is deleted first**. An organ whose *discriminative*
signal (thin boundaries, small size) sits at high $\rho$ loses a larger fraction of its information — that fraction is
exactly $\text{HF}_R(o)$, and its "center of mass" is $\text{centroid}(o)$.

### 4.2 Networks learn high frequencies LAST (spectral bias / frequency principle)
For a network $f_\theta$ trained by gradient descent, the Fourier mode of the target at frequency $\xi$ is fit at a
rate that **decreases with $\lvert\xi\rvert$** — low frequencies first, high frequencies slowest (Xu et al. 2019,
"Frequency Principle", arXiv:2201.07395; Rahaman et al. 2019). In the NTK view, the neural-tangent-kernel eigenvalues
decay with frequency, so high-frequency target structure is systematically under-fit.

### 4.3 Double jeopardy → the law
High-frequency organ content is **(a) removed by the scanner** (§4.1) **and (b) the part the segmenter learns worst**
(§4.2). Both act on the high-$\rho$ band. The spectral centroid measures how much of an organ lives in that vulnerable
band, so:

$$ \Delta_o \;\uparrow\quad\text{as}\quad \text{centroid}(o)\;\uparrow . $$

SA/V correlates because thin/small shapes (high SA/V) *also* have high-frequency spectra — it is a **geometric proxy**
for the centroid, which is the underlying physical variable. This is why centroid predicts the *magnitude* better
(§3.3): it is the quantity the mechanism is actually about.

---

## 5. The law + statistics (how we validate)

**Correlation.** Across the $n=13$ organs, Spearman rank correlation $r$ between predictor and $\Delta_o$.

**Out-of-sample magnitude (the discriminating metric).** Leave-one-out CV $R^2$: for each organ $i$, fit a linear
$\Delta \sim \text{predictor}$ on the other $n-1$, predict $\hat\Delta_i$, then

$$ R^2_{\text{LOO}} = 1 - \frac{\sum_i(\Delta_i-\hat\Delta_i)^2}{\sum_i(\Delta_i-\bar\Delta)^2}. $$

Code: `law_v2.py::loo_r2`, `make_law_leaderboard.py`.

**Cross-validation.** nnU-Net's 5 folds give 5 independent $\{\Delta_o\}$ measurements → we report $r$ and $R^2$ as
**mean ± std across folds**. Results: SA/V $r=0.83\pm0.02$, **centroid $r=0.84\pm0.02$**, centroid$\to$R\* $r=-0.85\pm0.03$.
Code: `multifold_aggregate.py`, `law_multifold_aggregate.py`. Replicated on a 2nd dataset (AMOS-MRI).

---

## 6. R\* — the safe acceleration limit (the actionable "act")

The largest acceleration at which an organ stays within tolerance $\tau$ (=0.05 Dice) of its unaccelerated Dice,
by linear interpolation between the measured $R$ points:

$$ R^\*_o=\max\{\,R : \text{DSC}_o(R)\ge \text{DSC}_o(1)-\tau\,\}. $$

Low $R^\*$ = fragile = **flag/protect**. The law predicts it from anatomy: $\text{centroid}\uparrow \Rightarrow R^\*\downarrow$
($r=-0.85\pm0.03$, 5-fold; $-0.86/-0.91$ on MRISeg/AMOS). Code: `predict_rstar.py::rstar`.

---

## 7. The method — learned sampling (LOUPE) and FG-LOUPE

### 7.1 LOUPE: a differentiable, task-optimized k-space mask
Learnable logits $w\in\mathbb{R}^H$. Line probabilities, rescaled to the budget and with ACS forced on:

$$ p = \operatorname{rescale}\big(\sigma(s\,w)\big)\ \text{s.t. } \mathbb{E}[p]=1/R,\qquad p_{\text{ACS}}=1. $$

Differentiable sampling via the **binary-concrete / relaxed Bernoulli**:

$$ M_k=\sigma\!\Big(\tfrac{1}{\tau}\big(\operatorname{logit}p_k+\operatorname{logit}u_k\big)\Big),\quad u_k\sim\mathcal U(0,1), $$

so task-loss gradients flow to $w$; at deployment we take the top-$(H/R)$ lines. Code: `scripts/loupe.py::LOUPEMask`
(`probs`, `forward`, `rescale_probs`). Trained end-to-end with a DiceCE loss (`b1_joint.py::dice_ce`, per-sample soft
Dice + weighted CE). **Result:** learned mask beats fixed masks by $+0.044$ tail Dice @R8 (5 seeds).

### 7.2 FG-LOUPE (fragility-steered sampling) — and why it FAILS
Idea: bias the learned mask toward the frequencies fragile organs need, via a coverage penalty
$\lambda\,H(t,q)$ where $q=p/\lVert p\rVert_1$ is the mask distribution and $t$ is a "fragility target":

$$ t \;\propto\; \frac{S_{\text{frag}}}{S_{\text{all}}},\qquad
   S_{\text{frag}}[k]=\!\sum_{o}(w_o-1)\,\lvert\mathcal F(x\odot m_o)\rvert^2[k],\quad
   S_{\text{all}}[k]=\lvert\mathcal F(x)\rvert^2[k]. $$

Code: `b1_joint.py::frag_coverage_target` + the penalty in the train loop. **It fell $-0.089$ vs plain LOUPE.**
The audit (`METHOD_FG_LOUPE.md`) found two compounding bugs:

1. **High-frequency-pathological target.** Dividing by $S_{\text{all}}\sim\rho^{-\alpha}$ (tiny at high $\rho$)
   *inflates* the peripheral lines → $t$ points at high frequencies.
2. **Forcing gradient blow-up.** For cross-entropy $H(t,q)=-\sum_k t_k\log q_k$,
   $$ \frac{\partial H}{\partial q_k}=-\frac{t_k}{q_k}\ \xrightarrow[q_k\to 0]{}\ -\infty, $$
   which *forces* $q\!\approx\!t$ even at small $\lambda$ (fingerprint: $\lambda=0.05\approx\lambda=0.15$, saturation).
   The learned mask collapses into a **fixed, high-frequency mask** — worse than variable-density.

**Lesson (a real, reportable finding):** task-driven LOUPE is already near-optimal; steering it with a hand-computed
frequency prior *removes* the low-frequency coverage every good mask needs. The fix is not to steer the mask but to
improve the *pipeline* (learned recon in the loop / multi-coil) — see `WEAKNESSES.md`.

---

## 8. Code map

| file | what it computes |
|---|---|
| `kspace.py` | `vd_cartesian_mask`, `undersample_slice` — fixed-mask acquisition (§1) |
| `loupe.py` | `LOUPEMask` (learnable differentiable mask), `undersample` (§7.1) |
| `fragility_eval.py` | per-organ Dice vs $R$ → the drop $\Delta_o$ (§2) |
| `h_a_predict.py` | SA/V, volume, boundary contrast (§3.1) |
| `law_v2.py` | `radial_spec` (centroid, HF-fraction), `box_fd` (fractal), `loo_r2` (§3.2, §5) |
| `predict_rstar.py` | $R^\*$ per organ (§6) |
| `b1_joint.py` | 2-D LOUPE + seg training; `dice_ce`; `frag_coverage_target` (§7) |
| `multifold_aggregate.py`, `law_multifold_aggregate.py` | 5-fold mean±std of the benchmark + laws (§5) |
| `make_*` | all figures (schematics, Figure 1, laws_crossval/intuitive/leaderboard, explainers) |

---

## 8.5 Real-k-space validation (SKM-TEA) + echo-invariance via Parseval

The main results are retrospective, magnitude-domain simulations on abdominal MRI (weakness **W1**). To test whether
the **mechanism** survives a *real* acquisition, we validate on **SKM-TEA** — genuine fully-sampled multicoil qDESS
knee k-space `(512×512×160, 2 echoes, 16 coils)` + ESPIRiT maps + official Poisson masks + a 6-structure knee
segmentation (patellar / femoral / medial+lateral tibial cartilage / medial+lateral meniscus). This is a *transfer*
test: a different organ system (thin knee cartilage), a different scanner/contrast, real coils and noise.

**Per-structure quantities.** For structure $o$ on slice $z$, with $x$ the coil-combined reference and $M_R$ the
variable-density mask:
- energy removed (amplitude): $\; a_o(R) = \big\lVert (1-M_R)\odot \mathcal F(x\,\mathbf 1_o)\big\rVert_2$
- reconstruction error (amplitude): $\; \varepsilon_o(R) = \big\lVert \mathrm{recon}_R(x)-x \big\rVert_{2,\,o}$

**The mechanism = Parseval's theorem.** Because $\mathcal F$ is unitary, the energy discarded in k-space equals the
error injected in image space: $\;\lVert (1-M_R)\odot \mathcal F u\rVert_2 = \lVert \mathcal F^{-1}[(1-M_R)\odot\mathcal F u]\rVert_2.$
So $\varepsilon_o(R) \propto a_o(R)$ **as an identity**, independent of tissue contrast or brightness. Fragility of the
*image* is therefore governed by absolute energy removed — the measurable first half of the double-jeopardy chain — on
real data.

**Result (MTR_001, `skmtea_law_v4.py`).** Across 6 structures × 4 accelerations, $\log a_o(R)\!\to\!\log\varepsilon_o(R)$:
**echo-1 $r=0.94$, echo-2 $r=0.92$, combined $r=0.94$** (Spearman 0.91) — **both echoes collapse onto one line**
(`skmtea_law_v4_mechanism.png`; per-echo curves `skmtea_law_v4_predictors_echo{1,2}.png`).

**Spatial-aliasing refinement (`skmtea_law_v5.py`, the tightest version).** The isolated $a_o$ measures energy displaced
over *all* space, but $\varepsilon_o$ is measured *inside* the mask — and aliasing **redistributes** error spatially. A
compact, isolated structure (patellar cartilage) leaks most of its removed-energy error *outside* its own region (only
~19% stays in-region at $R{=}2$; residual $-0.40$), so $a_o$ overpredicts; conversely the menisci *receive* leaked
aliasing from adjacent bright cartilage and sit above the line. Using the **in-region** predictor
$\tilde a_o(R)=\big\lVert \mathcal F^{-1}[(1-M_R)\odot\mathcal F(x\mathbf 1_o)]\big\rVert_{2,\,o}$ (the structure's own
error field restricted to its mask) corrects this: **patellar residual $-0.40\!\to\!-0.07$, combined $r=0.94\!\to\!0.98$**
(echo-1 0.98, echo-2 0.99; all 6 structures within $\pm0.12$). The small positive menisci residual that remains is the
complementary *neighbor-leakage-in*, capturable only by the full reconstruction (i.e. the Dice endpoint).
Figs `skmtea_law_v5_{mechanism,predictors_echo1,predictors_echo2}.png`.

**Normalization caveat (why absolute, not fractional).** An earlier framing used a *fractional* input
$E_\text{lost}=a_o^2/\lVert\mathcal F(x\mathbf 1_o)\rVert_2^2$ against a globally-scaled error; the mismatch broke on the
dark T2 echo (echo-2 $r$ fell to 0.39), because a low-DC structure (meniscus) has a huge *fraction* but tiny *absolute*
high-frequency energy. This is itself an instance of the **metric-blindness** point: image-scale/normalization choices
can mislead. The absolute (Parseval) framing is dimensionally consistent and echo-invariant. Random-noise subtraction
did **not** rescue the fractional framing — confirming the issue was normalization, not SNR.

**Scope (honest).** SKM-TEA validates the *mechanism* (energy removed → reconstruction error) on real multicoil k-space,
not yet the anatomy→**Dice** law (needs a knee segmenter). The image-degradation proxy is *not* the endpoint — Dice is —
but the mechanism is the harder-to-argue causal half, and it holds. More cases (download in progress) → error bars.

---

## 9. One-paragraph summary
An organ's segmentation fragility under k-space acceleration is governed by **how much of its signal lives at high
spatial frequency**, because that band is *both* discarded by the scanner (variable-density undersampling) *and*
learned last by the network (spectral bias) — a double jeopardy. The **spectral centroid** measures this directly and
predicts fragility with LOO-CV $R^2=0.69\pm0.05$ (5-fold, 2 datasets), beating the geometric SA/V proxy ($0.43$); it
yields a per-organ **safe-acceleration limit $R^\*$** ($r=-0.85$) so a pipeline can flag fragile organs a priori.
Steering a learned sampler by this prior (FG-LOUPE) *fails* — task-driven sampling is already near-optimal — a clean
negative that focuses the contribution on the **law + benchmark + $R^\*$ prediction**.
