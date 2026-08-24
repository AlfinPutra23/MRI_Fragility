# Explainer / Primer — understand this project from the ground up

*Written to be readable without prior knowledge of accelerated MRI. You already know segmentation/nnU-Net;
this fills in the MRI-acceleration side, the project's ideas, and every term/value we report.*

---
## 1. The 30-second version
MRI scans are slow. You can make them **faster** by collecting **less data** ("acceleration"), but the image gets
blurrier. Usually people judge the blurry image with a quality score (SSIM). **We instead ask: how good is the
AI *segmentation* (organ outlines) on the accelerated scan, organ by organ?** We found: **small organs' outlines
collapse long before the quality score notices**, the collapse is **predictable from each organ's shape**, it's
caused by a training **imbalance**, and a loss tweak **partly fixes** it.

---
## 2. The MRI side, in plain words
- **k-space** = the raw data an MRI actually collects. It's the **Fourier transform** (frequency form) of the
  image. The scanner fills k-space line by line; that's what takes time. An inverse Fourier transform turns
  k-space back into the picture you see.
- **Undersampling / acceleration** = skip some k-space lines to scan faster. **Acceleration factor R** = how
  much faster: **R=2** means collect half the lines (2× faster), **R=8** means collect ⅛ (8× faster). Higher R =
  less data = blurrier/aliased image.
- **Reconstruction** = turning the incomplete k-space back into an image. The simplest ("zero-filled") just
  inverse-Fourier-transforms the missing data as zeros → blur. Fancy methods (E2E-VarNet) learn to fill the gaps.
- We **simulate** this: take a clean image → Fourier → drop lines (a "mask") → inverse Fourier → blurry image.
  (`scripts/kspace.py`.) We do R∈{1,2,4,6,8}; R=1 = clean (no acceleration).

## 3. The segmentation side (you know this)
A model (nnU-Net) outputs a label for every voxel → organ masks. **Dice** = overlap between predicted mask and
ground truth, 0 (no overlap) to 1 (perfect). We measure Dice **per organ, at each R**.

---
## 4. The core ideas we found (the heart of the project)
- **Fragility** = how much an organ's Dice **drops** as R increases. Small/thin organs are *fragile* (big drop);
  big compact organs are *robust* (tiny drop).
- **Metric-blindness** = the image score (SSIM) stays high while small-organ Dice crashes → image quality is a
  *blind* proxy for whether you can segment the organ. *(Others showed this too; we cite them.)*
- **Gradient imbalance (the cause)** = the model learns from a "feedback signal" (the loss gradient). Big organs
  have far more voxels, so they **dominate that signal (~40×)** and small organs get *starved* of learning.
- **The predictive law (our headline)** = you can **predict** an organ's fragility from its **shape** — its
  surface-to-volume ratio — *before* running any experiment. Thin → fragile; chunky → robust. Holds on 2 datasets.
- **The fix** = up-weight the fragile organs in the loss so they get more learning signal → they recover a bit
  (+2.2 Dice @R8), specifically under acceleration.

---
## 5. Glossary — every value/term we report
**The H-A table you asked about** (`tail  drop  SA/V  vol_cm3  contrast`):
| column | what it means | example |
|---|---|---|
| **organ** | the organ name | adrenal_R, liver |
| **tail** (`*`) | is it a **"tail" organ**? = one of the small/hard organs we focus on (gallbladder, esophagus, pancreas, adrenals, duodenum). "Tail" = the small end of the organ-size distribution (statistics term). | `*` = yes |
| **drop** | **fragility** = Dice at R1 *minus* Dice at R8. How much segmentation quality **falls** under heavy acceleration. Bigger = more fragile. | adrenal +0.285 (lost a lot); liver +0.023 (barely moved) |
| **SA/V** | **surface-to-volume ratio** = how much *boundary* the organ has relative to its *bulk*. **High = thin/wispy (fragile)**, low = chunky/compact (robust). This is our **predictor**. | adrenal 0.65–0.76 (high); liver 0.10 (low) |
| **vol_cm3** | organ **volume** in cubic centimeters — how big it is. | liver ~1200–1600; adrenal ~3–5 |
| **contrast** | **boundary contrast** = how sharp/bright the organ's *edge* is in the image (image gradient at the surface). Higher = easier-to-see edge = (weakly) more robust. | — |

**Other terms you'll see:**
- **R / R1 / R8 / clean** — acceleration factor; R1 = clean (no acceleration), R8 = 8× faster (worst blur).
- **NSD** (Normalized Surface Dice) — like Dice but for the *boundary* (within a tolerance); good for thin organs.
- **HD95** (95th-percentile Hausdorff Distance, mm) — worst-case boundary error (lower = better).
- **Cohen's d** — effect size: how *big* a difference is (d>0.8 = large; ours 1.2–2.4 = very large).
- **p-value / Wilcoxon** — is a difference statistically real (not luck)? p<0.05 = yes; ours p≈1e-26 = extremely.
- **seed / seed-control** — random start of training. Two runs differ a bit by chance ("seed noise"). **Seed-control**
  = run two variants from the *same* seed so any difference is the *method*, not luck. (This saved us from a wrong
  "the method does nothing" conclusion.)
- **λ (lambda) / ×2,×4,×8** — how *strongly* we up-weight fragile organs in the loss (×4 = up to 4× weight).
- **mixed-R** — training on a *mix* of acceleration levels (not just R2) so the model sees R8-blur during training.

---
## 6. The experiments (what each run is)
| name | what it does | result |
|---|---|---|
| **M0** | train @R2, test across R → per-organ fragility benchmark | ✅ small organs break first |
| **M1** | measure the per-organ training signal (gradient) | ✅ big organs dominate ~40× |
| **H-A** | does shape (SA/V) predict fragility? | ✅ r=0.85 & 0.90 (2 datasets) |
| **M2 / sweep** | does up-weighting fragile organs help? (seed-controlled) | ✅ +2.2 Dice @R8 |
| **AMOS** | repeat everything on a 2nd dataset | ✅ replicates |
| **λ-extension** *(running)* | is ×4 the best strength, or does ×6/×8 help more? | 🔄 |
| **mixed-R** *(running)* | does training on mixed acceleration help more? | 🔄 |

---
## 7. Phases & where we are
The plan has milestones **M0 → M4**:
- **M0 (benchmark)** ✅ done · **M1 (mechanism)** ✅ done · **H-A (predictive law)** ✅ done, 2 datasets.
- **M2 (the method)** — ✅ basic method works (+2.2); **🔄 we are HERE** — running *upgrades* (λ, mixed-R) to make
  the small-organ recovery bigger, and the predictive-law-as-weights "loop closer."
- **M3 (generalization)** — AMOS ✅ done; phase-splits + real-k-space (fastMRI) remain.
- **M4 (write-up)** — the paper (intro hook + figures already drafted).

**So: we're in late-M2 / M3 — strengthening the method and the results before writing.**

---
## 8. Your direct questions
**"Are we increasing accuracy on small organs?"** — **Yes**, that's exactly what the method (loss-weighting) does
(+2.2 so far), and the running upgrades (λ, mixed-R, and next: boundary-aware loss) are attempts to make that
recovery *bigger*. **Honest caveat:** the accuracy boost is the *secondary* contribution — it's modest and the
technique isn't novel. Our *primary, novel* contribution is the **predictive law** (predicting fragility from
shape). So we *are* improving small-organ accuracy, but we lead the paper with the prediction insight.

**"What's the next plan?"** — see `docs/NEXT_STEPS.md`. In short: finish the upgrades → the "loop-closer"
(use the predicted fragility as the weights) → boundary-aware loss → then write. The headline (benchmark +
predictive law, 2 datasets) is already done; everything now is upside on the method.

---
## 9. "Isn't it obvious that small organs break?" — the glass analogy (use this to explain to anyone)
**Concede the obvious part first (honesty wins):** yes, "thin organs are more fragile" is intuitive —
*"thin glass breaks more easily than thick glass."* That alone is *not* our contribution. So we don't lead with
it. Three things ARE non-obvious — those are the point:

**① It's not "small" — it's SHAPE, and it's *predictable*.**
"Small = fragile" is actually *wrong*. A big wine glass with a thin stem shatters; a tiny thick shot glass
survives. Our counter-examples: the **colon is large but fragile** (thin-walled), the **pancreas is small but
tough** (compact). The real rule is **surface-to-volume ratio**, and we found a **formula that predicts breakage
from shape alone, before you ever drop it** (r=0.85 & 0.90, two datasets).
> *"Everyone says small glasses break. We found it's not size — it's wall-thinness — and we can hand you a
> number, from the shape, that predicts which items shatter."*

**② The quality scanner says "fine" while it's shattered (the danger).**
The MRI-acceleration field speeds up scans by watching an **image-quality score (SSIM)**. We show that score
stays **green** while the small organs are already destroyed.
> *"The shipping company's 'is the box OK?' X-ray shows a perfect package — green light — while the wine glass
> inside is in pieces. They'd ship broken product thinking it's fine."*

**③ Part of the breakage is bad packing, not weak glass (and we can fix it).**
The cause is partly a **training imbalance** — the AI starves small organs of learning signal because big organs
dominate it ~40×. That's not the glass being weak; it's the **packer spending all the bubble wrap on the big
sturdy items**. Rebalance → the delicate one survives better (+2.2).
> *"Some of why it broke isn't the glass — it's that we packed it badly. Fix the packing and it survives."*

**🎤 The 20-second version for a professor:**
> *"Everyone knows thin organs are harder to segment. We show three non-obvious things: (1) it's not size, it's
> surface-to-volume **shape**, and we can **predict** which organs break from anatomy alone — before any scan;
> (2) the standard image-quality score is **blind** to it, so a faster-scan protocol would **silently fail** on
> the small clinically-critical organs; and (3) part of the failure is a **fixable training imbalance**, not the
> organ being inherently unsegmentable."*

**What we're fixing (in priority):** (1) the *blindness + unpredictability* — give a shape-based rule for *which*
organs are at risk (the novel part); (2) the *accuracy itself* — rebalance training so starved small organs
recover under acceleration (secondary, modest).
