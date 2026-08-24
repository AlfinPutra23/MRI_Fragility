# The project, explained from scratch (read this one)

*Last updated 2026-07-01. This is the "I want to actually understand the whole thing" document.
For the paper pitch see `PAPER_HOOK.md`; for the plan see `WACV_PLAN.md`; for results see `M0_M1_RESULTS.md`.*

---

## 0. TL;DR (the whole paper in 6 lines)

1. **Fast MRI is undersampled MRI.** To scan faster you skip data in k-space (the raw frequency space). We then
   run an automatic organ segmenter on the faster (lower-quality) image.
2. **Some organs break, some don't.** As you scan faster, big solid organs (liver) stay fine; small/thin/hollow
   organs (adrenal glands, gallbladder, esophagus, duodenum) collapse. We call these the **tail organs**.
3. **We can *predict* which organs break — from anatomy alone.** A simple shape number (surface-to-volume ratio)
   predicts fragility with correlation ~0.85–0.90, on **two independent datasets**. That's the headline: a
   **predictive law**, not just an observation.
4. **We know *why*.** During training the network's learning signal is dominated by big organs (~44× stronger
   gradient for liver than adrenal). Small organs are literally under-trained. It's a mechanism, not a mystery.
5. **We can *fix* it — on the acquisition side.** Instead of changing the loss (modest, +2 Dice), we let the
   scanner **learn which k-space lines to sample** so the fragile organs survive. That gives **+0.089 tail Dice**
   at 8× acceleration. This learned, fragility-aware sampling is the novel method.
6. **The arc is: predict → explain → act.** That's a complete, defensible story for one paper.

---

## 1. The problem, in plain language

**Why MRI is slow.** An MRI scanner doesn't take a photo. It fills in a grid of *frequencies* called **k-space**,
one line at a time, and the image is the (inverse) Fourier transform of that grid. Filling every line is slow
(minutes → patient discomfort, motion, cost). **Acceleration** = skip some lines. If you acquire only 1/R of the
lines, that's "R× acceleration" (R = 2, 4, 6, 8…). Skipping lines makes the reconstructed image blurrier / aliased.

**Why segmentation.** Downstream, a neural network labels each pixel by organ (spleen, liver, kidney, pancreas,
adrenal glands, …). This is **multi-organ abdominal segmentation** — the thing radiologists and treatment-planning
tools rely on. If the fast scan wrecks the segmentation of a clinically important small organ, the "fast MRI" is
useless for that organ.

**The gap nobody filled.** Everyone measures accelerated MRI by *image quality* (PSNR/SSIM — "does the picture
look nice"). Almost nobody asks: **when I accelerate, which *organs'* segmentations fail, how fast, and can I
predict it?** That per-organ, task-level fragility question is our lane.

---

## 2. The three findings (this is the paper)

### Finding A — the benchmark: fragility is real and organ-specific
We take real abdominal MRI, retrospectively undersample k-space at R = 1, 2, 4, 6, 8, segment, and measure
per-organ **Dice** (overlap of prediction vs truth; 1.0 = perfect) at each R.

- **Tail organs** (small/thin/hollow: gallbladder, esophagus, pancreas, both adrenals, duodenum) lose **~0.15
  Dice** from R1→R8. Cleanest example: **adrenals 0.64 → 0.44** (a ~30% relative collapse).
- **Large solid organs** (liver, spleen, kidneys) barely move — **liver 0.99 → 0.97**.
- So the damage is **not uniform** — it's concentrated in a predictable set of organs.

### Finding B — the law: you can predict fragility from anatomy
For each organ compute its **surface-to-volume ratio (SA/V)** — literally how much boundary it has per unit of
bulk. A sphere has low SA/V (compact); a thin tube or a tiny gland has high SA/V.

- **SA/V predicts the R1→R8 Dice drop with Spearman r ≈ +0.85** on the primary dataset (MRISegmentator) and
  **r ≈ +0.90** on a second dataset (AMOS22-MRI). Volume alone is a *worse* predictor (r ≈ −0.66).
- **Why this matters:** it turns a description ("small organs break") into a **predictive law** ("give me an
  organ's shape and I'll tell you its safe acceleration limit"). *That* is the novelty — nobody predicts
  acceleration-fragility from geometry.

### Finding C — the mechanism: why they break (during training)
This is the "why," and it's honest about a subtlety.

- **Naïve guess:** small organs break because they're small and the loss ignores them. Partly true, but the real
  story is in the **gradients** (the learning signal). Cross-entropy loss produces a gradient roughly proportional
  to an organ's pixel count. We instrumented a full 250-epoch training run: the liver's gradient stays **~44×
  stronger than the adrenal's**, sustained across training (not just at initialization).
- **So small organs are chronically under-trained** — the network spends its capacity on the big, easy organs.
  When you *also* degrade the image (acceleration), the under-trained organs have no margin and collapse first.
- **The glass analogy (for your professor):** drop a tray of glassware. The thick mug (liver) survives; the thin
  wine glass (adrenal) shatters. Not because you aimed at the wine glass — because *thin things have less margin*.
  Acceleration is the drop; SA/V is the glass thickness; the gradient imbalance is *which glasses you practiced
  catching*. Same physics, three views.

---

## 3. The method (the "act" — our contribution beyond the benchmark)

We tried **two** ways to protect the fragile organs. Being honest about both is part of the paper's credibility.

### Lever 1 — change the *loss* (the obvious move). Verdict: works, but modest.
Up-weight the fragile organs' cross-entropy so they get more of the learning signal. Seed-controlled test:
**+2.2 tail Dice at R8** (statistically real, p ≪ 0.001, 178/240 cases improve). Good — but small, and we found
it's **subsumed** by a simpler trick: just training on a *mix* of acceleration levels (mixed-R augmentation) gives
**+8.8** on its own. So the loss is a modest, somewhat-redundant knob. We report it honestly and move on.

### Lever 2 — change what you *sample* (the novel move). Verdict: this is the method.
Here's the key insight: **the loss can only make the best of the data you acquired. The bigger lever is acquiring
different data.** So instead of a fixed undersampling pattern, we make the k-space sampling mask **learnable** and
train it *end-to-end with the segmenter*, with a fragility-aware objective. The mask learns to spend its limited
line budget on the frequencies that keep the fragile organs alive.

Result at 8× acceleration (2D proof-of-concept), tail Dice, clean monotonic ladder:

```
random fixed         0.478
variable-density     0.509     <- standard fixed baseline
LOUPE (learned)      0.573     <- learning the mask alone: +0.064
OURS (learned+frag)  0.598     <- + fragility prior:       +0.025
                               ------------------------------------
                               total vs fixed:            +0.089
```

**+0.089 tail Dice** is large in this game (recall the whole R1→R8 drop is ~0.15). And the ordering is perfectly
monotonic, which is the sign of a real effect rather than noise. **This — task-aware, fragility-guided learned
sampling — is the paper's method.**

---

## 4. What's running tonight, and *why* (closing the two honest weaknesses)

A good reviewer will attack two things in the current results. We're fixing both *right now* (overnight batch,
launched 15:36, ~18 h):

| Weakness a reviewer flags | The fix running tonight | What it proves |
|---|---|---|
| **"Your benchmark is one fold — no error bars."** | **Multifold**: train nnU-Net folds 1–4 (fold 0 done), re-measure fragility + the SA/V law on each. | The benchmark and the r≈0.85 law hold **with cross-validated error bars**, not a lucky split. |
| **"Your B1 method uses zero-filled images — that's not a real fast-MRI pipeline."** | **B1 + recon-UNet**: put a learned reconstruction network in the loop, re-run all sampling variants. | The +0.089 ordering **survives a realistic learned recon** — it's not a zero-filling artifact. |
| (bonus) **"Is +0.089 just seed luck?"** | **Multiseed**: 5 sampling variants × 3 seeds. | The gain comes with **mean ± std** across seeds. |

If these come back clean (results land through tomorrow), the two biggest objections to the paper are gone.

**How to check progress yourself:**
- `outputs/logs/b1_multiseed.log` — which of the 15 runs are done.
- `outputs/logs/overnight_chain.log` — the top-level stage markers.
- `outputs/results/b1_*_s*.json` — each finished run's tail/large Dice.
- Later: `outputs/logs/fold{1..4}.log` (multifold training), `outputs/results/multifold.json` (the aggregated law).

---

## 5. Our WACV chances — honest odds

**Context.** WACV (IEEE Winter Conf. on Applications of Computer Vision) accepts roughly **40–45%** of submissions
— noticeably kinder than MICCAI (~30%). It explicitly welcomes **applications + analysis + benchmark** papers,
which is exactly what this is. Your professor wants WACV; it's a good fit. Deadline ~Sept 2026 (round 2).

**Where we stand *right now* (before tonight's results): ~50%.**
- *For us:* a genuinely novel angle (predict fragility from anatomy), a clean 3-act story (predict→explain→act),
  **two datasets**, a real mechanism, and a method that beats baseline by a clear margin. That's a complete paper.
- *Against us:* single-fold benchmark (no error bars), the B1 method is 2D + zero-filled (proof-of-concept, not a
  full pipeline), and the fragility *loss* is modest/subsumed. A tough reviewer dings all three.

**If everything tonight turns out good (multifold clean + recon holds + multiseed tight): ~65–70%.**
- Cross-validated benchmark + a law that holds on 2 datasets *with error bars* is hard to argue with — it becomes
  a **contribution reviewers can cite and build on**, which is what gets analysis papers accepted.
- The recon result upgrades the method from "toy" to "plausible pipeline," neutralizing the biggest method attack.
- At that point the remaining risk is *framing/novelty perception* ("is predicting fragility enough of a new idea
  for a vision venue?") and writing quality — both under our control.

**What would push it higher (>70%), if we have time before the deadline:**
1. **Real (not retrospective) k-space** or a public raw-k-space set → removes "you only simulated acceleration."
2. **The loop-closer (A3):** use the *predicted* SA/V fragility as the prior that *drives* the sampling — so the
   law (Finding B) literally powers the method (Finding C). That makes the three findings one tight loop instead
   of three separate results. High narrative payoff.
3. **A third dataset / a downstream clinical readout** (e.g., organ volume error, not just Dice).

**What would drop it (the honest risks):**
- Multifold reveals the law is fold-dependent (r swings a lot) → we'd have to soften "law" to "trend."
- The recon network *erases* the sampling advantage (if a good recon fixes everything, our sampling edge shrinks)
  → we'd reframe around low-data / hard-organ regimes where it still helps.
- A reviewer decides it's "too medical / not vision-y enough" for WACV → mitigated by submitting to the
  applications/datasets track and foregrounding the *method* (learned sampling) over the clinical framing.

**Bottom line:** we have a real paper *today* (~50/50). Tonight's batch is specifically engineered to convert the
two things reviewers would reject on into strengths, which is what takes it to **~two-in-three**. That's a good
bet for WACV, and everything we build is reusable for a stronger MICCAI version later.

---

## 6. Glossary (so you can talk to your professor without hedging)

- **k-space** — the raw frequency grid an MRI acquires; the image is its inverse Fourier transform.
- **Acceleration R** — acquire only 1/R of k-space lines to scan R× faster; higher R = worse image.
- **Undersampling mask** — which k-space lines you keep. *Fixed* patterns (random, variable-density,
  equispaced) vs a *learned* mask (LOUPE-style) trained to help the task.
- **Zero-filled recon** — the crudest reconstruction: set missing lines to zero and inverse-FT. Fast but aliased.
  "B1 + recon" replaces this with a *learned* denoiser (more realistic).
- **Dice** — segmentation overlap score, 0–1. Our main metric, reported per organ.
- **Tail organs** — our fragile set: gallbladder, esophagus, pancreas, adrenal ×2, duodenum (small/thin/hollow).
- **SA/V (surface-to-volume ratio)** — the shape number that predicts fragility. High = thin/small = fragile.
- **Fragility / the R1→R8 drop** — how much an organ's Dice falls from no-acceleration to 8×. Our target variable.
- **Critical acceleration R\*** — the largest R at which an organ still segments acceptably; the "safe limit"
  the law lets you predict per organ.
- **Gradient imbalance** — big organs produce a much stronger learning signal (~44×), so small organs are
  under-trained; the mechanism behind fragility.
- **LOUPE** — the technique for making the sampling mask differentiable so it can be *learned* end-to-end.
- **nnU-Net** — the standard self-configuring segmentation framework we use as the segmenter backbone.
- **Fold / cross-validation** — nnU-Net splits the data 5 ways; each "fold" is one train/val split. Multiple
  folds → error bars → trustworthy numbers.
- **Seed** — the random initialization. Matching seeds isolates a *method's* effect from random-run noise.
