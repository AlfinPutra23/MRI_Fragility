# Law → attention: predicting each organ's safe acceleration limit (the actionable "act")

*2026-07-02. Code `scripts/predict_rstar.py`; figures `outputs/plots/{m0,amos}_rstar.png`; results `{m0,amos}_rstar.json`.
This is the clean, baseline-free realization of "use the law so the pipeline knows which organ needs attention."*

## Idea
The law (spectral centroid, see `LAW_V2.md`) predicts fragility from anatomy a priori. Turn that into a **per-organ
safe acceleration limit R\***: the largest R at which an organ's Dice stays within `tol` (=0.05) of its
unaccelerated Dice. **Low R\* = fragile = the pipeline must flag/protect it.** No acceleration experiment needed —
R\* is predicted from the organ's spatial-frequency content.

## Result — replicated on both datasets
The law predicts the safe limit strongly and consistently:

| dataset | law (centroid) → R\* | fragile organs (R\* ≲ 4, "FLAG") | robust (R\*=8) |
|---|---|---|---|
| **MRISeg** | Spearman **r = −0.86** | gallbladder 4.1, adrenals 4.2/4.3, colon 4.6 | spleen, kidneys, liver |
| **AMOS** | Spearman **r = −0.91** | adrenal_R 3.4, gallbladder 3.8, adrenal_L 4.0 | spleen, kidneys, liver |

Higher spectral centroid ⇒ lower R\*. The pipeline can therefore say, *from anatomy alone*: "at R=8 trust
liver/spleen/kidney; distrust adrenals/gallbladder/esophagus — flag them for protection or re-acquisition."

## Why this is the right "act" (given FG-LOUPE failed)
- **It doesn't need to beat LOUPE.** FG-LOUPE (steering the mask by fragility) failed — task-driven learned sampling
  is already near-optimal, and forcing it toward a fragility spectrum hurt (`docs/METHOD_FG_LOUPE.md` audit). This
  R\*-prediction sidesteps that entirely: it's a *use* of the law, not a competing sampler.
- **It's clinically meaningful and novel.** A per-organ, anatomy-predicted safe-acceleration limit is a usable output
  (protocol design, reliability flagging) that nobody has published.
- **It closes predict→act cleanly:** predict fragility (law) → predict R\* → pipeline flags/protects low-R\* organs.

## Wiring it into the model (training-side attention) — options
The R\* / predicted-fragility can drive the model's attention a priori:
1. **Predicted-fragility oversampling** — set per-organ oversampling by the law's predicted fragility (we already
   oversample rare tail organs; making the weights law-derived is the principled version, and oversampling is the
   lever that demonstrably worked — it lifted esophagus off 0).
2. **Inference-time reliability flag** — mask/flag organs with predicted R\* < operating R (no retraining).
3. (Loss weighting is a *wash* — do NOT rely on it; see multiseed `frag-loss = −0.007`.)

**Honest note:** the training-side (weighting) is expected to be a *loop-closure* result ("predicted fragility drives
attention as well as measured") rather than a big accuracy jump — because we've shown loss-reweighting is marginal and
data-oversampling is the real lever. The **headline "act" is the R\* prediction itself.**

## Paper framing
Contributions become: (1) fragility **law** (spectral centroid, cross-validated), (2) safe-limit **benchmark**,
(3) **actionable**: predict each organ's safe acceleration limit R\* from anatomy (r=−0.86/−0.91, two datasets) so the
pipeline flags fragile organs a priori. Learned sampling (LOUPE) beats fixed masks; fragility-*steered* sampling does
not beat LOUPE (reported honestly as a negative).
