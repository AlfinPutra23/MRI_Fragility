# Related work & positioning — what SOTA exists, and why we don't need to beat it

*2026-07-02. For the paper's Related Work + the "why not just use X?" rebuttals reviewers will demand.
Bottom line: a large SOTA exists in *better samplers/reconstructors*, but it plays a different game (global
quality, on real k-space). Our contribution is a question none of them answer: per-organ fragility, predicted
from anatomy, with a per-organ safe-acceleration limit. We compare to baselines to show the phenomenon; we cite
the SOTA as related work; we do not (and need not) beat it.*

## The SOTA landscape (three lines)

**1. Learned k-space sampling.**
- LOUPE (Bahadir, Dalca, Sabuncu, IPMI'19 / TCI'20) — learnable population mask, optimized end-to-end.
- AutoSamp (variational information maximization) — joint optimization of sampling pattern + reconstruction.
- Scan-adaptive / patient-specific undersampling (2025) — reportedly beats population LOUPE on fastMRI knee.
- Active / sequential acquisition — SeqMRI (arXiv:2105.06460), adaptive acquisition policies (arXiv:2203.16392),
  Reducing Uncertainty with Active Acquisition (Zhang et al. 2019).
- PGA-DPS (Kang & Seo, ICLR 2026; target lab) — active subsampling with a diffusion prior. See [[impel-pga-dps-positioning]].

**2. Segmentation-aware / joint reconstruction–segmentation.**
- Segmentation-Aware MRI Reconstruction (MICCAI'22, Springer 10.1007/978-3-031-17247-2_6) — seg error backprops a
  pseudo-attention into the reconstructor; curriculum from full-sampling to target R.
- FR-Net (joint recon+seg, cardiac CS-MRI); Segmentation-guided reconstruction (arXiv:2407.18026, 2024).
- MOST (arXiv:2409.10394, 2024) — reconstruction optimized for multiple downstream tasks via continual learning.
- Direct segmentation from k-space (cardiac transformers, arXiv:2406.00192, 2024).

**3. Benchmark (closest existing).**
- **K2S Challenge** (Bioengineering 2023, 10.3390/bioengineering10020267; k2s.grand-challenge.org) — 300-patient
  multicoil raw k-space → knee bone/cartilage segmentation at 8×. Winner weighted Dice 0.910. 87 teams.
- Robustness of breast-lesion segmentation under MRI undersampling improves with k-space-aware DL (arXiv:2605.22327).

## Why we do NOT compete on the method leaderboard
1. **We can't fairly.** Those methods need **real multicoil raw k-space** (K2S provides it). Our data
   (MRISegmentator, AMOS-MRI) is **magnitude-only** — retrospective simulation. We can't enter that leaderboard.
2. **We tested it and it doesn't win.** FG-LOUPE (fragility-steered sampling) does **not** beat plain LOUPE
   (`docs/METHOD_FG_LOUPE.md`): task-driven learned sampling is already near-optimal; steering it by a fragility
   prior hurts. Reported as an honest negative.
3. **They answer a different question.** Every method above optimizes **global** reconstruction/segmentation
   quality. **None** analyze which *organs* fail, predict fragility from *anatomy*, or give a per-organ **safe
   acceleration limit R\***. That is our contribution — an empty lane, not a crowded one.

## Two literature facts that SUPPORT us (cite as motivation)
- **K2S: "no correlation between reconstruction and segmentation metrics."** This is exactly our thesis — image
  quality (PSNR/SSIM) ≠ downstream task reliability. Strong motivation for a *task-level, per-organ* study.
- **Per-organ difficulty is known but only under generic perturbations** (small/tubular organs — pancreas,
  gallbladder, bowel — deteriorate faster). **Nobody predicts it from anatomy or as a function of acceleration.**
  Confirms the novelty gap; cite to show we know the space.

## "Why not just use X?" — reviewer rebuttals
- *"Use segmentation-aware reconstruction / task-adapted sampling."* → Those optimize **global** quality; our
  FG-LOUPE result shows task-aware sampling **does not automatically protect the fragile organs**. Our contribution
  is the **prediction** (which organs need attention, a priori), orthogonal to any sampler — and complementary:
  our predicted fragility could *supervise* any of these methods.
- *"Small organs are obviously harder — what's new?"* → We don't just observe it; we **predict fragility from
  anatomy** (spectral centroid, LOO-CV R²≈0.65, two datasets) and turn it into a **per-organ safe limit R\***
  (r≈−0.86/−0.91). Prediction + safe-limit, not observation.
- *"Only simulated acceleration."* → State it as a limitation; real multicoil k-space (K2S-style) is the planned
  extension (see MICCAI plan). WACV (applications/eval track) accepts retrospective simulation when disclosed.
- *"Just one architecture (nnU-Net)."* → It's the standard; cross-validated over folds; law replicates on a 2nd
  dataset and a 2nd metric (spectral centroid). The phenomenon is model-agnostic in principle (mechanism =
  spectral bias × k-space HF removal, both architecture-independent).

## Our framing (one line)
Not "a better sampler" (crowded, needs real k-space, we lose) — but **"which organs break under acceleration, why
(high-frequency content × spectral bias), how to predict it from anatomy, and each organ's safe limit"** — a
task-level fragility analysis + predictive law + safe-limit tool that the sampler/reconstructor SOTA does not provide.
See [[m0-m1-results-and-m2-queued]], `docs/LAW_V2.md`, `docs/LAW_ATTENTION.md`.

## Benchmark landscape (datasets, and what they contain)

**Reconstruction benchmarks — have raw k-space, mostly no segmentation:**
- **fastMRI** (Zbontar 2018; RSNA-AI 2020) — largest raw multicoil k-space: knee, brain, prostate, breast, chest.
  No native seg (fastMRI+ adds bounding boxes/annotations).
- **CMRxRecon / CMRxRecon2024** (Nature Sci Data 2024; MICCAI challenge) — cardiac, 330 volunteers, multicoil,
  multiview/multimodality k-space. Recon-focused.
- **fastMRI Breast** (2024) — radial k-space, breast DCE.

**Abdominal multi-organ segmentation benchmarks — image-domain, NO raw k-space (what we use):**
- **AMOS** (Ji 2022, arXiv:2206.08023) — 500 CT + 100 MRI, 15 abdominal organs. We use AMOS-MRI (Dataset502).
- **MRSegmentator / MRISegmentator** (Radiology 2024) — multi-organ MRI, 40 classes. Our primary (Dataset501).
- **TotalSegmentator-MRI** — sequence-independent multi-organ MRI segmentation.

**Raw-k-space → segmentation (the intersection — closest to us, but KNEE):**
- **K2S Challenge** (Bioengineering 2023) — undersampled multicoil k-space → knee bone/cartilage seg, 300 pts, 8×.
- **SKM-TEA** (Desai 2022, NeurIPS D&B) — 155 multicoil T2 knee scans: **raw k-space + DICOM + tissue segmentation
  (femoral/tibial/patellar cartilage, meniscus) + pathology.** The one benchmark with real k-space AND seg labels.

**The gap = our novelty + our limitation (W1):** there is **no raw-k-space, multi-organ *abdominal* segmentation
benchmark.** The only "raw k-space + seg" sets (K2S, SKM-TEA) are knee. So (a) nobody has done abdominal per-organ
acceleration fragility (novelty), and (b) we cannot get real abdominal k-space (W1). **SKM-TEA is the escape hatch:**
validate the fragility law on its *real* multicoil k-space (thin cartilage = fragile, bulk = robust) → "not a
simulation artifact." Plan in `docs/SKMTEA_PLAN.md`.

**Directly-adjacent recent work to cite/check:** "Understanding Benefits and Pitfalls of Current Methods for the
Segmentation of Undersampled MRI Data" (arXiv:2508.18975, 2025) — segmentation of undersampled MRI; read to confirm
it does not scoop the per-organ fragility/prediction angle.

---

## ★ CONFIRMED POSITIONING (2026-07-15, after reading the closest papers)

**arXiv:2508.18975 does NOT scoop us — and we address its one tension.** It is a "first unified benchmark" of **7
segmentation METHODS** on undersampled MRI (one-stage joint vs two-stage recon-then-seg), finding **data-consistency
learned recon-then-seg wins**. It does per-organ fragility ✗, predictive law ✗, mechanism ✗, R\* ✗. Different axis.

**Positioning table (paper-ready):**
| Work | Does | Per-organ fragility | Predict from anatomy | Mechanism | R\* |
|---|---|:---:|:---:|:---:|:---:|
| **Ours** | fragility benchmark + law + mechanism + R\* + limit | ✅ 13 organs | ✅ centroid r=0.84 | ✅ Parseval r=0.978 | ✅ |
| 2508.18975 (2025) | benchmark of 7 seg *methods*; DC-recon-then-seg wins | ✗ | ✗ | ✗ | ✗ |
| Breast robustness 2605.22327 (2026) | train-on-undersampled ↑ robustness (1 structure) | ✗ | ✗ | ✗ | ✗ |
| K2S (2023) / SKM-TEA | real-k-space knee seg benchmark/dataset | ✗ | ✗ | ✗ | ✗ |

**Two reviewer tensions → defenses:**
1. *"Not the first benchmark."* → Ours is the first **per-organ FRAGILITY** benchmark (they benchmark methods). Cite them;
   we are complementary — they answer "which method", we answer "which organs break, why, and each organ's safe limit."
2. *"They found recon-then-seg is best; you say train-for-degradation beats recon."* → Different axis (they never tested
   segmenter-side mixed-R augmentation). AND we upgrade our recon opponent from classical CS to a **learned
   data-consistency (unrolled) recon** (`condseg_knee_recon.py`) — if mixed-R still wins, the claim is airtight ("even the
   DC recon they crowned is beaten, at zero inference cost"); else we honestly reframe to "matches the best recon for free."

**Do NOT claim as novel:** the mitigation (train-on-undersampled — breast paper did it) or a new sampler/recon. The novelty
is the **prediction framework + mechanism + R\* + fundamental-limit finding** (empty lane).
