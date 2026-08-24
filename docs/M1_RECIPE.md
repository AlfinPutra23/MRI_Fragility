# M1 — mechanism de-risk (run recipe)

**Goal:** decide whether the *method* (anatomy-prior gradient-rebalanced loss) has room, **before**
building it (M2). M0 proves the *phenomenon* (small organs break first). M1 tests the *cause* and the
*metric-blindness*, both reusing the M0 model — **no new training run**.

Fire **after M0 completes** (needs `predsTs_*` + the trained checkpoint):
```bash
bash scripts/run_m1.sh          # (a) metric-blindness  +  (b) gradient-mass probe
```

---
## (a) Metric-blindness — `m1_metric_blindness.py`
Per test case × R: image **SSIM** (clean vs R) vs per-organ **Dice** (M0 preds vs GT). Reports per-organ
Spearman ρ, both pooled and **within-R** (within-R removes the global acceleration trend = the clean test
of "at matched R, does a sharper image buy a better mask?").
- **HOLDS:** tail organs ρ_withinR ≈ 0 (image quality carries no info about small-organ segmentability)
  while large organs track (ρ clearly positive). → metric-blindness is real (benchmark Result, also feeds
  the method's motivation).
- Outputs: `outputs/results/m1_metric_blindness.json`, `outputs/plots/m1_metric_blindness.png`.

## (b) Gradient-mass probe — `m1_gradient_probe.py`  (the real M1 de-risk)
Loads the **M0 nnU-Net**, runs real preprocessed patches, and measures per-organ `|dL/dlogits|` mass,
**split into CE-term and Dice-term**. The premise (plan §0) is "large organs dominate the seg-loss
gradient." nnU-Net's loss is Dice+CE and **Dice is volume-normalized**, so this checks whether the 448×
static imbalance survives the actual loss.
- Self-test (no model, validates the math): `python m1_gradient_probe.py --self_test`
  (✅ verified: CE mass tracks volume 577×≈575×; Dice rebalances ~62× relative to CE).
- Gate read (liver/adrenal TOTAL grad-mass ratio):
  - **≥5× → MECHANISM REAL:** seg-loss gradient still liver-dominated → rebalancing has room → **M2**.
  - CE imbalanced but Dice + total balanced → **re-scope:** imbalance lives in CE / on the recon side,
    not the Dice term — the method must target that, not a generic per-organ reweight.
- Outputs: `outputs/results/m1_gradient_probe.json`, `outputs/plots/m1_gradient_probe.png`.

---
## The M1 gate
- **Mechanism real + small-organ gap remains** → build the method (M2: LOUPE-Task + gradient-rebalanced loss).
- **LOUPE-Task / Dice already close the gap** → **stop, publish the benchmark** (M0 + metric-blindness is
  still a paper). This is the honest off-ramp in plan §4.

## Notes
- The probe measures gradient **at the logits** (cleanest per-organ attribution of "gradient mass that
  drives learning"); it is a faithful, cheap proxy for the full LOUPE-Task gradient measurement (which adds
  the learned-mask + recon coupling and is part of M2, not this gate).
- Patches are centered on a tail-organ voxel so small organs are actually present in the measured patch.
- GPU0 is free once M0 (train+predict) is done; the probe is seconds-per-case.
