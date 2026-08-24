# Not All Organs Break Equally — MRI-Acceleration Fragility

Per-organ fragility of multi-organ abdominal MRI segmentation under k-space acceleration: a
benchmark, a spectral law that predicts *which* organ breaks from anatomy alone, a real-k-space
mechanism, per-organ safe-acceleration budgets (R\*), and an honest account of what fixes it
(mixed-R training) and what does not (three reweighting-style interventions).

**Paper:** [`paper/`](paper/) — preprint PDF + LaTeX source (not yet submitted to a venue).

## The result in one paragraph

Under retrospective undersampling (R = 1→8), per-organ Dice loss spans an order of magnitude
(adrenal −0.198 vs. liver −0.019; 13/13 organs Holm-significant; two datasets), while SSIM barely
moves. An organ's **spectral centroid** — the energy-weighted mean radial spatial frequency of its
content — rank-predicts its drop (Spearman ≈ 0.86 on MRISegmentator and AMOS-MRI, 5-fold
0.841 ± 0.023) and survives a raw-volume control: fragility is frequency content, not just size. On
real multicoil knee k-space the mechanism is an energy identity (removed k-space energy → image
error, r = 0.978). This yields a per-organ **acceleration budget** R\* (adrenals ≈ 4, liver/kidneys
≈ 8+, cross-dataset ordering ρ = 0.989). Training on mixed acceleration levels recovers the worst
organs (+0.088–0.134 tail Dice); reweighting the loss or steering a learned sampler by the same
prior does not — you cannot reweight your way out of deleted k-space.

## Layout

| path | contents |
|---|---|
| `paper/` | preprint (LaTeX + PDF + figures) |
| `scripts/` | all analysis/training code (relocatable via `paths.py`; label map in `labels.py` — the MRISegmentator prose order ≠ integer order, use this file) |
| `outputs/plots/` | every figure in the paper + supporting plots |
| `outputs/results/` | serialized results (JSON) behind every number in the paper |
| `docs/` | scientific documentation: explainer, experiment logs (M0/M1), run recipes, the spectral-law notes, related-work map |

## Reproducing

- **CPU-only de-risk + law figures:** `python scripts/m0_audit.py`, `python scripts/m0_kspace_sim.py`,
  `python scripts/law_v2.py`, `python scripts/metric_vs_law.py`, `python scripts/energy_causal.py`.
- **The full benchmark** needs the datasets (MRISegmentator-AB, AMOS22-MRI, SKM-TEA — distributed by
  their owners under their own licenses; download helpers are not included because portal tokens are
  per-user) and an nnU-Net v2 environment; see `docs/M0_RECIPE.md` / `docs/M1_RECIPE.md`.
- Environment: `scripts/requirements.txt` (torch, nnunetv2, nibabel, scikit-image, scipy, matplotlib).

## Honest-negatives policy

The negative results (fragility-weighted CE subsumed by mixed-R; FG-Seg wash; FG-TDR tie; FG-LOUPE
negative with audited failure modes; a width-matched causal frequency probe that is an informative
null) are reported in the paper with the same rigor as the positives.

## License

MIT (see [LICENSE](LICENSE)). The datasets remain under their original licenses.
