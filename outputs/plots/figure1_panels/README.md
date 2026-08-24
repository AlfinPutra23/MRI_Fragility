# Figure 1 — build-it-yourself kit

Clean, high-res (1800 px) panel images with **no titles / borders / arrows** — drop them into
**Figma, Illustrator, PowerPoint, or Keynote** and add the text + arrows yourself for full control.

## The panels
| file | what it is | use in figure |
|---|---|---|
| `01_clean_mri.png` | full-scan MRI slice | step 1 — "Clean MRI" |
| `02_kspace.png` | its k-space (Fourier), magma | step 2 — "k-space" |
| `03_undersampled_kspace.png` | after ×8 undersampling (kept lines) | step 3 — "Undersample 8×" |
| `04_fast_image_R8.png` | the aliased/blurred fast image | step 4 — "Fast image (8×)" |
| `05_segmentation.png` | colored organ segmentation on the MRI | step 5 — "Segmentation" |
| `05b_masks_transparent.png` | organ masks only, transparent bg | overlay on `01` at your own opacity |
| `06_sampling_mask.png` | the 1-D pattern of kept k-space lines | optional inset on step 3 |

## Suggested layout (two rows)
**Row (a) MEASURE** — the 5 panels left→right with arrows between:
`Clean MRI → k-space → Undersample 8× → Fast image → Segmentation`
subtitles: `full scan · Fourier domain · keep 32 of 256 lines · 8× · aliased · per-organ Dice`

**Row (b) PREDICT** — a 5-box chain (make the boxes in your tool):
`Organ anatomy → Spectral centroid → Fragility law → Safe limit R* → Flag fragile organs`
subtitles: `shape of one organ · how fine its detail is · r = 0.84 (5-fold) · per organ · adrenals·gallbladder`

Connect the rows with one short arrow labelled *"predict a priori — no fast scan."*

## Exact numbers (verified, 5-fold cross-validated)
- Fragility: tail organs drop **0.150** vs large **0.071** (R1→R8)
- SA/V law **r = 0.83 ± 0.02** · **spectral-centroid law r = 0.84 ± 0.02** (the one to headline)
- Safe limit: centroid → R\* **r = −0.85 ± 0.03** (MRISeg) / **−0.86 / −0.91** (MRISeg / AMOS single-fold)
- Fragile (low R\*): adrenals, gallbladder, esophagus, duodenum, colon · Robust (R\*=8): liver, kidneys, spleen

## Design tips (to look like a real paper figure)
- **Font:** Helvetica / Arial / Inter (not Times, not Calibri). One family, 2 weights.
- **Panel titles** ~18–22 pt bold; **subtitles** ~13–15 pt regular gray (#5c6470).
- **Arrows:** solid, thick (3–4 pt), dark (#1a1a1a), filled triangle heads. Keep them short.
- **(a) / (b) labels** bold, top-left of each row.
- **Law-chain boxes:** a light→deep coral gradient (`#f4ddd6 #eab6a6 #df8a73 #cf5b43 #a83226`); white text on the two darkest.
- Keep whitespace even; align panel tops; don't stretch the square MRI images (lock aspect ratio).

*(Regenerate any panel with `python scripts/export_panels.py`. The all-in-one matplotlib version is `outputs/plots/figure1.png`.)*
