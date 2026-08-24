# FG-TDR architecture figure — asset package

Everything needed to view, regenerate, or **hand off to a design tool** the FG-TDR (Fragility-Guided Task-Driven
Reconstruction) architecture figure. This folder holds the individual real-MRI thumbnails; the composed figure and the
design brief live one/two levels up (paths below are relative to the **project root** `mri_fragility/`).

---

## 1. File manifest

| File | What it is |
|---|---|
| `outputs/plots/fgtdr_architecture.png` | **The composed figure** (raster, ~2100×1010). Drop straight into slides/paper. |
| `outputs/plots/fgtdr_architecture.svg` | **Editable vector** (~600 KB, real thumbnails embedded as raster). Open in Figma / Illustrator / Inkscape / Claude design and restyle freely. |
| `outputs/plots/fgtdr_thumbs/*.png` | **The 6 real knee thumbnails** as standalone transparent PNGs (this folder) — so a designer *places* real images instead of redrawing them. |
| `docs/FGTDR_FIGURE_BRIEF.md` | **Design brief** — story, per-element "picture ideas", exact hex palette, arrow semantics, typography, legend. Paste into a design tool to regenerate as clean vector art. |
| `scripts/make_fgtdr_architecture.py` | Generator for the composed PNG + SVG. |
| `scripts/export_fgtdr_thumbs.py` | Generator for the 6 thumbnails in this folder. |

## 2. The 6 thumbnails (this folder)

| File | Shows | How it's made |
|---|---|---|
| `1_undersampled_input.png` | Aliased/blurred knee MRI (the degraded input) | zero-filled recon at **R = 8** variable-density Cartesian |
| `2_kspace_mask.png` | k-space with the sampling mask | `log|masked FFT|`; bright center band = kept low freqs |
| `3_reconstruction.png` | Clean, sharp knee MRI (x̂) | fully-sampled magnitude (illustrative target of R_θ) |
| `4_pred_segmentation.png` | Prediction ŝ with colored overlay | input + segmentation overlay (thin cartilage/meniscus) |
| `5_ground_truth.png` | Ground truth s\* with colored overlay | clean image + GT overlay |
| `6_fragility_Wk.png` | **The fragility prior W(k)** — k-space heatmap | **real** `1 + α·P_struct(k)/P_all(k)`, log-scaled; the novel term |

## 3. Regenerate (CPU only — never touches the GPU)

Use the **base anaconda** interpreter (has `h5py` 3.8 to read the SKM-TEA compound-complex `target`, plus `matplotlib`):

```bash
cd mri_fragility
/home/user/anaconda3/bin/python scripts/make_fgtdr_architecture.py   # -> composed .png + .svg
/home/user/anaconda3/bin/python scripts/export_fgtdr_thumbs.py       # -> the 6 thumbs in this folder
```

> Env note: the `mrifrag` conda env has **no h5py**; only base-anaconda (or `magicnet` with `PYTHONNOUSERSITE=1`) can
> read the SKM-TEA `.h5`. That is why these figure scripts run under base-anaconda, not `mrifrag`.

## 4. Data provenance

- **Source:** SKM-TEA real multicoil knee **qDESS** k-space, case **MTR_020**, the abdominal-analogue slice with the most
  segmented structures. Segmentation = the raw-data-track NIfTI.
- **Not synthetic:** input, k-space, reconstruction, and W(k) are all real data. Only the *layout* is drawn.
- **License:** SKM-TEA requires its data-use agreement; the download SAS token is time-limited (expires 2026-08-05). Do
  not commit raw `.h5`/token anywhere public.

## 5. Color palette (hex)

| Role | Hex |
|---|---|
| Unrolled recon R_θ | `#c0392b` red |
| Data consistency (DC) | `#f0c5bd` pink |
| U-Net encoder | `#2c7fb8` blue |
| Bottleneck | `#f4c430` yellow |
| U-Net decoder | `#2ca25f` green |
| Fragility prior W(k) / accents | `#8c6bb1` purple |
| Task+freq loss / LOSS box | `#d99000` amber · `#b8860b` border · `#fdf3d8` fill |

## 6. Using it with a design tool

- **Fastest:** open `fgtdr_architecture.svg` and nudge colors/fonts/spacing directly.
- **From scratch (prettiest):** paste `docs/FGTDR_FIGURE_BRIEF.md` into Claude design + attach the 6 thumbnails from this
  folder; it regenerates the layout as native vectors you can polish.
- Layout rule to preserve: **every picture→arrow gap and every flow-arrow length is identical** (the script solves for one
  shared gap). Keep that — it's what makes the row read cleanly.

## 7. Honest caveat (carry into the paper)

In the composed figure both **ŝ** (predicted) and **s\*** (ground truth) currently render the *ground-truth* overlay —
standard idealized flow for an architecture diagram. The real degraded R8 prediction gets swapped into ŝ once the
GPU-gated FG-TDR experiment (`scripts/condseg_knee_fgtdr.py`, queued behind user training) finishes and produces a
segmenter. Everything else is real.

## 8. See also

- `docs/METHOD_PAPER_PLAN.md` — the FG-TDR math, ablations, and win/lose criterion this figure illustrates.
- `docs/FGTDR_FIGURE_BRIEF.md` — the design brief.
