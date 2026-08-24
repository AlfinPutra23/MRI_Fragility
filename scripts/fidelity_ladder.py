"""SIMULATION-FIDELITY LADDER (defuses weakness #1: 'abdominal k-space is simulated from magnitude → artifact').
Build 4 rungs of R8 undersampling with INCREASING acquisition realism, sharing ONE VD mask per case so the *only*
difference between rungs is the added physics:
  L0 mag    : FFT(|x|) -> mask -> IFFT -> |.|              (the current benchmark; no phase/coils/noise)
  L1 phase  : + smooth B0-like image phase
  L2 coils  : + Nc=8 complex coil sensitivities (RSS combine)
  L3 noise  : + complex k-space noise                      (== full A=M·F·S, reproduces imagesTs_R8_cx)
Segment each rung with the EXISTING Uniform_s42 nnU-Net (inference only), then test whether the per-organ fragility
RANKING and the centroid law are INVARIANT across rungs. If they are, the fragility ordering is not a magnitude-sim
artifact. Endpoints (L0 vs L3) already gave ordering Spearman 0.978 (complex_compare.json); this decomposes the path.
  gen : python fidelity_ladder.py --stage gen  [--n_cases N]      (CPU; safe to run)
  eval: python fidelity_ladder.py --stage eval [--min_area 30]    (CPU; needs predsLAD_* from the GPU predict stage)
Predict stage is GPU-gated in run_fidelity_ladder.sh.  -> outputs/results/fidelity_ladder.json"""
import os, glob, sys, json, argparse, numpy as np, nibabel as nib
from scipy.stats import spearmanr, pearsonr
sys.path.insert(0, "scripts")
from kspace import vd_cartesian_mask
from complex_forward import coil_maps, smooth_phase
import labels as L

D = "nnUNet_raw/Dataset501_MRIfrag"
R = 8
RUNGS = ["mag", "phase", "coils", "noise"]
CFG = {  # (use_phase, use_coils, noise_frac)
    "mag":   (False, False, 0.0),
    "phase": (True,  False, 0.0),
    "coils": (True,  True,  0.0),
    "noise": (True,  True,  0.004),
}
NC = 8


def forward_rung(x, S, mask1d, rng, use_phase, use_coils, noise_frac):
    """one slice through a chosen fidelity rung. PE = rows (axis0). RSS combine (single coil -> |img|)."""
    ph = smooth_phase(*x.shape, rng) if use_phase else np.zeros(x.shape, np.float32)
    xc = (x * np.exp(1j * ph)).astype(np.complex64)
    Smaps = S if use_coils else np.ones((1, x.shape[0], x.shape[1]), np.complex64)
    kc = np.fft.fftshift(np.fft.fft2(xc[None] * Smaps, axes=(-2, -1)), axes=(-2, -1))
    if noise_frac > 0:
        s = noise_frac * np.abs(kc).mean()
        kc = kc + s * (rng.standard_normal(kc.shape) + 1j * rng.standard_normal(kc.shape)).astype(np.complex64)
    kc[:, ~mask1d, :] = 0
    imgc = np.fft.ifft2(np.fft.ifftshift(kc, axes=(-2, -1)), axes=(-2, -1))
    return np.sqrt((np.abs(imgc) ** 2).sum(0)).astype(np.float32)


def gen(args):
    for rg in RUNGS: os.makedirs(f"{D}/imagesTs_R{R}_L{rg}", exist_ok=True)
    cases = sorted(glob.glob(f"{D}/imagesTs_clean/*_0000.nii.gz"))
    if args.n_cases: cases = cases[:args.n_cases]
    print(f"fidelity ladder gen: {len(cases)} cases, 4 rungs @ R{R}, one shared mask/case", flush=True)
    for idx, cp in enumerate(cases):
        tag = os.path.basename(cp)
        if all(os.path.exists(f"{D}/imagesTs_R{R}_L{rg}/{tag}") for rg in RUNGS): continue
        im = nib.load(cp); vol = np.asanyarray(im.dataobj).astype(np.float32)
        v = np.moveaxis(vol, 2, 0); H, W = v.shape[1], v.shape[2]
        S = coil_maps(H, W, seed=idx)
        mask = vd_cartesian_mask(H, R, seed=idx)                 # SHARED across rungs (single acquisition)
        for rg in RUNGS:
            up, uc, nf = CFG[rg]
            rng = np.random.default_rng(1000 + idx)              # SAME rng seed/rung path so phase is identical
            out = np.empty_like(v)
            for i in range(v.shape[0]):
                out[i] = forward_rung(v[i], S, mask, rng, up, uc, nf)
            out = np.moveaxis(out, 0, 2).astype(np.float32)
            nib.save(nib.Nifti1Image(out, im.affine, im.header), f"{D}/imagesTs_R{R}_L{rg}/{tag}")
        if idx % 20 == 0: print(f"  {idx+1}/{len(cases)} {tag}", flush=True)
    print("gen DONE")


def _dice(a, b):
    s = a.sum() + b.sum(); return 2.0 * np.logical_and(a, b).sum() / s if s else np.nan


def evaluate(args):
    law = {r["id"]: (r["organ"], r["centroid"]) for r in json.load(open("outputs/results/m0_law_v2.json"))["rows"]}
    ids = list(L.ABDO)
    gts = sorted(glob.glob(f"{D}/labelsTs/*.nii.gz"))
    # per-organ mean drop for each rung, vs the forward-model-independent clean prediction
    acc = {rg: {o: [] for o in ids} for rg in RUNGS}
    for gp in gts:
        case = os.path.basename(gp)[:-7]
        pc = f"{D}/predsTs_clean/{case}.nii.gz"
        if not os.path.exists(pc): continue
        gt = np.asanyarray(nib.load(gp).dataobj).astype(np.int16)
        prc = np.asanyarray(nib.load(pc).dataobj).astype(np.int16)
        for rg in RUNGS:
            pf = f"{D}/predsLAD_{rg}/{case}.nii.gz"
            if not os.path.exists(pf): continue
            pr = np.asanyarray(nib.load(pf).dataobj).astype(np.int16)
            for o in ids:
                if (gt == o).sum() >= args.min_area:
                    acc[rg][o].append(_dice(prc == o, gt == o) - _dice(pr == o, gt == o))
    drops = {rg: {o: float(np.nanmean(acc[rg][o])) for o in ids if len(acc[rg][o])} for rg in RUNGS}
    present = [rg for rg in RUNGS if drops[rg]]
    common = [o for o in ids if all(o in drops[rg] for rg in present)]
    # ranking invariance across rungs (Spearman of per-organ drop vectors)
    rank_mat = {}
    for a in present:
        for b in present:
            if a < b:
                va = [drops[a][o] for o in common]; vb = [drops[b][o] for o in common]
                rank_mat[f"{a}_vs_{b}"] = round(float(spearmanr(va, vb).correlation), 3)
    # per-rung centroid law
    perrung_law = {}
    for rg in present:
        cen = [law[o][1] for o in common]; dr = [drops[rg][o] for o in common]
        perrung_law[rg] = {"pearson": round(float(pearsonr(cen, dr)[0]), 3),
                           "spearman": round(float(spearmanr(cen, dr).correlation), 3)}
    out = {"R": R, "rungs_present": present, "n_organs": len(common),
           "ranking_invariance_spearman": rank_mat,
           "per_rung_centroid_law": perrung_law,
           "per_organ_drop": {law[o][0]: {rg: round(drops[rg][o], 4) for rg in present} for o in common}}
    json.dump(out, open("outputs/results/fidelity_ladder.json", "w"), indent=2)
    print("\n=== FIDELITY LADDER ===")
    print("  ranking invariance (Spearman across rungs):", rank_mat)
    print("  per-rung centroid law:", perrung_law)
    print("wrote fidelity_ladder.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gen", "eval"], required=True)
    ap.add_argument("--n_cases", type=int, default=0)
    ap.add_argument("--min_area", type=int, default=30)
    a = ap.parse_args()
    gen(a) if a.stage == "gen" else evaluate(a)
