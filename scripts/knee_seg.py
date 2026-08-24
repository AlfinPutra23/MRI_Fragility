"""W2: real-k-space anatomy->Dice test on SKM-TEA (44 qDESS knee cases). Train a 2D U-Net on the CLEAN coil-combined
target (echo 0), then measure per-structure Dice at clean + R{2,4,6,8} (retrospective undersampling of the COMPLEX
target's k-space). Then test whether per-structure fragility (Dice drop) is predicted by:
  (a) spectral centroid  (anatomy predictor -- the Spectral Fragility Law), and
  (b) in-region |energy removed|  (the proven Parseval mechanism -> Dice, closes the mechanism->task gap).
HONEST NOTE: the 6 knee structures are all thin cartilage/meniscus (narrow centroid range), so (a) may be weak here
even if true -- (b) connects to the r=0.98 mechanism and is the robust real-k-space Dice test.
-> outputs/results/knee_law.json , outputs/plots/knee_law.png .  Run with base-anaconda python (has h5py+torch)."""
import os, glob, json, argparse, numpy as np, h5py, nibabel as nib
from scipy.stats import pearsonr, spearmanr
import torch, torch.nn as nn, torch.nn.functional as F
try:
    import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt; HAVE_MPL = True
except Exception:
    HAVE_MPL = False   # magicnet env lacks mpl; json is the deliverable, re-plot from it with base anaconda

RS = [2, 4, 6, 8]; NCLS = 7            # bg + 6 knee structures
LAB = {1: "patellar", 2: "femoral", 3: "tibial-med", 4: "tibial-lat", 5: "menisc-med", 6: "menisc-lat"}
PLOTS, RES = "outputs/plots", "outputs/results"
dev = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- model (reused from b1_joint.py) ----------
class DoubleConv(nn.Module):
    def __init__(s, i, o):
        super().__init__(); s.c = nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.InstanceNorm2d(o), nn.LeakyReLU(inplace=True),
                                                 nn.Conv2d(o, o, 3, padding=1), nn.InstanceNorm2d(o), nn.LeakyReLU(inplace=True))
    def forward(s, x): return s.c(x)


class UNet2D(nn.Module):
    def __init__(s, ncls, ch=(32, 64, 128, 256)):
        super().__init__()
        s.d1, s.d2, s.d3 = DoubleConv(1, ch[0]), DoubleConv(ch[0], ch[1]), DoubleConv(ch[1], ch[2])
        s.bott = DoubleConv(ch[2], ch[3]); s.pool = nn.MaxPool2d(2)
        s.u3 = nn.ConvTranspose2d(ch[3], ch[2], 2, 2); s.c3 = DoubleConv(ch[3], ch[2])
        s.u2 = nn.ConvTranspose2d(ch[2], ch[1], 2, 2); s.c2 = DoubleConv(ch[2], ch[1])
        s.u1 = nn.ConvTranspose2d(ch[1], ch[0], 2, 2); s.c1 = DoubleConv(ch[1], ch[0])
        s.out = nn.Conv2d(ch[0], ncls, 1)
    def forward(s, x):
        e1 = s.d1(x); e2 = s.d2(s.pool(e1)); e3 = s.d3(s.pool(e2)); b = s.bott(s.pool(e3))
        d = s.c3(torch.cat([s.u3(b), e3], 1)); d = s.c2(torch.cat([s.u2(d), e2], 1)); d = s.c1(torch.cat([s.u1(d), e1], 1))
        return s.out(d)


# ---------- k-space helpers ----------
def vd_mask(W, R, acs=0.08, seed=0):
    rng = np.random.RandomState(seed)
    m = np.zeros(W, bool); c = W // 2; na = max(1, int(acs * W)); m[c - na // 2:c + na // 2 + 1] = True
    fr = np.abs(np.arange(W) - c); p = 1.0 / (fr + 1); p[m] = 0; p /= p.sum()
    m[rng.choice(W, W // R - m.sum(), False, p=p)] = True
    return m


MASKS = {R: vd_mask(512, R) for R in RS}


def undersample(cimg, mask):
    """complex image slice -> zero-filled magnitude at the given PE mask (columns)."""
    Fk = np.fft.fftshift(np.fft.fft2(cimg)); Fk[:, ~mask] = 0
    return np.abs(np.fft.ifft2(np.fft.ifftshift(Fk)))


def norm(x):                                   # robust per-slice normalization
    p = np.percentile(x, 99.5); return (x / p).clip(0, 2.0) if p > 0 else x


def spectral_centroid(mag, M):
    P = np.abs(np.fft.fftshift(np.fft.fft2(mag * M))) ** 2
    H, W = mag.shape; yy, xx = np.mgrid[0:H, 0:W]
    rad = np.sqrt(((yy - H // 2) / H) ** 2 + ((xx - W // 2) / W) ** 2)
    return float((P * rad).sum() / P.sum()) if P.sum() > 0 else np.nan


def energy_removed(cimg, M, mask):             # in-region leakage-corrected |energy removed| (v5 predictor)
    Fo = np.fft.fftshift(np.fft.fft2(cimg * M)); Fo[:, mask] = 0
    ef = np.abs(np.fft.ifft2(np.fft.ifftshift(Fo)))
    return float(np.sqrt((ef[M] ** 2).sum()))


# ---------- data ----------
def load_case_slices(h5path, segpath, min_vox=200):
    seg = np.asanyarray(nib.load(segpath).dataobj).astype(np.int16)
    with h5py.File(h5path, "r") as f:
        TGT = f["target"]; Z = TGT.shape[2]
        zs = [z for z in range(Z) if (seg[:, :, z] > 0).sum() > min_vox]
        cimg = np.stack([TGT[:, :, z, 0, 0] for z in zs]).astype(np.complex64)   # echo0 complex
        lab = np.stack([seg[:, :, z] for z in zs]).astype(np.int64)
    return cimg, lab


def dice(pred, gt, k):
    a = pred == k; b = gt == k; s = a.sum() + b.sum()
    return (2 * np.logical_and(a, b).sum() / s) if s else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--ntest", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_cases", type=int, default=0, help="cap total cases (smoke test)")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    h5s = sorted(glob.glob("data/skmtea/kspace/**/*.h5", recursive=True))
    cases = [(h, f"data/skmtea/seg/{os.path.basename(h)[:-3]}_raw-data-track.nii.gz") for h in h5s]
    cases = [(h, s) for h, s in cases if os.path.exists(s)]
    if args.max_cases: cases = cases[:args.max_cases]
    rng = np.random.RandomState(args.seed); idx = rng.permutation(len(cases))
    test_i = set(idx[:args.ntest].tolist())
    print(f"{len(cases)} cases -> {len(cases)-args.ntest} train / {args.ntest} test  (dev={dev})", flush=True)

    # ----- build train tensors (clean magnitude) -----
    Xtr, Ytr = [], []
    for j, (h, s) in enumerate(cases):
        if j in test_i: continue
        cimg, lab = load_case_slices(h, s)
        for k in range(len(cimg)):
            Xtr.append(norm(np.abs(cimg[k])).astype(np.float32)); Ytr.append(lab[k])
    Xtr = torch.from_numpy(np.stack(Xtr))[:, None]                     # CPU float32 (move batches -> low GPU footprint)
    Ytr = torch.from_numpy(np.stack(Ytr).astype(np.int8))             # CPU int8
    print(f"train slices: {len(Xtr)}", flush=True)

    # ----- train -----
    net = UNet2D(NCLS).to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3)
    cw = torch.ones(NCLS, device=dev); cw[1:] = 3.0                    # upweight the thin structures
    bs = 8
    for ep in range(args.epochs):
        net.train(); perm = torch.randperm(len(Xtr)); tot = 0.0
        for i in range(0, len(Xtr), bs):
            b = perm[i:i + bs]; opt.zero_grad()
            xb = Xtr[b].to(dev); yb = Ytr[b].to(dev).long()
            lo = net(xb); ce = F.cross_entropy(lo, yb, weight=cw)
            pr = torch.softmax(lo, 1); dl = 0.0
            for k in range(1, NCLS):
                pk = pr[:, k]; gk = (yb == k).float()
                dl = dl + (1 - (2 * (pk * gk).sum() + 1) / (pk.sum() + gk.sum() + 1))
            loss = ce + dl / (NCLS - 1); loss.backward(); opt.step(); tot += loss.item()
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"  ep {ep:2d} loss {tot/(len(Xtr)//bs+1):.3f}", flush=True)

    # ----- eval: per-structure Dice at clean + R{2,4,6,8} (pooled over test slices) -----
    net.eval()
    conds = ["clean"] + [f"R{R}" for R in RS]
    inter = {c: {k: 0 for k in LAB} for c in conds}; union = {c: {k: 0 for k in LAB} for c in conds}
    cent = {k: [] for k in LAB}; ener = {k: {R: [] for R in RS} for k in LAB}
    with torch.no_grad():
        for j, (h, s) in enumerate(cases):
            if j not in test_i: continue
            cimg, lab = load_case_slices(h, s)
            for k in range(len(cimg)):
                c = cimg[k]; gt = lab[k]; mag_clean = norm(np.abs(c)).astype(np.float32)
                inputs = {"clean": mag_clean}
                for R in RS: inputs[f"R{R}"] = norm(undersample(c, MASKS[R])).astype(np.float32)
                for cond, mg in inputs.items():
                    x = torch.from_numpy(mg)[None, None].to(dev)
                    pr = net(x).argmax(1)[0].cpu().numpy()
                    for kk in LAB:
                        inter[cond][kk] += int(np.logical_and(pr == kk, gt == kk).sum())
                        union[cond][kk] += int((pr == kk).sum() + (gt == kk).sum())
                # predictors (from clean complex) per structure present in this slice
                for kk in LAB:
                    M = gt == kk
                    if M.sum() < 60: continue
                    cent[kk].append(spectral_centroid(mag_clean, M))
                    for R in RS: ener[kk][R].append(energy_removed(c, M, MASKS[R]))

    def D(cond, k): return (2 * inter[cond][k] / union[cond][k]) if union[cond][k] else np.nan
    per = {LAB[k]: {cond: D(cond, k) for cond in conds} for k in LAB}
    centroid = {LAB[k]: float(np.nanmean(cent[k])) if cent[k] else np.nan for k in LAB}
    drop = {LAB[k]: float(D("clean", k) - D("R8", k)) for k in LAB}         # clean->R8 Dice drop

    # (a) centroid -> Dice drop  (6 structures)
    ks = [k for k in LAB if not np.isnan(centroid[LAB[k]]) and not np.isnan(drop[LAB[k]])]
    cx = np.array([centroid[LAB[k]] for k in ks]); dy = np.array([drop[LAB[k]] for k in ks])
    law_a = {"spearman": float(spearmanr(cx, dy).correlation) if len(ks) > 2 else None,
             "pearson": float(pearsonr(cx, dy)[0]) if len(ks) > 2 else None, "n": len(ks)}

    # (b) energy removed -> Dice drop  (structure x R, using clean->R Dice drop)
    EX, EY = [], []
    for k in LAB:
        for R in RS:
            if ener[k][R]:
                EX.append(float(np.mean(ener[k][R]))); EY.append(D("clean", k) - D(f"R{R}", k))
    EX, EY = np.array(EX), np.array(EY); good = ~np.isnan(EY)
    law_b = {"pearson_log": float(pearsonr(np.log(EX[good] + 1e-9), EY[good])[0]) if good.sum() > 2 else None,
             "spearman": float(spearmanr(EX[good], EY[good]).correlation) if good.sum() > 2 else None, "n": int(good.sum())}

    out = {"n_cases": len(cases), "n_test": args.ntest, "per_structure_dice": per, "centroid": centroid,
           "dice_drop_clean_to_R8": drop, "law_a_centroid_to_dicedrop": law_a, "law_b_energy_to_dicedrop": law_b}
    os.makedirs(RES, exist_ok=True); json.dump(out, open(f"{RES}/knee_law.json", "w"), indent=2)
    print("\n===== per-structure Dice (clean -> R8) =====")
    for k in LAB: print(f"  {LAB[k]:11} clean {D('clean',k):.3f} -> R8 {D('R8',k):.3f}  (drop {drop[LAB[k]]:+.3f}, centroid {centroid[LAB[k]]:.4f})")
    print(f"\n(a) centroid -> Dice-drop : spearman {law_a['spearman']}, n={law_a['n']}")
    print(f"(b) energy   -> Dice-drop : pearson(log) {law_b['pearson_log']}, spearman {law_b['spearman']}, n={law_b['n']}")

    # ----- figure -----
    if not HAVE_MPL:
        print("matplotlib unavailable here -> skipped figure (json written; re-plot from knee_law.json)"); return
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    xR = [0] + RS
    for k in LAB: ax[0].plot(xR, [D(c, k) for c in conds], "-o", label=LAB[k])
    ax[0].set_xlabel("acceleration R"); ax[0].set_ylabel("Dice"); ax[0].set_title("Per-structure Dice vs R (real qDESS)", fontweight="bold"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    ax[1].scatter(cx, dy, s=70, color="#d73027")
    for k in ks: ax[1].annotate(LAB[k], (centroid[LAB[k]], drop[LAB[k]]), fontsize=7)
    ax[1].set_xlabel("spectral centroid (anatomy)"); ax[1].set_ylabel("Dice drop clean→R8")
    ax[1].set_title(f"(a) centroid → Dice drop\nSpearman {law_a['spearman']}", fontweight="bold"); ax[1].grid(alpha=.3)
    ax[2].scatter(EX[good], EY[good], s=45, color="#1a9850", alpha=.8)
    ax[2].set_xscale("log"); ax[2].set_xlabel("|energy removed| (in-region, log)"); ax[2].set_ylabel("Dice drop clean→R")
    ax[2].set_title(f"(b) energy → Dice drop\nr(log) {law_b['pearson_log']}", fontweight="bold"); ax[2].grid(alpha=.3, which="both")
    fig.suptitle(f"SKM-TEA real-k-space law test (n={len(cases)} cases, {args.ntest} test)", fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{PLOTS}/knee_law.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    print("wrote knee_law.png , knee_law.json")


if __name__ == "__main__":
    main()
