"""PROVE THE MITIGATION ON REAL K-SPACE. We've shown the MECHANISM (energy→error) and the mechanism→Dice link on real
qDESS knee k-space — but never that our METHOD (fragility-aware / mixed-R training) actually RECOVERS structure Dice on
real k-space. Here: baseline = segmenter trained on the CLEAN real coil-combined target; method = same net trained with
mixed-R augmentation (each batch, undersample the REAL k-space at a random R). Both tested per-structure at R{2,4,6,8}.
If mixed-R recovers the fragile knee structures under real-k-space undersampling → the method works on real data.
Multi-seed. Run: magicnet (PYTHONNOUSERSITE=1). -> outputs/results/condseg_knee.json"""
import os, glob, json, argparse, numpy as np, torch, torch.nn.functional as F
import sys; sys.path.insert(0, "scripts")
from knee_seg import UNet2D, vd_mask, load_case_slices, LAB, NCLS, dev

RS = [2, 4, 6, 8]
MASKS = {R: torch.tensor(vd_mask(512, R).astype(np.float32), device=dev) for R in RS}


def undersample_gpu(c, R):                                # c (B,512,512) complex on GPU -> zero-filled magnitude, 99.5-normed
    K = torch.fft.fftshift(torch.fft.fft2(c), dim=(-2, -1)); K = K * MASKS[R][None, :, None]
    x = torch.fft.ifft2(torch.fft.ifftshift(K, dim=(-2, -1)), dim=(-2, -1)).abs()
    q = torch.quantile(x.flatten(1), 0.995, dim=1).clamp(min=1e-6)
    return (x / q[:, None, None]).clamp(0, 2)


def clean_mag(c):
    x = c.abs(); q = torch.quantile(x.flatten(1), 0.995, dim=1).clamp(min=1e-6)
    return (x / q[:, None, None]).clamp(0, 2)


def gather(cases):
    C, Y = [], []
    for h, s in cases:
        cimg, lab = load_case_slices(h, s)
        for k in range(len(cimg)):
            C.append(cimg[k].astype(np.complex64)); Y.append(lab[k].astype(np.int8))
    return np.stack(C), np.stack(Y)


def train_eval(mode, Ctr, Ytr, Cte, Yte, epochs, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    net = UNet2D(NCLS).to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3)
    cw = torch.ones(NCLS, device=dev); cw[1:] = 3.0; bs = 8
    Xc = torch.from_numpy(Ctr); Yt = torch.from_numpy(Ytr.astype(np.int64))
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(Xc))
        for i in range(0, len(Xc), bs):
            b = perm[i:i + bs]; cb = Xc[b].to(dev); yb = Yt[b].to(dev)
            x = clean_mag(cb)[:, None] if mode == "clean" else undersample_gpu(cb, int(np.random.choice(RS)))[:, None]
            lo = net(x); ce = F.cross_entropy(lo, yb, weight=cw); pr = torch.softmax(lo, 1); dl = 0.
            for k in range(1, NCLS):
                pk = pr[:, k]; gk = (yb == k).float(); dl = dl + (1 - (2 * (pk * gk).sum() + 1) / (pk.sum() + gk.sum() + 1))
            loss = ce + dl / (NCLS - 1); opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); inter = {R: {k: 0 for k in LAB} for R in RS}; union = {R: {k: 0 for k in LAB} for R in RS}
    Xe = torch.from_numpy(Cte)
    with torch.no_grad():
        for R in RS:
            for i in range(0, len(Xe), bs):
                cb = Xe[i:i + bs].to(dev); g = Yte[i:i + bs]
                pr = net(undersample_gpu(cb, R)[:, None]).argmax(1).cpu().numpy()
                for j in range(len(g)):
                    for k in LAB:
                        inter[R][k] += int(np.logical_and(pr[j] == k, g[j] == k).sum()); union[R][k] += int((pr[j] == k).sum() + (g[j] == k).sum())
    Dk = lambda R, k: (2 * inter[R][k] / union[R][k]) if union[R][k] else np.nan
    return {R: {LAB[k]: Dk(R, k) for k in LAB} for R in RS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ntest", type=int, default=14); ap.add_argument("--ntrain", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=45); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    args = ap.parse_args()
    h5s = sorted(glob.glob("data/skmtea/kspace/**/*.h5", recursive=True))
    cases = [(h, f"data/skmtea/seg/{os.path.basename(h)[:-3]}_raw-data-track.nii.gz") for h in h5s]
    cases = [(h, s) for h, s in cases if os.path.exists(s)]
    idx = np.random.RandomState(0).permutation(len(cases))
    test = [cases[i] for i in idx[:args.ntest]]; train = [cases[i] for i in idx[args.ntest:args.ntest + args.ntrain]]
    print(f"real-k-space method test: {len(train)} train / {len(test)} test cases", flush=True)
    Ctr, Ytr = gather(train); Cte, Yte = gather(test)
    print(f"  {len(Ctr)} train / {len(Cte)} test slices", flush=True)
    res = {"clean": {R: {LAB[k]: [] for k in LAB} for R in RS}, "mixedr": {R: {LAB[k]: [] for k in LAB} for R in RS}}
    for seed in args.seeds:
        for mode in ["clean", "mixedr"]:
            r = train_eval(mode, Ctr, Ytr, Cte, Yte, args.epochs, seed)
            for R in RS:
                for k in LAB: res[mode][R][LAB[k]].append(r[R][LAB[k]])
            m = float(np.nanmean([r[8][LAB[k]] for k in LAB])); print(f"  seed{seed} [{mode}] R8 mean Dice {m:.4f}", flush=True)

    def mean(mode, R, o): return float(np.nanmean(res[mode][R][o]))
    out = {"per_R_mean_dice": {mode: {f"R{R}": {o: round(np.nanmean([mean(mode, R, o)], 0).item(), 4) for o in [LAB[k] for k in LAB]} for R in RS} for mode in res},
           "structure_avg": {mode: {f"R{R}": round(float(np.nanmean([mean(mode, R, LAB[k]) for k in LAB])), 4) for R in RS} for mode in res}}
    out["delta_mixedr_minus_clean"] = {f"R{R}": round(out["structure_avg"]["mixedr"][f"R{R}"] - out["structure_avg"]["clean"][f"R{R}"], 4) for R in RS}
    json.dump(out, open("outputs/results/condseg_knee.json", "w"), indent=2)
    print("\n=== structure-avg Dice per R (REAL knee k-space): clean-trained vs mixed-R ===")
    for R in RS:
        c = out["structure_avg"]["clean"][f"R{R}"]; m = out["structure_avg"]["mixedr"][f"R{R}"]
        print(f"  R{R}: clean {c:.4f} | mixed-R {m:.4f}  ({m-c:+.4f})")
    print("\n=== per-structure @R8 (clean -> mixed-R) ===")
    for k in LAB:
        c = mean("clean", 8, LAB[k]); m = mean("mixedr", 8, LAB[k]); print(f"  {LAB[k]:11} {c:.3f} -> {m:.3f}  ({m-c:+.3f})")
    print("wrote condseg_knee.json")


if __name__ == "__main__":
    main()
