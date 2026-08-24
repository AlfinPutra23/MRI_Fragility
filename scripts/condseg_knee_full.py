"""(a) RIGOROUS real-k-space method proof. Upgrades condseg_knee: 3 seeds + PER-CASE Dice + Wilcoxon + a fair
recon-then-segment KNEE baseline (so mixed-R isn't just beating a strawman clean-trained model). Three eval conditions
per case at each R on REAL qDESS knee k-space:
  A clean-trained @ zero-filled   (naive baseline / fragility)
  B clean-trained @ CS-recon      (recon-then-segment, the FAIR opponent)
  C mixed-R-trained @ zero-filled  (our method)
Per-case structure-avg Dice -> Wilcoxon(C vs B) and (C vs A). Run: magicnet. -> outputs/results/condseg_knee_full.json"""
import glob, os, json, argparse, numpy as np, torch, torch.nn.functional as F
from scipy.stats import wilcoxon
import sys; sys.path.insert(0, "scripts")
from condseg_knee import undersample_gpu, clean_mag, MASKS, RS
from knee_seg import UNet2D, LAB, NCLS, load_case_slices, dev


def _gk(sigma=0.8, sz=5):
    ax = torch.arange(sz, device=dev) - sz // 2; g = torch.exp(-(ax.float() ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    return torch.outer(g, g)[None, None]
GKER = _gk()


def cs_recon(c, R, niter=8):                          # POCS-Gaussian CS recon of undersampled complex -> normed magnitude
    m = MASKS[R]; K0 = torch.fft.fftshift(torch.fft.fft2(c), dim=(-2, -1)) * m[None, :, None]
    x = torch.fft.ifft2(torch.fft.ifftshift(K0, dim=(-2, -1)), dim=(-2, -1)).abs()
    for _ in range(niter):
        x = F.conv2d(x[:, None], GKER, padding=2)[:, 0]
        K = torch.fft.fftshift(torch.fft.fft2(x.to(torch.complex64)), dim=(-2, -1))
        K = torch.where(m[None, :, None].bool(), K0, K)
        x = torch.fft.ifft2(torch.fft.ifftshift(K, dim=(-2, -1)), dim=(-2, -1)).abs()
    q = torch.quantile(x.flatten(1), 0.995, dim=1).clamp(min=1e-6); return (x / q[:, None, None]).clamp(0, 2)


def gather(cases):
    C, Y, cid = [], [], []
    for ci, (h, s) in enumerate(cases):
        cimg, lab = load_case_slices(h, s)
        for k in range(len(cimg)):
            C.append(cimg[k].astype(np.complex64)); Y.append(lab[k].astype(np.int8)); cid.append(ci)
    return np.stack(C), np.stack(Y), np.array(cid)


def train(mode, Xc, Yt, epochs, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    net = UNet2D(NCLS).to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3)
    cw = torch.ones(NCLS, device=dev); cw[1:] = 3.0; bs = 8
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(Xc))
        for i in range(0, len(Xc), bs):
            b = perm[i:i + bs]; cb = Xc[b].to(dev); yb = Yt[b].to(dev)
            x = clean_mag(cb)[:, None] if mode == "clean" else undersample_gpu(cb, int(np.random.choice(RS)))[:, None]
            lo = net(x); ce = F.cross_entropy(lo, yb, weight=cw); pr = torch.softmax(lo, 1); dl = 0.
            for k in range(1, NCLS):
                pk = pr[:, k]; gk = (yb == k).float(); dl = dl + (1 - (2 * (pk * gk).sum() + 1) / (pk.sum() + gk.sum() + 1))
            (ce + dl / (NCLS - 1)).backward(); opt.step(); opt.zero_grad()
    return net


def percase_dice(net, Cte, Yte, cid, R, recon):        # -> per-case structure-avg Dice list
    bs = 8; ncase = cid.max() + 1
    inter = {c: {k: 0 for k in LAB} for c in range(ncase)}; union = {c: {k: 0 for k in LAB} for c in range(ncase)}
    Xe = torch.from_numpy(Cte)
    with torch.no_grad():
        for i in range(0, len(Xe), bs):
            cb = Xe[i:i + bs].to(dev); g = Yte[i:i + bs]; cc = cid[i:i + bs]
            x = cs_recon(cb, R)[:, None] if recon else undersample_gpu(cb, R)[:, None]
            pr = net(x).argmax(1).cpu().numpy()
            for j in range(len(g)):
                for k in LAB:
                    inter[cc[j]][k] += int(np.logical_and(pr[j] == k, g[j] == k).sum()); union[cc[j]][k] += int((pr[j] == k).sum() + (g[j] == k).sum())
    out = []
    for c in range(ncase):
        ds = [(2 * inter[c][k] / union[c][k]) for k in LAB if union[c][k] > 0]
        if ds: out.append(float(np.mean(ds)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ntest", type=int, default=14); ap.add_argument("--ntrain", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=45); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    h5s = sorted(glob.glob("data/skmtea/kspace/**/*.h5", recursive=True))
    cases = [(h, f"data/skmtea/seg/{os.path.basename(h)[:-3]}_raw-data-track.nii.gz") for h in h5s]
    cases = [(h, s) for h, s in cases if os.path.exists(s)]
    idx = np.random.RandomState(0).permutation(len(cases))
    test = [cases[i] for i in idx[:args.ntest]]; train_c = [cases[i] for i in idx[args.ntest:args.ntest + args.ntrain]]
    Ctr, Ytr, _ = gather(train_c); Cte, Yte, cid = gather(test)
    Xc = torch.from_numpy(Ctr); Yt = torch.from_numpy(Ytr.astype(np.int64))
    print(f"{len(Xc)} train / {len(Cte)} test slices, {cid.max()+1} test cases, seeds {args.seeds}", flush=True)
    # per-case Dice @R8, pooled over seeds (each seed contributes a per-case vector)
    A, B, C = [], [], []
    for seed in args.seeds:
        clean = train("clean", Xc, Yt, args.epochs, seed); mixr = train("mixedr", Xc, Yt, args.epochs, seed)
        A += percase_dice(clean, Cte, Yte, cid, 8, recon=False)
        B += percase_dice(clean, Cte, Yte, cid, 8, recon=True)
        C += percase_dice(mixr, Cte, Yte, cid, 8, recon=False)
        print(f"  seed{seed}: A(clean@zf) {np.mean(A[-14:]):.3f} | B(recon) {np.mean(B[-14:]):.3f} | C(mixedR) {np.mean(C[-14:]):.3f}", flush=True)
    A, B, C = np.array(A), np.array(B), np.array(C)
    def wil(a, b):
        n = min(len(a), len(b)); return float(wilcoxon(a[:n], b[:n]).pvalue) if n > 1 else None
    out = {"n_percase": len(C), "R8_structavg_mean": {"A_clean_zf": round(float(A.mean()), 4), "B_recon_then_seg": round(float(B.mean()), 4), "C_mixedR": round(float(C.mean()), 4)},
           "R8_std": {"A": round(float(A.std()), 4), "B": round(float(B.std()), 4), "C": round(float(C.std()), 4)},
           "delta_C_minus_A": round(float((C - A[:len(C)]).mean()), 4), "delta_C_minus_B": round(float((C - B[:len(C)]).mean()), 4),
           "wilcoxon_C_vs_A": wil(C, A), "wilcoxon_C_vs_B": wil(C, B)}
    json.dump(out, open("outputs/results/condseg_knee_full.json", "w"), indent=2)
    print("\n=== REAL knee k-space @R8 (per-case, seeds pooled) ===")
    print(f"  A clean@zero-filled  : {A.mean():.4f} ± {A.std():.4f}")
    print(f"  B recon-then-segment : {B.mean():.4f} ± {B.std():.4f}")
    print(f"  C mixed-R (method)   : {C.mean():.4f} ± {C.std():.4f}")
    print(f"  C-A {out['delta_C_minus_A']:+.4f} (p={out['wilcoxon_C_vs_A']:.1e}) | C-B {out['delta_C_minus_B']:+.4f} (p={out['wilcoxon_C_vs_B']:.1e})")
    print("wrote condseg_knee_full.json")


if __name__ == "__main__":
    main()
