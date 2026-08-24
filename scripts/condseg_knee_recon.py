"""(a) STRONG recon baseline. arXiv:2508.18975 found data-consistency LEARNED recon-then-segment is the best two-stage
method — our earlier recon baseline was only classical CS/POCS. Here we train an UNROLLED data-consistency recon
(learned denoiser + k-space DC, VarNet/MoDL-style) on real qDESS knee k-space, then recon-then-segment with a clean-
trained net, and test whether mixed-R STILL beats it. Everything on prepped (per-slice unit-scaled) complex for a
consistent normalization. Per-case Wilcoxon. Run: magicnet. -> outputs/results/condseg_knee_recon.json"""
import glob, os, json, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import wilcoxon
import sys; sys.path.insert(0, "scripts")
from condseg_knee import MASKS, RS
from knee_seg import UNet2D, DoubleConv, LAB, NCLS, load_case_slices, dev


class ReconNet(nn.Module):                          # residual denoiser for the unrolled recon
    def __init__(s, ch=(24, 48, 96)):
        super().__init__(); s.d1, s.d2 = DoubleConv(1, ch[0]), DoubleConv(ch[0], ch[1]); s.bott = DoubleConv(ch[1], ch[2]); s.pool = nn.MaxPool2d(2)
        s.u2 = nn.ConvTranspose2d(ch[2], ch[1], 2, 2); s.c2 = DoubleConv(ch[2], ch[1])
        s.u1 = nn.ConvTranspose2d(ch[1], ch[0], 2, 2); s.c1 = DoubleConv(ch[1], ch[0]); s.out = nn.Conv2d(ch[0], 1, 1)
    def forward(s, x):
        e1 = s.d1(x); e2 = s.d2(s.pool(e1)); b = s.bott(s.pool(e2))
        d = s.c2(torch.cat([s.u2(b), e2], 1)); d = s.c1(torch.cat([s.u1(d), e1], 1)); return (x + s.out(d)).clamp(min=0)


def prep(c):
    q = torch.quantile(c.abs().flatten(1), 0.995, dim=1).clamp(min=1e-6); return c / q[:, None, None]
def zf_mag(c, R):
    K0 = torch.fft.fftshift(torch.fft.fft2(c), dim=(-2, -1)) * MASKS[R][None, :, None]
    return torch.fft.ifft2(torch.fft.ifftshift(K0, dim=(-2, -1)), dim=(-2, -1)).abs().clamp(0, 2)
def unrolled(net, c, R, niter=3):                   # learned denoise + k-space data consistency
    m = MASKS[R]; K0 = torch.fft.fftshift(torch.fft.fft2(c), dim=(-2, -1)) * m[None, :, None]
    xc = torch.fft.ifft2(torch.fft.ifftshift(K0, dim=(-2, -1)), dim=(-2, -1)); x = xc.abs(); ph = xc.angle()
    for _ in range(niter):
        x = net(x[:, None])[:, 0]; xc = x * torch.exp(1j * ph)
        K = torch.fft.fftshift(torch.fft.fft2(xc), dim=(-2, -1)); K = torch.where(m[None, :, None].bool(), K0, K)
        xc = torch.fft.ifft2(torch.fft.ifftshift(K, dim=(-2, -1)), dim=(-2, -1)); x = xc.abs(); ph = xc.angle()
    return x.clamp(0, 2)


def gather(cases):
    C, Y, cid = [], [], []
    for ci, (h, s) in enumerate(cases):
        cimg, lab = load_case_slices(h, s)
        for k in range(len(cimg)): C.append(cimg[k].astype(np.complex64)); Y.append(lab[k].astype(np.int8)); cid.append(ci)
    return np.stack(C), np.stack(Y), np.array(cid)


def seg_step(net, x, yb):
    lo = net(x); ce = F.cross_entropy(lo, yb, weight=SEGW); pr = torch.softmax(lo, 1); dl = 0.
    for k in range(1, NCLS):
        pk = pr[:, k]; gk = (yb == k).float(); dl = dl + (1 - (2 * (pk * gk).sum() + 1) / (pk.sum() + gk.sum() + 1))
    return ce + dl / (NCLS - 1)


SEGW = None
def train_recon(Xc, epochs, seed):
    torch.manual_seed(seed); net = ReconNet().to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3); bs = 8
    for ep in range(epochs):
        perm = torch.randperm(len(Xc))
        for i in range(0, len(Xc), bs):
            cb = prep(Xc[perm[i:i + bs]].to(dev)); R = int(np.random.choice(RS))
            xr = unrolled(net, cb, R); loss = F.l1_loss(xr, cb.abs().clamp(0, 2)); opt.zero_grad(); loss.backward(); opt.step()
    return net


def train_seg(mode, Xc, Yt, epochs, seed):
    torch.manual_seed(seed); np.random.seed(seed); net = UNet2D(NCLS).to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3); bs = 8
    for ep in range(epochs):
        perm = torch.randperm(len(Xc))
        for i in range(0, len(Xc), bs):
            b = perm[i:i + bs]; cb = prep(Xc[b].to(dev)); yb = Yt[b].to(dev)
            x = cb.abs().clamp(0, 2)[:, None] if mode == "clean" else zf_mag(cb, int(np.random.choice(RS)))[:, None]
            loss = seg_step(net, x, yb); opt.zero_grad(); loss.backward(); opt.step()
    return net


def percase(segnet, recnet, Cte, Yte, cid, R, kind):     # kind: zf | recon
    bs = 8; nc = cid.max() + 1; inter = {c: {k: 0 for k in LAB} for c in range(nc)}; union = {c: {k: 0 for k in LAB} for c in range(nc)}
    Xe = torch.from_numpy(Cte)
    with torch.no_grad():
        for i in range(0, len(Xe), bs):
            cb = prep(Xe[i:i + bs].to(dev)); g = Yte[i:i + bs]; cc = cid[i:i + bs]
            x = (unrolled(recnet, cb, R) if kind == "recon" else zf_mag(cb, R))[:, None]
            pr = segnet(x).argmax(1).cpu().numpy()
            for j in range(len(g)):
                for k in LAB: inter[cc[j]][k] += int(np.logical_and(pr[j] == k, g[j] == k).sum()); union[cc[j]][k] += int((pr[j] == k).sum() + (g[j] == k).sum())
    return [float(np.mean([2 * inter[c][k] / union[c][k] for k in LAB if union[c][k] > 0])) for c in range(nc) if any(union[c][k] > 0 for k in LAB)]


def main():
    global SEGW
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=35); ap.add_argument("--recon_epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1]); args = ap.parse_args()
    SEGW = torch.ones(NCLS, device=dev); SEGW[1:] = 3.0
    h5s = sorted(glob.glob("data/skmtea/kspace/**/*.h5", recursive=True))
    cases = [(h, f"data/skmtea/seg/{os.path.basename(h)[:-3]}_raw-data-track.nii.gz") for h in h5s]
    cases = [(h, s) for h, s in cases if os.path.exists(s)]; idx = np.random.RandomState(0).permutation(len(cases))
    test = [cases[i] for i in idx[:14]]; train = [cases[i] for i in idx[14:44]]
    Ctr, Ytr, _ = gather(train); Cte, Yte, cid = gather(test); Xc = torch.from_numpy(Ctr); Yt = torch.from_numpy(Ytr.astype(np.int64))
    print(f"{len(Xc)} train / {len(Cte)} test, {cid.max()+1} cases", flush=True)
    A, Bcs, Blr, C = [], [], [], []
    from condseg_knee_full import cs_recon  # classical CS for reference
    for seed in args.seeds:
        rec = train_recon(Xc, args.recon_epochs, seed); clean = train_seg("clean", Xc, Yt, args.epochs, seed); mixr = train_seg("mixedr", Xc, Yt, args.epochs, seed)
        A += percase(clean, rec, Cte, Yte, cid, 8, "zf")
        Blr += percase(clean, rec, Cte, Yte, cid, 8, "recon")
        C += percase(mixr, rec, Cte, Yte, cid, 8, "zf")
        print(f"  seed{seed}: clean@zf {np.mean(A[-14:]):.3f} | clean@LEARNED-recon {np.mean(Blr[-14:]):.3f} | mixedR {np.mean(C[-14:]):.3f}", flush=True)
    A, Blr, C = np.array(A), np.array(Blr), np.array(C)
    wil = lambda a, b: float(wilcoxon(a[:min(len(a), len(b))], b[:min(len(a), len(b))]).pvalue)
    out = {"n_percase": len(C), "R8": {"A_clean_zf": round(float(A.mean()), 4), "B_learned_recon_then_seg": round(float(Blr.mean()), 4), "C_mixedR": round(float(C.mean()), 4)},
           "delta_C_minus_learnedrecon": round(float((C - Blr[:len(C)]).mean()), 4), "wilcoxon_C_vs_learnedrecon": wil(C, Blr)}
    json.dump(out, open("outputs/results/condseg_knee_recon.json", "w"), indent=2)
    print("\n=== REAL knee k-space @R8 vs LEARNED data-consistency recon ===")
    print(f"  clean@zero-filled       : {A.mean():.4f}")
    print(f"  clean@LEARNED-recon (DC): {Blr.mean():.4f}   <- the strong opponent (2508.18975-style)")
    print(f"  mixed-R (ours)          : {C.mean():.4f}")
    print(f"  mixed-R vs learned-recon: {out['delta_C_minus_learnedrecon']:+.4f}  (p={out['wilcoxon_C_vs_learnedrecon']:.1e})")
    print("wrote condseg_knee_recon.json")


if __name__ == "__main__":
    main()
