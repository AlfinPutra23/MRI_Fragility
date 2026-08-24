"""NOVEL METHOD — degradation-aware (R-conditioned) segmentation. Hypothesis: mixed-R training makes ONE model that is
a *compromise* across all acceleration levels. If we additionally TELL the network the acceleration R it is currently
facing (an extra input channel), it can SPECIALIZE its inference per-R -> beating plain mixed-R (the strongest known
lever, +0.088). This directly operationalizes Double Jeopardy: the net knows how much high-freq is missing. It's also
realistic -- R is known at acquisition time. Compares mixed-R baseline (1-ch) vs R-conditioned (2-ch), same data/seed,
per-organ Dice at each R. Run: magicnet (PYTHONNOUSERSITE=1). -> outputs/results/condseg.json"""
import os, glob, json, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import sys; sys.path.insert(0, "scripts")
from seg_resnet_frag import keep_abdo, norm, load_case, ORG, NCLS, D, SZ   # reuse data pipeline (GPU-batched resize)
import labels as L
from kspace import vd_cartesian_mask

dev = "cuda"; RSET = [1, 2, 4, 6, 8]


class DoubleConv(nn.Module):
    def __init__(s, i, o):
        super().__init__(); s.c = nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.InstanceNorm2d(o), nn.LeakyReLU(inplace=True),
                                                 nn.Conv2d(o, o, 3, padding=1), nn.InstanceNorm2d(o), nn.LeakyReLU(inplace=True))
    def forward(s, x): return s.c(x)


class UNet2D(nn.Module):
    def __init__(s, ncls, in_ch=1, ch=(32, 64, 128, 256)):
        super().__init__()
        s.d1, s.d2, s.d3 = DoubleConv(in_ch, ch[0]), DoubleConv(ch[0], ch[1]), DoubleConv(ch[1], ch[2])
        s.bott = DoubleConv(ch[2], ch[3]); s.pool = nn.MaxPool2d(2)
        s.u3 = nn.ConvTranspose2d(ch[3], ch[2], 2, 2); s.c3 = DoubleConv(ch[3], ch[2])
        s.u2 = nn.ConvTranspose2d(ch[2], ch[1], 2, 2); s.c2 = DoubleConv(ch[2], ch[1])
        s.u1 = nn.ConvTranspose2d(ch[1], ch[0], 2, 2); s.c1 = DoubleConv(ch[1], ch[0])
        s.out = nn.Conv2d(ch[0], ncls, 1)
    def forward(s, x):
        e1 = s.d1(x); e2 = s.d2(s.pool(e1)); e3 = s.d3(s.pool(e2)); b = s.bott(s.pool(e3))
        d = s.c3(torch.cat([s.u3(b), e3], 1)); d = s.c2(torch.cat([s.u2(d), e2], 1)); d = s.c1(torch.cat([s.u1(d), e1], 1))
        return s.out(d)


MASKS = {R: torch.tensor(vd_cartesian_mask(SZ, R).astype(np.float32), device=dev) for R in RSET}   # (SZ,) over PE rows


def undersample_gpu(x, Rs):    # x:(B,SZ,SZ) real -> zero-filled magnitude at each sample's R
    K = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
    m = torch.stack([MASKS[int(r)] for r in Rs])                 # (B,SZ) row mask
    K = K * m[:, :, None]
    return torch.fft.ifft2(torch.fft.ifftshift(K, dim=(-2, -1)), dim=(-2, -1)).abs()


def rchan(Rs):                 # R-conditioning channel: constant map = R/8 (fraction of full acceleration)
    return torch.tensor([float(r) / 8.0 for r in Rs], device=dev).view(-1, 1, 1, 1).expand(-1, 1, SZ, SZ)


def load_split(paths, labdir):
    Xs, Ys = [], []
    for p in paths:
        lb = f"{labdir}/" + os.path.basename(p).replace("_0000", "")
        if not os.path.exists(lb): continue
        X, Y = load_case(p, lb)
        if X is not None: Xs.append(X); Ys.append(Y)
    return torch.from_numpy(np.concatenate(Xs)), torch.from_numpy(np.concatenate(Ys))


def train_eval(cond, Xtr, Ytr, Xte, Yte, epochs):
    in_ch = 1 if cond == "none" else 2
    net = UNet2D(NCLS, in_ch).to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3); bs = 16
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), bs):
            b = perm[i:i + bs]; xb = Xtr[b].to(dev); yb = Ytr[b].to(dev)
            Rs = np.random.choice(RSET, len(b))                   # mixed-R: random acceleration per slice
            xu = undersample_gpu(xb, Rs).unsqueeze(1)
            inp = xu if cond == "none" else torch.cat([xu, rchan(Rs)], 1)
            opt.zero_grad(); lo = net(inp); ce = F.cross_entropy(lo, yb); pr = torch.softmax(lo, 1); dl = 0.0
            pres = [k for k in ORG if (yb == k).any()]
            for k in pres:
                pk = pr[:, k]; gk = (yb == k).float(); dl = dl + (1 - (2 * (pk * gk).sum() + 1) / (pk.sum() + gk.sum() + 1))
            loss = ce + dl / max(len(pres), 1); loss.backward(); opt.step()
        if ep % 10 == 0 or ep == epochs - 1: print(f"  [{cond}] ep{ep} loss {loss.item():.3f}", flush=True)
    # eval per-organ Dice at each R
    net.eval(); inter = {R: {k: 0 for k in ORG} for R in RSET}; union = {R: {k: 0 for k in ORG} for R in RSET}
    with torch.no_grad():
        for R in RSET:
            for i in range(0, len(Xte), bs):
                xb = Xte[i:i + bs].to(dev); yb = Yte[i:i + bs].numpy()
                xu = undersample_gpu(xb, [R] * len(xb)).unsqueeze(1)
                inp = xu if cond == "none" else torch.cat([xu, rchan([R] * len(xb))], 1)
                pr = net(inp).argmax(1).cpu().numpy()
                for k in ORG:
                    inter[R][k] += int(np.logical_and(pr == k, yb == k).sum()); union[R][k] += int((pr == k).sum() + (yb == k).sum())
    D_ = lambda R, k: (2 * inter[R][k] / union[R][k]) if union[R][k] else np.nan
    return {R: {L.ABDO[k]: D_(R, k) for k in ORG} for R in RSET}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=120); ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--max_slices", type=int, default=7000); ap.add_argument("--ntest", type=int, default=40)
    args = ap.parse_args(); torch.manual_seed(0); np.random.seed(0)

    tr = sorted(glob.glob(f"{D}/imagesTr/*_0000.nii.gz"))[:args.n_train]
    Xtr, Ytr = load_split(tr, f"{D}/labelsTr")
    if len(Xtr) > args.max_slices:
        i = np.random.RandomState(0).choice(len(Xtr), args.max_slices, replace=False); Xtr, Ytr = Xtr[i], Ytr[i]
    te = sorted(glob.glob(f"{D}/imagesTs_clean/*_0000.nii.gz"))[:args.ntest]
    Xte, Yte = load_split(te, f"{D}/labelsTs")
    print(f"cond-seg: {len(Xtr)} train / {len(Xte)} test slices", flush=True)

    TAIL = [L.ABDO[k] for k in L.TAIL]
    res = {}
    for cond in ["none", "R"]:
        torch.manual_seed(0); np.random.seed(0)                    # SAME seed/data -> isolates the conditioning
        res[cond] = train_eval(cond, Xtr, Ytr, Xte, Yte, args.epochs)

    def tail(cond, R): return float(np.nanmean([res[cond][R][o] for o in TAIL]))
    out = {"per_R_tail_dice": {cond: {f"R{R}": tail(cond, R) for R in RSET} for cond in res},
           "delta_Rcond_minus_baseline": {f"R{R}": tail("R", R) - tail("none", R) for R in RSET},
           "per_organ_R8": {o: {"baseline": res["none"][8][o], "Rcond": res["R"][8][o]} for o in [L.ABDO[k] for k in ORG]}}
    json.dump(out, open("outputs/results/condseg.json", "w"), indent=2)
    print("\n=== tail Dice per R (baseline mixed-R vs R-conditioned) ===")
    for R in RSET: print(f"  R{R}: baseline {tail('none',R):.4f}  |  R-cond {tail('R',R):.4f}  ({tail('R',R)-tail('none',R):+.4f})")
    print("wrote condseg.json")


if __name__ == "__main__":
    main()
