"""NOVEL METHOD v2 — FiLM-conditioned (degradation-aware) segmentation, MULTI-SEED. The input-channel R-conditioning gave
a small, mechanism-consistent +0.012 @R8 (single seed, condseg.py). FiLM injects the acceleration R into EVERY feature
map's scale/shift -> a far stronger conditioning than a constant channel the net can ignore. Multi-seed (n=3): does FiLM
amplify the effect AND hold across seeds? Baseline mixed-R vs FiLM-conditioned mixed-R, paired per seed. Run: magicnet.
-> outputs/results/condseg_film.json"""
import glob, json, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import sys; sys.path.insert(0, "scripts")
from condseg import DoubleConv, UNet2D, undersample_gpu, load_split, RSET, dev
from seg_resnet_frag import ORG, NCLS, D
import labels as L


class FiLM(nn.Module):
    def __init__(s, c): super().__init__(); s.f = nn.Linear(16, 2 * c)
    def forward(s, x, e):
        g, b = s.f(e).chunk(2, 1); return x * (1 + g[:, :, None, None]) + b[:, :, None, None]


class FiLMUNet2D(nn.Module):
    """UNet whose every stage is FiLM-modulated by the acceleration R (via a shared 16-d embedding of R/8)."""
    def __init__(s, ncls, ch=(32, 64, 128, 256)):
        super().__init__()
        s.emb = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU())
        s.d1, s.d2, s.d3 = DoubleConv(1, ch[0]), DoubleConv(ch[0], ch[1]), DoubleConv(ch[1], ch[2])
        s.f1, s.f2, s.f3 = FiLM(ch[0]), FiLM(ch[1]), FiLM(ch[2])
        s.bott = DoubleConv(ch[2], ch[3]); s.fb = FiLM(ch[3]); s.pool = nn.MaxPool2d(2)
        s.u3 = nn.ConvTranspose2d(ch[3], ch[2], 2, 2); s.c3 = DoubleConv(ch[3], ch[2])
        s.u2 = nn.ConvTranspose2d(ch[2], ch[1], 2, 2); s.c2 = DoubleConv(ch[2], ch[1])
        s.u1 = nn.ConvTranspose2d(ch[1], ch[0], 2, 2); s.c1 = DoubleConv(ch[1], ch[0])
        s.out = nn.Conv2d(ch[0], ncls, 1)
    def forward(s, x, r):
        e = s.emb(r)
        e1 = s.f1(s.d1(x), e); e2 = s.f2(s.d2(s.pool(e1)), e); e3 = s.f3(s.d3(s.pool(e2)), e); b = s.fb(s.bott(s.pool(e3)), e)
        d = s.c3(torch.cat([s.u3(b), e3], 1)); d = s.c2(torch.cat([s.u2(d), e2], 1)); d = s.c1(torch.cat([s.u1(d), e1], 1))
        return s.out(d)


def train_eval(cond, Xtr, Ytr, Xte, Yte, epochs, seed):
    torch.manual_seed(seed); np.random.seed(seed)                  # same seed -> paired mixed-R sequence per (cond,seed)
    net = (FiLMUNet2D(NCLS) if cond == "film" else UNet2D(NCLS, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3); bs = 16
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), bs):
            b = perm[i:i + bs]; xb = Xtr[b].to(dev); yb = Ytr[b].to(dev)
            Rs = np.random.choice(RSET, len(b))
            xu = undersample_gpu(xb, Rs).unsqueeze(1)
            r = torch.tensor([float(x) / 8. for x in Rs], device=dev).view(-1, 1)
            lo = net(xu, r) if cond == "film" else net(xu)
            opt.zero_grad(); ce = F.cross_entropy(lo, yb); pr = torch.softmax(lo, 1); dl = 0.
            pres = [k for k in ORG if (yb == k).any()]
            for k in pres:
                pk = pr[:, k]; gk = (yb == k).float(); dl = dl + (1 - (2 * (pk * gk).sum() + 1) / (pk.sum() + gk.sum() + 1))
            loss = ce + dl / max(len(pres), 1); loss.backward(); opt.step()
    net.eval(); inter = {R: {k: 0 for k in ORG} for R in RSET}; union = {R: {k: 0 for k in ORG} for R in RSET}
    with torch.no_grad():
        for R in RSET:
            for i in range(0, len(Xte), bs):
                xb = Xte[i:i + bs].to(dev); yb = Yte[i:i + bs].numpy()
                xu = undersample_gpu(xb, [R] * len(xb)).unsqueeze(1)
                r = torch.full((len(xb), 1), R / 8., device=dev)
                lo = net(xu, r) if cond == "film" else net(xu)
                pr = lo.argmax(1).cpu().numpy()
                for k in ORG:
                    inter[R][k] += int(np.logical_and(pr == k, yb == k).sum()); union[R][k] += int((pr == k).sum() + (yb == k).sum())
    Dk = lambda R, k: (2 * inter[R][k] / union[R][k]) if union[R][k] else np.nan
    return {R: float(np.nanmean([Dk(R, k) for k in L.TAIL])) for R in RSET}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=120); ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--max_slices", type=int, default=6000); ap.add_argument("--ntest", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    tr = sorted(glob.glob(f"{D}/imagesTr/*_0000.nii.gz"))[:args.n_train]
    Xtr, Ytr = load_split(tr, f"{D}/labelsTr")
    if len(Xtr) > args.max_slices:
        i = np.random.RandomState(0).choice(len(Xtr), args.max_slices, replace=False); Xtr, Ytr = Xtr[i], Ytr[i]
    te = sorted(glob.glob(f"{D}/imagesTs_clean/*_0000.nii.gz"))[:args.ntest]
    Xte, Yte = load_split(te, f"{D}/labelsTs")
    print(f"FiLM cond-seg: {len(Xtr)} train / {len(Xte)} test, seeds {args.seeds}", flush=True)
    res = {"none": {R: [] for R in RSET}, "film": {R: [] for R in RSET}}
    for seed in args.seeds:
        for cond in ["none", "film"]:
            t = train_eval(cond, Xtr, Ytr, Xte, Yte, args.epochs, seed)
            for R in RSET: res[cond][R].append(t[R])
            print(f"  seed{seed} [{cond}] R8 tail {t[8]:.4f}", flush=True)
    out = {"per_R_mean_std": {c: {f"R{R}": [round(float(np.mean(res[c][R])), 4), round(float(np.std(res[c][R])), 4)] for R in RSET} for c in res},
           "delta_film_minus_baseline_mean": {f"R{R}": round(float(np.mean(np.array(res["film"][R]) - np.array(res["none"][R]))), 4) for R in RSET},
           "seeds": args.seeds}
    json.dump(out, open("outputs/results/condseg_film.json", "w"), indent=2)
    print("\n=== tail Dice per R (mean±std over seeds) ===")
    for R in RSET:
        nb = np.array(res["none"][R]); fb = np.array(res["film"][R])
        print(f"  R{R}: baseline {nb.mean():.4f}±{nb.std():.4f} | FiLM {fb.mean():.4f}±{fb.std():.4f}  (Δ {(fb-nb).mean():+.4f})")
    print("wrote condseg_film.json")


if __name__ == "__main__":
    main()
