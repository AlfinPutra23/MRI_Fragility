"""LAST NOVEL-METHOD SHOT — acceleration-invariant CONSISTENCY regularization. mixed-R works by making features robust to
degradation, but only IMPLICITLY. Enhance it: for each slice show the net TWO differently-degraded views (two random R)
and add a consistency loss forcing the SAME prediction across them (the organ is identical regardless of sampling).
Three arms to isolate the effect: base (1 view), two (2 views GT, no consistency = the fair control for 'more views'),
cons (2 views GT + consistency). Multi-seed. -> outputs/results/condseg_consistency.json . Run: magicnet."""
import glob, json, argparse, numpy as np, torch, torch.nn.functional as F
import sys; sys.path.insert(0, "scripts")
from condseg import UNet2D, undersample_gpu, load_split, RSET, dev
from seg_resnet_frag import ORG, NCLS, D
import labels as L


def seg_loss(lo, yb):
    ce = F.cross_entropy(lo, yb); pr = torch.softmax(lo, 1); dl = 0.
    pres = [k for k in ORG if (yb == k).any()]
    for k in pres:
        pk = pr[:, k]; gk = (yb == k).float(); dl = dl + (1 - (2 * (pk * gk).sum() + 1) / (pk.sum() + gk.sum() + 1))
    return ce + dl / max(len(pres), 1)


def train_eval(mode, cons_w, Xtr, Ytr, Xte, Yte, epochs, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    net = UNet2D(NCLS, 1).to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3); bs = 16
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), bs):
            b = perm[i:i + bs]; xb = Xtr[b].to(dev); yb = Ytr[b].to(dev)
            xu1 = undersample_gpu(xb, np.random.choice(RSET, len(b))).unsqueeze(1)
            lo1 = net(xu1)
            if mode == "base":
                loss = seg_loss(lo1, yb)
            else:
                xu2 = undersample_gpu(xb, np.random.choice(RSET, len(b))).unsqueeze(1)
                lo2 = net(xu2)
                loss = 0.5 * seg_loss(lo1, yb) + 0.5 * seg_loss(lo2, yb)           # GT on both views
                if mode == "cons":                                                # + acceleration-invariance
                    p1 = F.log_softmax(lo1, 1); p2 = F.log_softmax(lo2, 1)
                    cons = 0.5 * (F.kl_div(p1, p2.exp().detach(), reduction="batchmean") +
                                  F.kl_div(p2, p1.exp().detach(), reduction="batchmean"))
                    loss = loss + cons_w * cons
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); inter = {R: {k: 0 for k in ORG} for R in RSET}; union = {R: {k: 0 for k in ORG} for R in RSET}
    with torch.no_grad():
        for R in RSET:
            for i in range(0, len(Xte), bs):
                xb = Xte[i:i + bs].to(dev); yb = Yte[i:i + bs].numpy()
                pr = net(undersample_gpu(xb, [R] * len(xb)).unsqueeze(1)).argmax(1).cpu().numpy()
                for k in ORG:
                    inter[R][k] += int(np.logical_and(pr == k, yb == k).sum()); union[R][k] += int((pr == k).sum() + (yb == k).sum())
    Dk = lambda R, k: (2 * inter[R][k] / union[R][k]) if union[R][k] else np.nan
    return {R: float(np.nanmean([Dk(R, k) for k in L.TAIL])) for R in RSET}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=120); ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--max_slices", type=int, default=6000); ap.add_argument("--ntest", type=int, default=40)
    ap.add_argument("--cons_w", type=float, default=1.0); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    tr = sorted(glob.glob(f"{D}/imagesTr/*_0000.nii.gz"))[:args.n_train]
    Xtr, Ytr = load_split(tr, f"{D}/labelsTr")
    if len(Xtr) > args.max_slices:
        i = np.random.RandomState(0).choice(len(Xtr), args.max_slices, replace=False); Xtr, Ytr = Xtr[i], Ytr[i]
    te = sorted(glob.glob(f"{D}/imagesTs_clean/*_0000.nii.gz"))[:args.ntest]
    Xte, Yte = load_split(te, f"{D}/labelsTs")
    print(f"consistency: {len(Xtr)} train / {len(Xte)} test, seeds {args.seeds}, cons_w {args.cons_w}", flush=True)
    ARMS = [("base", 0.0), ("two", 0.0), ("cons", args.cons_w)]
    res = {a: {R: [] for R in RSET} for a, _ in ARMS}
    for seed in args.seeds:
        for a, cw in ARMS:
            t = train_eval(a, cw, Xtr, Ytr, Xte, Yte, args.epochs, seed)
            for R in RSET: res[a][R].append(t[R])
            print(f"  seed{seed} [{a}] R8 tail {t[8]:.4f}", flush=True)
    out = {"per_R_mean_std": {a: {f"R{R}": [round(float(np.mean(res[a][R])), 4), round(float(np.std(res[a][R])), 4)] for R in RSET} for a in res},
           "delta_cons_minus_two": {f"R{R}": round(float(np.mean(np.array(res["cons"][R]) - np.array(res["two"][R]))), 4) for R in RSET},
           "delta_cons_minus_base": {f"R{R}": round(float(np.mean(np.array(res["cons"][R]) - np.array(res["base"][R]))), 4) for R in RSET}}
    json.dump(out, open("outputs/results/condseg_consistency.json", "w"), indent=2)
    print("\n=== tail Dice per R (mean±std over seeds) ===")
    for R in RSET:
        s = "  ".join(f"{a}={np.mean(res[a][R]):.4f}±{np.std(res[a][R]):.3f}" for a, _ in ARMS)
        print(f"  R{R}: {s}  | cons-two {np.mean(np.array(res['cons'][R])-np.array(res['two'][R])):+.4f}")
    print("wrote condseg_consistency.json")


if __name__ == "__main__":
    main()
