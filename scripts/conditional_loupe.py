"""TARGET-CONDITIONED LOUPE (TC-LOUPE) — the novel method, grounded in the audit.

Audit finding: every fragility-PRIOR method fails (FG-TDR, FG-Seg, fragA/B masks, fgLOUPE, weighting sweep), but a
LEARNED task-driven sampling mask (LOUPE) WINS (+0.043 tail). Lesson: the lever is acquisition-side + task-driven, not a
hand prior. TC-LOUPE learns a SEPARATE task-driven mask per clinical target organ, and tells the segmenter which
acquisition it is seeing (mask as a 2nd input channel). At deploy: 'given your target organ, here is the acceleration-
optimal sampling scheme.'

Two arms, IDENTICAL 2-channel architecture / budget / data / seeds -> isolates the effect of per-target conditioning:
  generic     : ONE learned mask for all organs (= LOUPE + mask-channel segmenter)
  conditional : per-target learned masks (this method)
MAKE-OR-BREAK: conditional tail Dice > generic tail Dice (paired over seeds).

  python conditional_loupe.py --seeds 0 1 2 --R 8 --epochs 60
-> outputs/results/conditional_loupe.json
"""
import os, json, argparse, numpy as np, torch, torch.nn as nn
import sys; sys.path.insert(0, "scripts")
from loupe import undersample, rescale_probs
from b1_joint import DoubleConv, dice_ce, remap_labels, NCLS, ABDO_IDS, REMAP
import labels as L
from scipy.stats import wilcoxon

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TAIL_CLS = [REMAP[o] for o in L.TAIL]                     # fragile target classes (1..13 space)


class CondLOUPE(nn.Module):
    """Per-target learnable PE-line masks. n_targets=1 -> generic single mask (baseline)."""
    def __init__(self, n_lines, R, n_targets, slope=5.0, tau=0.5, acs_frac=0.08):
        super().__init__()
        self.n, self.R, self.slope, self.tau = n_lines, R, slope, tau
        self.logits = nn.Parameter(torch.zeros(n_targets, n_lines))
        c = n_lines // 2; n_acs = max(int(round(n_lines * acs_frac)), 4)
        acs = torch.zeros(n_lines); acs[c - n_acs // 2: c + (n_acs - n_acs // 2)] = 1.0
        self.register_buffer("acs", acs)

    def probs(self, t):
        p = torch.sigmoid(self.slope * self.logits[t])
        target = max((self.n / self.R - self.acs.sum().item()), 1.0) / (self.n - self.acs.sum().item())
        p = rescale_probs(p, target)
        return torch.maximum(p, self.acs)

    def forward(self, t, training=True):
        p = self.probs(t)
        if training:
            u = torch.rand_like(p).clamp(1e-6, 1 - 1e-6)
            lp = torch.log(p) - torch.log(1 - p)
            m = torch.sigmoid((lp + torch.log(u) - torch.log(1 - u)) / self.tau)
        else:
            k = int(round(self.n / self.R)); idx = torch.topk(p, k).indices
            m = torch.zeros_like(p); m[idx] = 1.0
        return torch.maximum(m, self.acs)


class UNetCond(nn.Module):
    """UNet2D with 2-channel input (undersampled image + broadcast sampling mask)."""
    def __init__(s, ncls, in_ch=2, ch=(32, 64, 128, 256)):
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


def mask_channel(m1d, N):                                 # broadcast 1D PE mask -> (N,N) image-space channel (PE = axis -2)
    return m1d.view(N, 1).expand(N, N)


def load_data(root, max_slices, seed):
    tr = np.load(f"{root}/train.npz"); X = torch.tensor(tr["images"].astype(np.float32)); Y = torch.tensor(tr["labels"].astype(np.int64))
    te = np.load(f"{root}/test.npz"); Xt = torch.tensor(te["images"].astype(np.float32)); Yt = torch.tensor(te["labels"].astype(np.int64))
    if max_slices and X.shape[0] > max_slices:
        Yg = Y.to(torch.int16).to(dev); ws = torch.ones(Yg.shape[0], device=dev)
        for oid, b in [(6, 25.), (4, 2.), (12, 2.), (13, 2.), (17, .5)]: ws[(Yg == oid).flatten(1).any(1)] += b
        idx = torch.multinomial(ws, max_slices, replacement=True).cpu(); del Yg; torch.cuda.empty_cache(); X, Y = X[idx], Y[idx]
    X = X.to(dev); Y = remap_labels(Y.to(dev)).to(torch.uint8); Yt = remap_labels(Yt.to(dev)).cpu()
    return X, Y, Xt, Yt


def train_arm(conditional, X, Y, N, R, epochs, bs, lr, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    n_tgt = len(TAIL_CLS) if conditional else 1
    mask = CondLOUPE(N, R, n_tgt).to(dev)
    net = UNetCond(NCLS, in_ch=2).to(dev)
    opt = torch.optim.Adam(list(net.parameters()) + list(mask.parameters()), lr=lr)
    for ep in range(epochs):
        net.train(); perm = torch.randperm(X.shape[0], device=dev)
        for i in range(0, X.shape[0], bs):
            idx = perm[i:i + bs]; xb = X[idx].unsqueeze(1); yb = Y[idx].long()
            if conditional:
                ti = int(np.random.randint(n_tgt)); tcls = TAIL_CLS[ti]           # sample a target organ
                m = mask(ti, training=True)
                w = torch.ones(NCLS, device=dev); w[1:] = 1.0; w[tcls] = 5.0        # focus loss on the target organ
            else:
                m = mask(0, training=True)
                w = torch.ones(NCLS, device=dev); w[1:] = 3.0                       # standard fragility-agnostic upweight (matches LOUPE baseline)
            xu = undersample(xb, m)
            mc = mask_channel(m, N).unsqueeze(0).unsqueeze(0).expand(xb.shape[0], 1, N, N)
            logits = net(torch.cat([xu, mc], 1))
            loss = dice_ce(logits, yb, w)
            opt.zero_grad(); loss.backward(); opt.step()
    return net, mask


def eval_arm(conditional, net, mask, Xt, Yt, N, R, bs):
    net.eval(); masks = {}
    per_organ = {}
    with torch.no_grad():
        if conditional:
            # for each fragile target, deploy ITS mask; measure THAT organ's Dice
            for ti, tcls in enumerate(TAIL_CLS):
                m = mask(ti, training=False); masks[ti] = m.detach().cpu().numpy()
                mc = mask_channel(m, N)
                inter = den = 0.0
                for i in range(0, Xt.shape[0], bs):
                    xb = Xt[i:i + bs].unsqueeze(1).to(dev); yb = Yt[i:i + bs].to(dev)
                    xu = undersample(xb, m)
                    mcb = mc.unsqueeze(0).unsqueeze(0).expand(xb.shape[0], 1, N, N)
                    pred = net(torch.cat([xu, mcb], 1)).argmax(1)
                    inter += (2 * ((pred == tcls) & (yb == tcls)).sum()).item()
                    den += ((pred == tcls).sum() + (yb == tcls).sum()).item()
                oid = [o for o in L.TAIL][ti]  # careful: TAIL_CLS order == list(L.TAIL) order
                per_organ[L.ABDO[oid]] = inter / den if den > 0 else float("nan")
            tail = float(np.nanmean(list(per_organ.values())))
            large = float("nan")   # conditional method targets fragile organs; large-organ eval not the point
        else:
            m = mask(0, training=False); masks[0] = m.detach().cpu().numpy(); mc = mask_channel(m, N)
            inter = torch.zeros(NCLS); den = torch.zeros(NCLS)
            for i in range(0, Xt.shape[0], bs):
                xb = Xt[i:i + bs].unsqueeze(1).to(dev); yb = Yt[i:i + bs].to(dev)
                xu = undersample(xb, m); mcb = mc.unsqueeze(0).unsqueeze(0).expand(xb.shape[0], 1, N, N)
                pred = net(torch.cat([xu, mcb], 1)).argmax(1)
                for c in range(1, NCLS):
                    inter[c] += (2 * ((pred == c) & (yb == c)).sum()).item(); den[c] += ((pred == c).sum() + (yb == c).sum()).item()
            dsc = {ABDO_IDS[c - 1]: (inter[c] / den[c]).item() if den[c] > 0 else float("nan") for c in range(1, NCLS)}
            per_organ = {L.ABDO[o]: dsc[o] for o in L.ABDO}
            tail = float(np.nanmean([dsc[o] for o in L.TAIL])); large = float(np.nanmean([dsc[o] for o in L.ABDO if o not in L.TAIL]))
    return tail, large, per_organ, masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--R", type=float, default=8)
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_slices", type=int, default=6000); ap.add_argument("--data", default="data/slices"); ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--out", default="outputs/results/conditional_loupe.json")
    args = ap.parse_args()
    root = args.data if os.path.isdir(args.data) else f"../{args.data}"
    res = {"generic": {}, "conditional": {}, "mask_divergence": {}}
    for seed in args.seeds:
        X, Y, Xt, Yt = load_data(root, args.max_slices, seed)
        if args.smoke:
            X, Y, Xt, Yt = X[:args.smoke], Y[:args.smoke], Xt[:args.smoke], Yt[:args.smoke]; args.epochs = 1
        N = X.shape[-1]
        for cond, key in [(False, "generic"), (True, "conditional")]:
            net, mask = train_arm(cond, X, Y, N, args.R, args.epochs, args.bs, args.lr, seed)
            tail, large, per_organ, masks = eval_arm(cond, net, mask, Xt, Yt, N, args.R, args.bs)
            res[key][str(seed)] = {"tail": round(tail, 4), "large": round(large, 4) if large == large else None, "per_organ": {k: round(v, 4) for k, v in per_organ.items()}}
            effR = [round(N / int(m.sum()), 2) for m in masks.values()]
            print(f"[seed {seed}] {key:11s} tail={tail:.4f}  eff_R={effR[:3]}{'...' if len(effR) > 3 else ''}")
            if cond:  # SANITY: do per-target masks actually differ? (else conditioning is vacuous)
                M = np.stack(list(masks.values())); div = float(np.mean([np.abs(M[a] - M[b]).mean() for a in range(len(M)) for b in range(a + 1, len(M))]))
                res["mask_divergence"][str(seed)] = round(div, 4)
                print(f"           per-target mask mean pairwise diff = {div:.4f} (0 = identical -> conditioning vacuous)")
    # make-or-break
    gt = np.array([res["generic"][str(s)]["tail"] for s in args.seeds])
    ct = np.array([res["conditional"][str(s)]["tail"] for s in args.seeds])
    d = ct - gt
    res["SUMMARY"] = {"generic_tail_mean": round(float(gt.mean()), 4), "conditional_tail_mean": round(float(ct.mean()), 4),
                      "delta": round(float(d.mean()), 4), "wilcoxon_p": float(wilcoxon(ct, gt).pvalue) if len(d) > 1 and (d != 0).any() else None,
                      "conditional_wins_seeds": int((d > 0).sum()), "n_seeds": len(args.seeds),
                      "mean_mask_divergence": round(float(np.mean(list(res["mask_divergence"].values()))), 4) if res["mask_divergence"] else None,
                      "known_generic_loupe_anchor": 0.5452}
    json.dump(res, open(args.out, "w"), indent=2)
    s = res["SUMMARY"]
    print(f"\n=== TC-LOUPE MAKE-OR-BREAK ===")
    print(f"  generic LOUPE tail     : {s['generic_tail_mean']}  (known anchor {s['known_generic_loupe_anchor']})")
    print(f"  conditional (ours) tail: {s['conditional_tail_mean']}")
    print(f"  Δ = {s['delta']:+.4f}  (p={s['wilcoxon_p']}, wins {s['conditional_wins_seeds']}/{s['n_seeds']} seeds)")
    print(f"  per-target mask divergence = {s['mean_mask_divergence']} (must be >0)")
    print("  VERDICT:", "NOVEL METHOD WINS" if s['delta'] > 0.005 and s['conditional_wins_seeds'] >= 2 else "ties/loses -> conditioning does not add (honest negative)")
    print("wrote conditional_loupe.json")


if __name__ == "__main__":
    main()
