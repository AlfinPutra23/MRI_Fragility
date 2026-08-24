"""B1: joint learnable-sampling + segmentation (the MICCAI method).
Pipeline:  clean slice  --[mask]-->  undersampled  -->  2D U-Net seg  -->  per-organ masks
trained end-to-end with a (optionally fragility-weighted) DiceCE loss. The mask is learnable (LOUPE) or fixed.

Variants (args):
  --mask {learned, vardensity, random, equispaced}   --loss {uniform, fragweighted}
  ours = learned + fragweighted ;  key baseline = learned + uniform (LOUPE-uniform) and fixed masks.

  python b1_joint.py --mask learned --loss fragweighted --R 8 --epochs 60 --tag ours
"""
import os, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from loupe import LOUPEMask, undersample, rescale_probs
from kspace import vd_cartesian_mask
import labels as L

ABDO_IDS = list(L.ABDO)                                   # 13 original label ids
REMAP = {o: i + 1 for i, o in enumerate(ABDO_IDS)}        # -> classes 1..13 (0 = background)
NCLS = len(ABDO_IDS) + 1
TAIL_CLS = [REMAP[o] for o in L.TAIL]
# fragility weights per new class (from M0 drops, normalized; same as sweep trainers, scale 3 -> max 4x)
NORM = {1: .156, 2: .112, 3: .117, 4: .844, 5: 0., 6: .609, 7: .263, 11: .302, 12: .933, 13: 1., 16: .525, 17: .682, 18: .804}


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


class ReconNet(nn.Module):
    """Small residual U-Net denoiser: aliased zero-filled image -> cleaned image (1->1 channel).
    Makes B1 a realistic 'learned sampling + learned recon + seg' pipeline (not just zero-filled)."""
    def __init__(s, ch=(24, 48, 96)):
        super().__init__()
        s.d1, s.d2 = DoubleConv(1, ch[0]), DoubleConv(ch[0], ch[1]); s.bott = DoubleConv(ch[1], ch[2]); s.pool = nn.MaxPool2d(2)
        s.u2 = nn.ConvTranspose2d(ch[2], ch[1], 2, 2); s.c2 = DoubleConv(ch[2], ch[1])
        s.u1 = nn.ConvTranspose2d(ch[1], ch[0], 2, 2); s.c1 = DoubleConv(ch[1], ch[0])
        s.out = nn.Conv2d(ch[0], 1, 1)
    def forward(s, x):
        e1 = s.d1(x); e2 = s.d2(s.pool(e1)); b = s.bott(s.pool(e2))
        d = s.c2(torch.cat([s.u2(b), e2], 1)); d = s.c1(torch.cat([s.u1(d), e1], 1))
        return x + s.out(d)                      # residual: recon = aliased + learned correction


def remap_labels(lab):
    out = torch.zeros_like(lab)
    for o, c in REMAP.items():
        out[lab == o] = c
    return out


def dice_ce(logits, target, w):
    # EXACT B1-gate formulation: PER-SAMPLE Dice (do NOT switch to batch-pooled -- it changes the
    # frag-weighting behavior on rare tail organs and is not a fair reproduction of the gate result).
    ce = F.cross_entropy(logits, target, weight=w)
    p = F.softmax(logits, 1); dsc = 0.0
    for c in range(1, NCLS):
        pc = p[:, c]; gc = (target == c).float()
        inter = (pc * gc).sum((1, 2)); den = pc.sum((1, 2)) + gc.sum((1, 2))   # per-sample (B,)
        dsc = dsc + w[c] * (1 - (2 * inter + 1e-5) / (den + 1e-5)).mean()       # mean over batch
    return ce + dsc / (NCLS - 1)


def focal_tversky(logits, target, w, alpha=0.3, beta=0.7, gamma=0.75):
    # MITIGATION loss: Tversky with beta>alpha penalises FALSE NEGATIVES (recall); the focal power (1-TI)^gamma
    # concentrates on hard/vanishing classes -> directly fights the "predict nothing -> 0 Dice" collapse.
    ce = F.cross_entropy(logits, target, weight=w)
    p = F.softmax(logits, 1); loss = 0.0
    for c in range(1, NCLS):
        pc = p[:, c]; gc = (target == c).float()
        tp = (pc * gc).sum((1, 2)); fp = (pc * (1 - gc)).sum((1, 2)); fn = ((1 - pc) * gc).sum((1, 2))
        ti = (tp + 1e-5) / (tp + alpha * fp + beta * fn + 1e-5)
        # clamp base to a small POSITIVE floor (not 0): grad of x**gamma is +inf at x=0 for gamma<1 -> NaN late in
        # training when a class becomes easy (ti->1). Floor 1e-6 keeps the loss value negligible but the grad finite.
        loss = loss + w[c] * ((1 - ti).clamp(min=1e-6) ** gamma).mean()
    return ce + loss / (NCLS - 1)


_LAP = None
def hf_loss(logits, target, w):
    # DOUBLE-JEOPARDY method, half 2 (counter spectral bias): penalise HIGH-FREQUENCY (edge/boundary) mismatch
    # between the predicted foreground probability and the GT mask. A Laplacian high-pass isolates the fine detail
    # the network otherwise learns LAST; matching it forces the boundary structure that acceleration erased.
    global _LAP
    if _LAP is None or _LAP.device != logits.device:
        _LAP = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=logits.device).view(1, 1, 3, 3)
    p = F.softmax(logits, 1); loss = 0.0
    for c in range(1, NCLS):
        hp_p = F.conv2d(p[:, c:c + 1], _LAP, padding=1)
        hp_g = F.conv2d((target == c).float().unsqueeze(1), _LAP, padding=1)
        loss = loss + w[c] * F.mse_loss(hp_p, hp_g)
    return loss / (NCLS - 1)


def fixed_mask(kind, n, R, device):
    if kind == "vardensity":
        m = vd_cartesian_mask(n, R)
    elif kind == "random":
        rng = np.random.default_rng(0); c = n // 2; m = np.zeros(n, bool); m[c-10:c+10] = True
        idx = rng.choice(n, int(n / R) - 20, replace=False); m[idx] = True
    elif kind == "equispaced":
        m = np.zeros(n, bool); m[::int(R)] = True; c = n // 2; m[c-10:c+10] = True
    return torch.tensor(m.astype(np.float32), device=device)


def frag_coverage_target(X, Y, wfrag, max_s=800):
    """FRAGILITY-GUIDED SAMPLING prior (the loop-closer). Which phase-encode lines carry the k-space energy that
    matters DISPROPORTIONATELY for the fragile organs? For each fragile organ c (wfrag[c]>1): energy per PE line of
    |FFT(image restricted to organ c)|^2, weighted by that organ's fragility; then divide by the OVERALL spectrum so
    we emphasize frequencies that are fragile-specific (not just where all energy sits, which LOUPE already gets).
    Returns (N,) summing to 1, aligned with LOUPEMask.probs() (fftshift-centered PE axis = axis -2)."""
    dev = X.device; n = X.shape[-2]
    idx = torch.randperm(X.shape[0], device=dev)[:min(max_s, X.shape[0])]
    S_all = torch.zeros(n, device=dev); S_frag = torch.zeros(n, device=dev)
    for i in range(0, idx.numel(), 64):
        sl = idx[i:i + 64]; xb = X[sl].to(torch.complex64); yb = Y[sl]
        Kall = torch.fft.fftshift(torch.fft.fft2(xb), dim=(-2, -1))
        S_all += (Kall.abs() ** 2).sum(-1).sum(0)                      # energy per PE line (sum over freq-enc + batch)
        for c in range(1, NCLS):
            if wfrag[c] <= 1.0: continue                              # only the weighted (fragile) organs
            Kc = torch.fft.fftshift(torch.fft.fft2((X[sl] * (yb == c)).to(torch.complex64)), dim=(-2, -1))
            S_frag += (wfrag[c] - 1.0) * (Kc.abs() ** 2).sum(-1).sum(0)
    tgt = S_frag / (S_all + 1e-8)                                     # fragile-specific emphasis
    return (tgt / (tgt.sum() + 1e-8)).detach()                       # (N,), sums to 1


def frag_density_mask(X, Y, wfrag, N, R, device, lam=1.0, floor=0.01, max_s=800):
    """R*-GUIDED fixed acquisition (the novel-method attempt). The fragility law holds in RELATIVE terms: fragile
    (low-R*) organs are 2.6-3.7x OVER-represented at mid/high freq (ratio S_frag/S_all) -- but in ABSOLUTE energy
    they are still ~82% central, so an energy-density mask just samples the center (= vardensity, audited). To spend
    the budget on the fragile-SPECIFIC high freq we up-weight by the RATIO S_frag/(S_all + floor*mean(S_all)) --
    FG-LOUPE's idea, but BOUNDED by `floor` so it does not explode at the very highest lines (that unbounded ratio
    was the FG-LOUPE pathology). Same ACS (8%) + total budget 1/R as vardensity -> FAIR comparison. lam blends
    VD (keeps some mid) vs the ratio (adds high); floor tunes how high the emphasis reaches."""
    n = N; c0 = n // 2; acs = max(int(round(n * 0.08)), 4)            # match vardensity ACS exactly
    idx = torch.randperm(X.shape[0], device=device)[:min(max_s, X.shape[0])]
    S_all = torch.zeros(n, device=device); S_frag = torch.zeros(n, device=device)
    for i in range(0, idx.numel(), 64):
        sl = idx[i:i + 64]
        Kall = torch.fft.fftshift(torch.fft.fft2(X[sl].to(torch.complex64)), dim=(-2, -1))
        S_all += (Kall.abs() ** 2).sum(-1).sum(0)                     # overall energy per PE line
        for c in range(1, NCLS):
            if wfrag[c] <= 1.0: continue
            Kc = torch.fft.fftshift(torch.fft.fft2((X[sl] * (Y[sl] == c)).to(torch.complex64)), dim=(-2, -1))
            S_frag += (wfrag[c] - 1.0) * (Kc.abs() ** 2).sum(-1).sum(0)
    ratio = S_frag / (S_all + floor * S_all.mean())                  # BOUNDED fragile-specific emphasis (-> mid/high)
    ratio = ratio / (ratio.sum() + 1e-8)
    d = 1.0 / ((torch.arange(n, device=device).float() - c0).abs() + 1.0); vd = d / d.sum()
    p = (1 - lam) * vd + lam * ratio                                 # R*-guided sampling density (free lines)
    m = torch.zeros(n, dtype=torch.bool, device=device); m[c0 - acs // 2:c0 + (acs - acs // 2)] = True
    need = max(int(round(n / R)), acs) - int(m.sum())
    if need > 0:
        p2 = p.clone(); p2[m] = 0.0; p2 = p2 / (p2.sum() + 1e-8)      # exclude ACS, renormalise
        # stochastic sampling (like vardensity) so the free budget SPREADS across mid/high per the density,
        # instead of top-k piling on the extreme-highest lines (which skips all mid freq). Seeded upstream.
        m[torch.multinomial(p2, min(need, int((p2 > 0).sum())), replacement=False)] = True
    return m.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", choices=["learned", "vardensity", "random", "equispaced", "fragility"], default="learned")
    ap.add_argument("--frag_lam", type=float, default=1.0, help="R*-guided mask: blend of fragility-ratio vs variable-density (0=VD, 1=pure ratio)")
    ap.add_argument("--frag_floor", type=float, default=0.01, help="R*-guided mask: floor on S_all in the ratio (smaller=emphasize higher freq)")
    ap.add_argument("--loss", choices=["uniform", "fragweighted"], default="fragweighted")
    ap.add_argument("--R", type=float, default=8); ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=16); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tag", default="ours"); ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--save_samples", action="store_true")
    ap.add_argument("--recon", action="store_true", help="insert a recon U-Net (learned recon, not zero-filled)")
    ap.add_argument("--frag_recon", action="store_true", help="FRAGILITY-AWARE recon: weight recon loss by per-pixel fragility (vs plain L1 = SOTA)")
    ap.add_argument("--save_net", default="", help="save the trained seg net state_dict here (to reuse as a teacher)")
    ap.add_argument("--teacher", default="", help="DISTILLATION: path to a clean-trained teacher net state_dict")
    ap.add_argument("--distill_w", type=float, default=0.0, help="distillation KL weight (0=off)")
    ap.add_argument("--distill_T", type=float, default=2.0, help="distillation temperature")
    ap.add_argument("--recon_w", type=float, default=1.0, help="weight of the recon L1 loss")
    ap.add_argument("--data", default="data/slices"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_slices", type=int, default=6000, help="cap train slices for speed (0=all)")
    ap.add_argument("--mixed_r", action="store_true", help="MITIGATION: train with a random acceleration R per batch (augmentation)")
    ap.add_argument("--tversky", action="store_true", help="MITIGATION: Focal-Tversky (recall) loss instead of Dice-CE, to fight the 0-Dice collapse")
    ap.add_argument("--hf_w", type=float, default=0.0, help="DJ-Seg: weight of the HF-emphasis (spectral-bias counter) loss; 0=off")
    ap.add_argument("--frag_cov_w", type=float, default=0.0,
                    help=">0 enables FRAGILITY-GUIDED SAMPLING: bias the learned mask toward the k-space lines "
                         "that carry fragile-organ energy (the loop-closer; the sampling-side use of the SA/V law)")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True             # fixed 256x256 input -> autotune fastest convs
    root = args.data if os.path.isdir(args.data) else f"../{args.data}"

    tr = np.load(f"{root}/train.npz"); X = torch.tensor(tr["images"].astype(np.float32)); Y = torch.tensor(tr["labels"].astype(np.int64))
    te = np.load(f"{root}/test.npz");  Xt = torch.tensor(te["images"].astype(np.float32)); Yt = torch.tensor(te["labels"].astype(np.int64))
    if args.smoke:
        X, Y, Xt, Yt = X[:args.smoke], Y[:args.smoke], Xt[:args.smoke], Yt[:args.smoke]; args.epochs = 1
    elif args.max_slices and X.shape[0] > args.max_slices:
        # tail-stratified oversampling (nnU-Net-style foreground oversampling): esophagus (id6) is in only
        # ~0.9% of slices and otherwise starves to 0 Dice; sample WITH REPLACEMENT, up-weighting slices that
        # contain rare tail organs so they are seen often enough to learn. (Same for every variant -> fair.)
        Yg = Y.to(torch.int16).to(dev)                 # transient GPU copy (~2GB) for a fast presence scan
        w_s = torch.ones(Yg.shape[0], device=dev)
        for oid, boost in [(6, 25.), (4, 2.), (12, 2.), (13, 2.), (17, .5)]:   # eso, gallbladder, adrenals, duodenum
            w_s[(Yg == oid).flatten(1).any(1)] += boost
        idx = torch.multinomial(w_s, args.max_slices, replacement=True).cpu()
        del Yg; torch.cuda.empty_cache(); X, Y = X[idx], Y[idx]
    X = X.to(dev); Y = remap_labels(Y.to(dev)).to(torch.uint8)  # train on GPU; labels uint8 -> small footprint
    Yt = remap_labels(Yt.to(dev)).cpu()               # remap test on GPU then move back to CPU
    N = X.shape[-1]                                    # Xt/Yt stay on CPU (single eval pass transfers per-batch)
    n_eso = int((Y == REMAP[6]).flatten(1).any(1).sum())   # confirm the rare organ is now represented
    wfrag = torch.ones(NCLS, device=dev)                            # per-class fragility weights (from M0 drops)
    for o, nd in NORM.items():
        if o in REMAP: wfrag[REMAP[o]] = 1 + 3.0 * nd
    w = wfrag.clone() if args.loss == "fragweighted" else torch.ones(NCLS, device=dev)   # SEG-loss weights
    frag_tgt = frag_coverage_target(X, Y, wfrag) if (args.mask == "learned" and args.frag_cov_w > 0) else None
    print(f"[B1] mask={args.mask} loss={args.loss} frag_cov_w={args.frag_cov_w} R={args.R} | {X.shape[0]} train ({n_eso} w/ esophagus) / {Xt.shape[0]} test slices, {NCLS} classes, dev={dev}")
    if frag_tgt is not None:
        k = max(int(N / args.R), 1)
        print(f"[FG] fragility-guided sampling ON: {100*frag_tgt.topk(k).values.sum().item():.0f}% of the fragility target mass is in its top-{k} lines")

    net = UNet2D(NCLS).to(dev)
    teacher = None
    if args.teacher:                                  # DISTILLATION: a clean-scan teacher guides the accelerated student
        teacher = UNet2D(NCLS).to(dev); teacher.load_state_dict(torch.load(args.teacher, map_location=dev)); teacher.eval()
        for p in teacher.parameters(): p.requires_grad_(False)
        print(f"[distill] clean teacher {args.teacher}, distill_w={args.distill_w} T={args.distill_T}")
    recon_net = ReconNet().to(dev) if args.recon else None
    mask_mod = LOUPEMask(N, args.R).to(dev) if args.mask == "learned" else None
    if args.mask == "learned":
        fmask = None
    elif args.mask == "fragility":                     # R*-guided fixed acquisition (novel-method attempt)
        fmask = frag_density_mask(X, Y, wfrag, N, args.R, dev, args.frag_lam, args.frag_floor)
        print(f"[R*-guided] fragility-ratio mask: lam={args.frag_lam} floor={args.frag_floor}, kept {int(fmask.sum())}/{N} lines (budget {int(N/args.R)})")
    else:
        fmask = fixed_mask(args.mask, N, args.R, dev)
    params = list(net.parameters()) + (list(mask_mod.parameters()) if mask_mod else []) \
        + (list(recon_net.parameters()) if recon_net else [])
    opt = torch.optim.Adam(params, lr=args.lr)

    for ep in range(args.epochs):
        net.train(); perm = torch.randperm(X.shape[0], device=dev); tot = 0.0
        for i in range(0, X.shape[0], args.bs):
            idx = perm[i:i+args.bs]; xb = X[idx].unsqueeze(1); yb = Y[idx].long()
            if args.mixed_r and mask_mod is None:                 # mixed-R augmentation: random acceleration per batch
                m = fixed_mask(args.mask, N, float(np.random.choice([1, 2, 4, 6, 8])), dev)
            else:
                m = mask_mod(training=True) if mask_mod else fmask
            xu = undersample(xb, m)
            rloss = 0.0
            if recon_net is not None:
                xu = recon_net(xu)
                if args.frag_recon:                              # FRAGILITY-AWARE recon (the method): weight the recon
                    wmap = wfrag[yb.long()].unsqueeze(1)          # error by per-pixel fragility -> reconstruct the
                    rloss = (wmap * (xu - xb).abs()).mean()       # fragile organs faithfully (not SSIM-blind)
                else:
                    rloss = F.l1_loss(xu, xb)                     # plain image-quality recon = the SOTA baseline (B2)
            logits = net(xu)
            loss = (focal_tversky if args.tversky else dice_ce)(logits, yb, w) + args.recon_w * rloss
            if args.hf_w > 0:                                     # DJ-Seg: counter spectral bias (learn HF detail)
                loss = loss + args.hf_w * hf_loss(logits, yb, w)
            if teacher is not None and args.distill_w > 0:        # DISTILL clean-teacher -> accelerated-student
                with torch.no_grad(): t_log = teacher(xb)         # teacher sees the CLEAN image (knows the fragile organs)
                T = args.distill_T
                kl = (F.softmax(t_log / T, 1) * (F.log_softmax(t_log / T, 1) - F.log_softmax(logits / T, 1))).sum(1)
                loss = loss + args.distill_w * (T * T) * (wfrag[yb.long()] * kl).mean()  # fragility-weighted KL
            if frag_tgt is not None:                              # fragility-guided sampling: pull the mask's line
                q = mask_mod.probs(); q = q / (q.sum() + 1e-8)     # -distribution toward the fragility target via a
                loss = loss + args.frag_cov_w * -(frag_tgt * (q + 1e-8).log()).sum()  # BOUNDED cross-entropy H(tgt,q)>=0
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"  ep{ep:3d} loss {tot/(X.shape[0]//args.bs+1):.3f}")

    if args.save_net:
        torch.save(net.state_dict(), args.save_net); print(f"[save] net -> {args.save_net}")

    # eval: deploy mask, per-organ Dice on test
    net.eval(); samples = {}
    with torch.no_grad():
        m = mask_mod(training=False) if mask_mod else fmask
        inter = torch.zeros(NCLS); den = torch.zeros(NCLS)
        for i in range(0, Xt.shape[0], args.bs):
            xb = Xt[i:i+args.bs].unsqueeze(1).to(dev); yb = Yt[i:i+args.bs].to(dev)
            xu = undersample(xb, m)
            if recon_net is not None: xu = recon_net(xu)
            pred = net(xu).argmax(1)
            if args.save_samples and not samples:
                samples = dict(clean=xb[:12, 0].cpu().numpy(), under=xu[:12, 0].cpu().numpy(),
                               gt=yb[:12].cpu().numpy(), pred=pred[:12].cpu().numpy())
            for c in range(1, NCLS):
                inter[c] += (2 * ((pred == c) & (yb == c)).sum()).item()
                den[c] += ((pred == c).sum() + (yb == c).sum()).item()
    if args.save_samples:
        np.savez(("outputs" if os.path.isdir("outputs") else "../outputs") + f"/results/b1_samples_{args.tag}.npz",
                 mask=m.detach().cpu().numpy(), **samples)
    dsc = {ABDO_IDS[c-1]: (inter[c] / den[c]).item() if den[c] > 0 else float("nan") for c in range(1, NCLS)}
    tail = np.nanmean([dsc[o] for o in L.TAIL]); large = np.nanmean([dsc[o] for o in L.ABDO if o not in L.TAIL])
    print(f"[B1:{args.tag}] @R{int(args.R)}  TAIL Dice {tail:.3f}  LARGE {large:.3f}  (eff_R={N/int(m.sum()):.1f})")
    os.makedirs("outputs/results" if os.path.isdir("outputs") else "../outputs/results", exist_ok=True)
    import json; json.dump(dict(tag=args.tag, mask=args.mask, loss=args.loss, R=args.R, tail=float(tail),
                                large=float(large), per_organ={L.ABDO[o]: dsc[o] for o in L.ABDO}),
                           open(("outputs" if os.path.isdir("outputs") else "../outputs") + f"/results/b1_{args.tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
