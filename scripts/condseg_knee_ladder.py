"""FAIR budget-matched ABLATION LADDER on real knee k-space (SKM-TEA) — fills the TBD baselines for the WACV paper:
the strong data-consistent TWO-STAGE (segmenter trained on reconstructions, Morshuis-style), the Focal-Frequency-Loss
NOVELTY CONTROL, FG-TDR, and mixed-R — all same split / seeds / budget, so every arm differs by one controlled factor.
Reports per-case structure-avg AND a pre-registered FRAGILE-structure subset (patellar + both menisci) + Wilcoxon.
Blackout-safe (per-seed checkpoint -> ladder_partial.json). -> outputs/results/ladder.json"""
import glob, os, json, argparse, numpy as np, torch, torch.nn.functional as F
from scipy.stats import wilcoxon
import sys; sys.path.insert(0, "scripts")
from condseg_knee_recon import ReconNet, unrolled, prep, gather, MASKS, RS, zf_mag
from knee_seg import UNet2D, LAB, NCLS, dev

SEGW = None
FRAGILE = [1, 5, 6]   # pre-registered fragile knee structures: patellar cartilage + medial/lateral meniscus (thin, high spectral centroid / highest Dice-drop)


def seg_loss(lo, yb):
    ce = F.cross_entropy(lo, yb, weight=SEGW); pr = torch.softmax(lo, 1); dl = 0.
    for k in range(1, NCLS):
        pk = pr[:, k]; gk = (yb == k).float(); dl = dl + (1 - (2 * (pk * gk).sum() + 1) / (pk.sum() + gk.sum() + 1))
    return ce + dl / (NCLS - 1)


def boundary_mask(sm, k=5):
    er = -F.max_pool2d(-sm[:, None], k, stride=1, padding=k // 2)[:, 0]; return (sm - er).clamp(0, 1)


def compute_W(Xc, Yt, alpha=4.0, boundary=False):
    bs = 8; Pf = None; Pa = None; n = 0
    with torch.no_grad():
        for i in range(0, len(Xc), bs):
            c = prep(Xc[i:i + bs].to(dev)); mag = c.abs(); sm = (Yt[i:i + bs].to(dev) > 0).float()
            if boundary: sm = boundary_mask(sm)
            Ff = torch.fft.fftshift(torch.fft.fft2(mag * sm), dim=(-2, -1)); Fa = torch.fft.fftshift(torch.fft.fft2(mag), dim=(-2, -1))
            pf = (Ff.abs() ** 2).sum(0); pa = (Fa.abs() ** 2).sum(0)
            Pf = pf if Pf is None else Pf + pf; Pa = pa if Pa is None else Pa + pa; n += len(c)
    W = 1 + alpha * (Pf / n) / ((Pa / n) + 1e-6); return (W / W.mean()).detach()


def freq_loss(xhat, xclean, W):
    Fh = torch.fft.fftshift(torch.fft.fft2(xhat), dim=(-2, -1)); Fc = torch.fft.fftshift(torch.fft.fft2(xclean), dim=(-2, -1))
    return ((W.sqrt()[None] * (Fh - Fc).abs()) ** 2).mean()


def ffl_loss(xhat, xclean):                       # Focal Frequency Loss (generic, error-adaptive, anatomy-agnostic) — the novelty control
    Fh = torch.fft.fft2(xhat); Fc = torch.fft.fft2(xclean); d = (Fh - Fc).abs()
    w = (d / (d.amax(dim=(-2, -1), keepdim=True) + 1e-8)).detach()      # per-image focal weight in [0,1]
    return (w * d ** 2).mean()


def train_seg(mode, Xc, Yt, epochs, seed, recon=None):
    torch.manual_seed(seed); np.random.seed(seed); net = UNet2D(NCLS).to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3); bs = 8
    for ep in range(epochs):
        perm = torch.randperm(len(Xc))
        for i in range(0, len(Xc), bs):
            b = perm[i:i + bs]; cb = prep(Xc[b].to(dev)); yb = Yt[b].to(dev); R = int(np.random.choice(RS))
            if mode == "clean": x = cb.abs().clamp(0, 2)[:, None]
            elif mode == "mixedr": x = zf_mag(cb, R)[:, None]
            else:                                                       # "recon": two-stage v2 — segment the RECONSTRUCTIONS
                with torch.no_grad(): x = unrolled(recon, cb, R)[:, None]
            loss = seg_loss(net(x), yb); opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); [p.requires_grad_(False) for p in net.parameters()]; return net


def train_recon(mode, Xc, Yt, segnet, W, epochs, seed, lam=0.15):
    torch.manual_seed(seed + 7); np.random.seed(seed + 7); net = ReconNet().to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3); bs = 8
    for ep in range(epochs):
        perm = torch.randperm(len(Xc))
        for i in range(0, len(Xc), bs):
            b = perm[i:i + bs]; cb = prep(Xc[b].to(dev)); yb = Yt[b].to(dev); R = int(np.random.choice(RS)); xr = unrolled(net, cb, R)
            if mode == "img": loss = F.l1_loss(xr, cb.abs().clamp(0, 2))
            else:
                loss = seg_loss(segnet(xr[:, None]), yb)
                if mode == "fgtdr": loss = loss + lam * freq_loss(xr, cb.abs().clamp(0, 2), W)
                elif mode == "ffl": loss = loss + lam * ffl_loss(xr, cb.abs().clamp(0, 2))
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); return net


def percase(predict, Cte, Yte, cid, R):
    """-> list over test cases of {struct_label: dice}."""
    bs = 8; nc = cid.max() + 1
    inter = {c: {k: 0 for k in LAB} for c in range(nc)}; union = {c: {k: 0 for k in LAB} for c in range(nc)}
    Xe = torch.from_numpy(Cte)
    with torch.no_grad():
        for i in range(0, len(Xe), bs):
            cb = prep(Xe[i:i + bs].to(dev)); g = Yte[i:i + bs]; cc = cid[i:i + bs]; pr = predict(cb, R)
            for j in range(len(g)):
                for k in LAB:
                    inter[cc[j]][k] += int(np.logical_and(pr[j] == k, g[j] == k).sum())
                    union[cc[j]][k] += int((pr[j] == k).sum() + (g[j] == k).sum())
    per = []
    for c in range(nc):
        if any(union[c][k] > 0 for k in LAB):
            per.append({k: (2 * inter[c][k] / union[c][k]) for k in LAB if union[c][k] > 0})
    return per


def avg(per, subset=None):
    ks = subset if subset else list(LAB)
    return np.array([np.mean([d[k] for k in ks if k in d]) for d in per if any(k in d for k in ks)])


def main():
    global SEGW
    ap = argparse.ArgumentParser(); ap.add_argument("--seg_epochs", type=int, default=35); ap.add_argument("--rec_epochs", type=int, default=28)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1]); ap.add_argument("--Reval", type=int, default=8); args = ap.parse_args()
    SEGW = torch.ones(NCLS, device=dev); SEGW[1:] = 3.0
    h5s = sorted(glob.glob("data/skmtea/kspace/**/*.h5", recursive=True))
    cases = [(h, f"data/skmtea/seg/{os.path.basename(h)[:-3]}_raw-data-track.nii.gz") for h in h5s]
    cases = [(h, s) for h, s in cases if os.path.exists(s)]; idx = np.random.RandomState(0).permutation(len(cases))
    test = [cases[i] for i in idx[:14]]; train = [cases[i] for i in idx[14:44]]
    Ctr, Ytr, _ = gather(train); Cte, Yte, cid = gather(test); Xc = torch.from_numpy(Ctr); Yt = torch.from_numpy(Ytr.astype(np.int64))
    print(f"{len(Xc)} train / {len(Cte)} test, {cid.max()+1} cases, eval@R{args.Reval}", flush=True)
    ARMS = ["recon_only", "two_stage", "mixedR", "task", "ffl", "fgtdr", "fgtdr_mixedr"]
    res = {a: [] for a in ARMS}; resf = {a: [] for a in ARMS}          # structure-avg  and  fragile-subset
    PART = "outputs/results/ladder_partial.json"; done = []
    if os.path.exists(PART):
        _d = json.load(open(PART)); res = _d["res"]; resf = _d["resf"]; done = list(_d["done"])
        print(f"RESUMING — seeds done: {done}", flush=True)
    for seed in args.seeds:
        if seed in done: print(f"  skip seed{seed}", flush=True); continue
        Sclean = train_seg("clean", Xc, Yt, args.seg_epochs, seed)
        Smixr = train_seg("mixedr", Xc, Yt, args.seg_epochs, seed)
        W = compute_W(Xc, Yt)
        Rimg = train_recon("img", Xc, Yt, Sclean, W, args.rec_epochs, seed)
        Srecon = train_seg("recon", Xc, Yt, args.seg_epochs, seed, recon=Rimg)     # two-stage v2 segmenter (trained on reconstructions)
        Rtask = train_recon("task", Xc, Yt, Sclean, W, args.rec_epochs, seed)
        Rffl = train_recon("ffl", Xc, Yt, Sclean, W, args.rec_epochs, seed)
        Rfg = train_recon("fgtdr", Xc, Yt, Sclean, W, args.rec_epochs, seed)
        R = args.Reval
        preds = {
            "recon_only":   lambda c, R: Sclean(unrolled(Rimg, c, R)[:, None]).argmax(1).cpu().numpy(),
            "two_stage":    lambda c, R: Srecon(unrolled(Rimg, c, R)[:, None]).argmax(1).cpu().numpy(),
            "mixedR":       lambda c, R: Smixr(zf_mag(c, R)[:, None]).argmax(1).cpu().numpy(),
            "task":         lambda c, R: Sclean(unrolled(Rtask, c, R)[:, None]).argmax(1).cpu().numpy(),
            "ffl":          lambda c, R: Sclean(unrolled(Rffl, c, R)[:, None]).argmax(1).cpu().numpy(),
            "fgtdr":        lambda c, R: Sclean(unrolled(Rfg, c, R)[:, None]).argmax(1).cpu().numpy(),
            "fgtdr_mixedr": lambda c, R: Smixr(unrolled(Rfg, c, R)[:, None]).argmax(1).cpu().numpy(),   # complementarity: fragility recon -> mixed-R segmenter
        }
        for a in ARMS:
            per = percase(preds[a], Cte, Yte, cid, R)
            res[a] += list(avg(per)); resf[a] += list(avg(per, FRAGILE))
        done.append(seed); json.dump({"res": res, "resf": resf, "done": done}, open(PART, "w"))
        print(f"  seed{seed} (struct-avg): " + " | ".join(f"{a} {np.mean(res[a][-14:]):.3f}" for a in ARMS), flush=True)
    A = {a: np.array(res[a]) for a in ARMS}; Af = {a: np.array(resf[a]) for a in ARMS}
    def wil(a, b): n = min(len(a), len(b)); return float(wilcoxon(a[:n], b[:n]).pvalue) if n > 1 and (a[:n] != b[:n]).any() else None
    def cmp(D, x, y): return {"delta": round(float((D[x] - D[y][:len(D[x])]).mean()), 4), "p": wil(D[x], D[y])}
    out = {"n_percase": len(A["fgtdr"]), "Reval": args.Reval,
           "mean_structavg": {a: round(float(A[a].mean()), 4) for a in ARMS},
           "mean_fragile": {a: round(float(Af[a].mean()), 4) for a in ARMS},
           "structavg": {"fgtdr_vs_mixedR": cmp(A, "fgtdr", "mixedR"), "fgtdr_vs_two_stage": cmp(A, "fgtdr", "two_stage"),
                          "fgtdr_vs_ffl": cmp(A, "fgtdr", "ffl"), "two_stage_vs_mixedR": cmp(A, "two_stage", "mixedR"),
                          "fgtdr_mixedr_vs_mixedR": cmp(A, "fgtdr_mixedr", "mixedR")},
           "fragile": {"fgtdr_vs_mixedR": cmp(Af, "fgtdr", "mixedR"), "fgtdr_vs_two_stage": cmp(Af, "fgtdr", "two_stage"),
                       "fgtdr_vs_ffl": cmp(Af, "fgtdr", "ffl"), "fgtdr_mixedr_vs_mixedR": cmp(Af, "fgtdr_mixedr", "mixedR")}}
    json.dump(out, open("outputs/results/ladder.json", "w"), indent=2)
    if os.path.exists(PART): os.remove(PART)
    print(f"\n=== LADDER @R{args.Reval} (real knee, n={len(A['fgtdr'])} cases) ===")
    print("  arm            struct-avg   fragile(patellar+menisci)")
    for a in ARMS: print(f"  {a:14} {A[a].mean():.4f}      {Af[a].mean():.4f}")
    print(f"  [novelty]  FG-TDR vs FFL      struct {out['structavg']['fgtdr_vs_ffl']['delta']:+.4f} (p={out['structavg']['fgtdr_vs_ffl']['p']}) | fragile {out['fragile']['fgtdr_vs_ffl']['delta']:+.4f} (p={out['fragile']['fgtdr_vs_ffl']['p']})")
    print(f"  [the win?] FG-TDR vs mixed-R  struct {out['structavg']['fgtdr_vs_mixedR']['delta']:+.4f} | FRAGILE {out['fragile']['fgtdr_vs_mixedR']['delta']:+.4f} (p={out['fragile']['fgtdr_vs_mixedR']['p']})")
    print(f"  [baseline] two-stage vs mixed-R struct {out['structavg']['two_stage_vs_mixedR']['delta']:+.4f}")
    print("wrote ladder.json")


if __name__ == "__main__":
    main()
