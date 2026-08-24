"""FG-TDR: Fragility-Guided Task-Driven Reconstruction — THE method-paper experiment (Adler-Öktem task-adapted recon +
our novel fragility-frequency prior). 4 arms on REAL qDESS knee k-space, per-structure Dice + per-case Wilcoxon:
  1 recon-then-segment  : image-L1 unrolled recon  + frozen clean segmenter   (2508.18975's winner = strong baseline)
  2 mixed-R             : train segmenter on the degradation, no recon         (strong non-recon baseline)
  3 task-adapted recon  : unrolled recon trained END-TO-END for Dice (λ_freq=0) (Adler-Öktem, no fragility)
  4 FG-TDR (OURS)       : task-adapted + fragility-frequency weighting √W(k)     (the novel method)
Novel claim: arm4 > arm3 (fragility prior helps) AND arm4 ≥ arm1,arm2 (beats baselines). Run: magicnet. -> fgtdr.json"""
import glob, os, json, argparse, numpy as np, torch, torch.nn.functional as F
from scipy.stats import wilcoxon
import sys; sys.path.insert(0, "scripts")
from condseg_knee_recon import ReconNet, unrolled, prep, gather, MASKS, RS, zf_mag
from knee_seg import UNet2D, LAB, NCLS, dev

SEGW = None


def seg_loss(lo, yb):
    ce = F.cross_entropy(lo, yb, weight=SEGW); pr = torch.softmax(lo, 1); dl = 0.
    for k in range(1, NCLS):
        pk = pr[:, k]; gk = (yb == k).float(); dl = dl + (1 - (2 * (pk * gk).sum() + 1) / (pk.sum() + gk.sum() + 1))
    return ce + dl / (NCLS - 1)


def boundary_mask(sm, k=5):                             # structure EDGES via erosion = -maxpool(-x); high-freq by construction
    er = -F.max_pool2d(-sm[:, None], k, stride=1, padding=k // 2)[:, 0]
    return (sm - er).clamp(0, 1)


def compute_W(Xc, Yt, alpha=4.0, boundary=False):       # fragility spectrum: up-weight frequencies where structures concentrate
    bs = 8; Pf = None; Pa = None; n = 0
    with torch.no_grad():
        for i in range(0, len(Xc), bs):
            c = prep(Xc[i:i + bs].to(dev)); mag = c.abs(); sm = (Yt[i:i + bs].to(dev) > 0).float()
            if boundary: sm = boundary_mask(sm)         # AUDIT FIX: weight structure EDGES (high-freq) not smooth interiors
            Ff = torch.fft.fftshift(torch.fft.fft2(mag * sm), dim=(-2, -1)); Fa = torch.fft.fftshift(torch.fft.fft2(mag), dim=(-2, -1))
            pf = (Ff.abs() ** 2).sum(0); pa = (Fa.abs() ** 2).sum(0)
            Pf = pf if Pf is None else Pf + pf; Pa = pa if Pa is None else Pa + pa; n += len(c)
    W = 1 + alpha * (Pf / n) / ((Pa / n) + 1e-6)
    return (W / W.mean()).detach()                      # normalized (H,W) on GPU


def freq_loss(xhat, xclean, W):
    Fh = torch.fft.fftshift(torch.fft.fft2(xhat), dim=(-2, -1)); Fc = torch.fft.fftshift(torch.fft.fft2(xclean), dim=(-2, -1))
    return ((W.sqrt()[None] * (Fh - Fc).abs()) ** 2).mean()


def train_seg(mode, Xc, Yt, epochs, seed):
    torch.manual_seed(seed); np.random.seed(seed); net = UNet2D(NCLS).to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3); bs = 8
    for ep in range(epochs):
        perm = torch.randperm(len(Xc))
        for i in range(0, len(Xc), bs):
            b = perm[i:i + bs]; cb = prep(Xc[b].to(dev)); yb = Yt[b].to(dev)
            x = cb.abs().clamp(0, 2)[:, None] if mode == "clean" else zf_mag(cb, int(np.random.choice(RS)))[:, None]
            loss = seg_loss(net(x), yb); opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); [p.requires_grad_(False) for p in net.parameters()]; return net


def train_recon(mode, Xc, Yt, segnet, W, epochs, seed, lam_freq=0.15):
    torch.manual_seed(seed + 7); np.random.seed(seed + 7); net = ReconNet().to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3); bs = 8
    for ep in range(epochs):
        perm = torch.randperm(len(Xc))
        for i in range(0, len(Xc), bs):
            b = perm[i:i + bs]; cb = prep(Xc[b].to(dev)); yb = Yt[b].to(dev); R = int(np.random.choice(RS))
            xr = unrolled(net, cb, R)                                       # magnitude recon (differentiable)
            if mode == "img":
                loss = F.l1_loss(xr, cb.abs().clamp(0, 2))                   # image-quality recon
            else:
                loss = seg_loss(segnet(xr[:, None]), yb)                     # task-adapted: recon FOR segmentation
                if mode == "fgtdr":
                    loss = loss + lam_freq * freq_loss(xr, cb.abs().clamp(0, 2), W)   # + fragility-frequency prior
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); return net


def percase(predict, Cte, Yte, cid, R):
    bs = 8; nc = cid.max() + 1; inter = {c: {k: 0 for k in LAB} for c in range(nc)}; union = {c: {k: 0 for k in LAB} for c in range(nc)}
    Xe = torch.from_numpy(Cte)
    with torch.no_grad():
        for i in range(0, len(Xe), bs):
            cb = prep(Xe[i:i + bs].to(dev)); g = Yte[i:i + bs]; cc = cid[i:i + bs]
            pr = predict(cb, R)
            for j in range(len(g)):
                for k in LAB: inter[cc[j]][k] += int(np.logical_and(pr[j] == k, g[j] == k).sum()); union[cc[j]][k] += int((pr[j] == k).sum() + (g[j] == k).sum())
    return [float(np.mean([2 * inter[c][k] / union[c][k] for k in LAB if union[c][k] > 0])) for c in range(nc) if any(union[c][k] > 0 for k in LAB)]


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
    res = {a: [] for a in ["mixedR", "task_adapted", "FGTDR", "FGTDR_bnd"]}
    PARTIAL = "outputs/results/fgtdr_bnd_partial.json"; seeds_done = []
    if os.path.exists(PARTIAL):                                          # blackout-safe resume: skip already-finished seeds
        _d = json.load(open(PARTIAL)); res = _d["res"]; seeds_done = list(_d["seeds_done"])
        print(f"RESUMING — seeds already checkpointed: {seeds_done}", flush=True)
    for seed in args.seeds:
        if seed in seeds_done:
            print(f"  skip seed{seed} (checkpointed)", flush=True); continue
        Sclean = train_seg("clean", Xc, Yt, args.seg_epochs, seed)
        Smixr = train_seg("mixedr", Xc, Yt, args.seg_epochs, seed)
        W = compute_W(Xc, Yt); Wb = compute_W(Xc, Yt, boundary=True)          # region prior  vs  boundary (high-freq) prior
        Rtask = train_recon("task", Xc, Yt, Sclean, W, args.rec_epochs, seed)
        Rfg = train_recon("fgtdr", Xc, Yt, Sclean, W, args.rec_epochs, seed)   # FG-TDR region-W (reproduces the earlier result)
        Rfgb = train_recon("fgtdr", Xc, Yt, Sclean, Wb, args.rec_epochs, seed) # FG-TDR boundary-W (the audit fix); same init -> isolates W
        R = args.Reval
        res["mixedR"] += percase(lambda c, R: Smixr(zf_mag(c, R)[:, None]).argmax(1).cpu().numpy(), Cte, Yte, cid, R)
        res["task_adapted"] += percase(lambda c, R: Sclean(unrolled(Rtask, c, R)[:, None]).argmax(1).cpu().numpy(), Cte, Yte, cid, R)
        res["FGTDR"] += percase(lambda c, R: Sclean(unrolled(Rfg, c, R)[:, None]).argmax(1).cpu().numpy(), Cte, Yte, cid, R)
        res["FGTDR_bnd"] += percase(lambda c, R: Sclean(unrolled(Rfgb, c, R)[:, None]).argmax(1).cpu().numpy(), Cte, Yte, cid, R)
        print(f"  seed{seed}: mixedR {np.mean(res['mixedR'][-14:]):.3f} | task {np.mean(res['task_adapted'][-14:]):.3f} | FG-TDR-region {np.mean(res['FGTDR'][-14:]):.3f} | FG-TDR-bnd {np.mean(res['FGTDR_bnd'][-14:]):.3f}", flush=True)
        seeds_done.append(seed); json.dump({"res": res, "seeds_done": seeds_done}, open(PARTIAL, "w"))   # checkpoint after each seed
    A = {a: np.array(v) for a, v in res.items()}
    def wil(a, b): n = min(len(a), len(b)); return float(wilcoxon(a[:n], b[:n]).pvalue) if n > 1 and (a[:n] != b[:n]).any() else None
    def cmp(x, y): return {"delta": round(float((A[x] - A[y][:len(A[x])]).mean()), 4), "p": wil(A[x], A[y])}
    out = {"n_percase": len(A["FGTDR_bnd"]), "Reval": args.Reval,
           "mean": {a: round(float(A[a].mean()), 4) for a in A}, "std": {a: round(float(A[a].std()), 4) for a in A},
           "FGTDRbnd_vs_FGTDR": cmp("FGTDR_bnd", "FGTDR"), "FGTDRbnd_vs_task_adapted": cmp("FGTDR_bnd", "task_adapted"),
           "FGTDRbnd_vs_mixedR": cmp("FGTDR_bnd", "mixedR"), "FGTDR_vs_task_adapted": cmp("FGTDR", "task_adapted")}
    json.dump(out, open("outputs/results/fgtdr_bnd.json", "w"), indent=2)
    if os.path.exists(PARTIAL): os.remove(PARTIAL)                       # done -> clear the resume checkpoint
    print(f"\n=== FG-TDR boundary-W @R{args.Reval} (real knee k-space, n={len(A['FGTDR_bnd'])} cases) ===")
    for a in ["mixedR", "task_adapted", "FGTDR", "FGTDR_bnd"]: print(f"  {a:16} {A[a].mean():.4f} ± {A[a].std():.4f}")
    print(f"  FG-TDR-bnd vs FG-TDR-region : {out['FGTDRbnd_vs_FGTDR']['delta']:+.4f} (p={out['FGTDRbnd_vs_FGTDR']['p']})  <- does BOUNDARY beat REGION?")
    print(f"  FG-TDR-bnd vs task-adapted  : {out['FGTDRbnd_vs_task_adapted']['delta']:+.4f} (p={out['FGTDRbnd_vs_task_adapted']['p']})")
    print(f"  FG-TDR-bnd vs mixed-R       : {out['FGTDRbnd_vs_mixedR']['delta']:+.4f} (p={out['FGTDRbnd_vs_mixedR']['p']})  <- does it finally beat mixed-R?")
    print("wrote fgtdr_bnd.json")


if __name__ == "__main__":
    main()
