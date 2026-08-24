"""ENHANCE THE REAL-K-SPACE RESULT. mixed-R recovers +0.277 @R8 on real qDESS knee k-space, but the hardest structures
(thin/rare menisci, tibial-lat) can still collapse to ~0 Dice (recall collapse). Combine the two things that individually
work — mixed-R (robustness) + Focal-Tversky (recall loss that fights the 0-Dice collapse) — to rescue them. Compares
mixed-R vs mixed-R+Focal-Tversky on REAL k-space, per-structure. Multi-seed. Run: magicnet. -> outputs/results/condseg_knee_enhance.json"""
import glob, json, os, argparse, numpy as np, torch, torch.nn.functional as F
import sys; sys.path.insert(0, "scripts")
from condseg_knee import gather, undersample_gpu, MASKS, RS
from knee_seg import UNet2D, LAB, NCLS, dev


def focal_tversky(logits, target, alpha=0.3, beta=0.7, gamma=0.75):   # recall-oriented (β>α penalizes FN)
    pr = torch.softmax(logits, 1); loss = 0.
    for c in range(1, NCLS):
        p = pr[:, c]; g = (target == c).float()
        tp = (p * g).sum(); fp = (p * (1 - g)).sum(); fn = ((1 - p) * g).sum()
        ti = (tp + 1e-5) / (tp + alpha * fp + beta * fn + 1e-5)
        loss = loss + ((1 - ti).clamp(min=1e-6) ** gamma)             # clamp>0 avoids inf grad when a class becomes easy
    return loss / (NCLS - 1)


def train_eval(mode, Ctr, Ytr, Cte, Yte, epochs, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    net = UNet2D(NCLS).to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3)
    cw = torch.ones(NCLS, device=dev); cw[1:] = 3.0; bs = 8
    Xc = torch.from_numpy(Ctr); Yt = torch.from_numpy(Ytr.astype(np.int64))
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(Xc))
        for i in range(0, len(Xc), bs):
            b = perm[i:i + bs]; cb = Xc[b].to(dev); yb = Yt[b].to(dev)
            x = undersample_gpu(cb, int(np.random.choice(RS)))[:, None]         # both arms = mixed-R
            lo = net(x); ce = F.cross_entropy(lo, yb, weight=cw); pr = torch.softmax(lo, 1); dl = 0.
            for k in range(1, NCLS):
                pk = pr[:, k]; gk = (yb == k).float(); dl = dl + (1 - (2 * (pk * gk).sum() + 1) / (pk.sum() + gk.sum() + 1))
            loss = ce + dl / (NCLS - 1)
            if mode == "focal":
                loss = loss + focal_tversky(lo, yb)                             # + recall loss
            opt.zero_grad(); loss.backward(); opt.step()
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
    Ctr, Ytr = gather(train); Cte, Yte = gather(test)
    print(f"enhance: {len(Ctr)} train / {len(Cte)} test slices", flush=True)
    res = {m: {R: {LAB[k]: [] for k in LAB} for R in RS} for m in ["mixedr", "focal"]}
    for seed in args.seeds:
        for mode in ["mixedr", "focal"]:
            r = train_eval(mode, Ctr, Ytr, Cte, Yte, args.epochs, seed)
            for R in RS:
                for k in LAB: res[mode][R][LAB[k]].append(r[R][LAB[k]])
            print(f"  seed{seed} [{mode}] R8 mean {np.nanmean([r[8][LAB[k]] for k in LAB]):.4f}", flush=True)
    avg = lambda m, R: float(np.nanmean([np.nanmean(res[m][R][LAB[k]]) for k in LAB]))
    out = {"structure_avg": {m: {f"R{R}": round(avg(m, R), 4) for R in RS} for m in res},
           "per_structure_R8": {LAB[k]: {m: round(float(np.nanmean(res[m][8][LAB[k]])), 3) for m in res} for k in LAB}}
    json.dump(out, open("outputs/results/condseg_knee_enhance.json", "w"), indent=2)
    print("\n=== mixed-R vs mixed-R+Focal (real knee k-space) ===")
    for R in RS: print(f"  R{R}: mixed-R {avg('mixedr',R):.4f} | +Focal {avg('focal',R):.4f}  ({avg('focal',R)-avg('mixedr',R):+.4f})")
    print("\n=== per-structure @R8 ===")
    for k in LAB: print(f"  {LAB[k]:11} mixed-R {np.nanmean(res['mixedr'][8][LAB[k]]):.3f} | +Focal {np.nanmean(res['focal'][8][LAB[k]]):.3f}")
    print("wrote condseg_knee_enhance.json")


if __name__ == "__main__":
    main()
