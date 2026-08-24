"""FG-Seg MAKE-OR-BREAK control. Does weighting the segmentation loss by the a-priori SPECTRAL-CENTROID LAW beat
weighting by the empirical WORST-ORGAN Dice-drop (the trivial control a reviewer will demand)? Same abdominal 2D proxy
(MRISegmentator), same mixed-R training, same normalization — ONLY the weight SOURCE differs (centroid=law, a-priori vs
drop=empirical). If 'centroid' does not at least TIE 'drop' (ideally beat it on tail organs), the fragility-law novelty
for FG-Seg is dead. Blackout-safe (checkpoint per arm×seed). -> outputs/results/fgseg_control.json"""
import os, json, argparse, numpy as np, torch
import sys; sys.path.insert(0, "scripts")
from b1_joint import UNet2D, dice_ce, remap_labels, REMAP, NCLS, ABDO_IDS, fixed_mask
from loupe import undersample
import labels as L

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_weights(key, rows, name2id):                 # key in {"centroid","drop"} — same 1+3*normalized formula, only the source differs
    w = torch.ones(NCLS, device=dev)
    vals = {r["organ"]: r[key] for r in rows if r["organ"] in name2id}
    lo, hi = min(vals.values()), max(vals.values())
    for nm, v in vals.items():
        w[REMAP[name2id[nm]]] = 1 + 3.0 * (v - lo) / (hi - lo + 1e-9)
    return w


def run_arm(w, Xtr, Ytr, Xte, Yte, epochs, seed, R_eval, train_R, bs=16):
    torch.manual_seed(seed); np.random.seed(seed)
    net = UNet2D(NCLS).to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3); N = Xtr.shape[-1]
    for ep in range(epochs):
        perm = torch.randperm(Xtr.shape[0], device=dev)
        for i in range(0, Xtr.shape[0], bs):
            idx = perm[i:i + bs]; xb = Xtr[idx].unsqueeze(1); yb = Ytr[idx].long()
            m = fixed_mask("vardensity", N, float(train_R), dev)   # FIXED-R training: weighting bites here (mixed-R washes it out per m2_mixedr.json)
            loss = dice_ce(net(undersample(xb, m)), yb, w)
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); inter = torch.zeros(NCLS); den = torch.zeros(NCLS)
    with torch.no_grad():
        m = fixed_mask("vardensity", N, R_eval, dev)
        for i in range(0, Xte.shape[0], bs):
            xb = Xte[i:i + bs].unsqueeze(1).to(dev); yb = Yte[i:i + bs].to(dev)
            pred = net(undersample(xb, m)).argmax(1)
            for c in range(1, NCLS):
                inter[c] += (2 * ((pred == c) & (yb == c)).sum()).item(); den[c] += ((pred == c).sum() + (yb == c).sum()).item()
    return {L.ABDO[ABDO_IDS[c - 1]]: (inter[c] / den[c]).item() if den[c] > 0 else float("nan") for c in range(1, NCLS)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=55); ap.add_argument("--Reval", type=int, default=6); ap.add_argument("--train_R", type=int, default=6)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--max_slices", type=int, default=6000); args = ap.parse_args()
    rows = json.load(open("outputs/results/m0_law_v2.json"))["rows"]
    name2id = {L.ABDO[o]: o for o in ABDO_IDS}
    W = {"uniform": torch.ones(NCLS, device=dev), "drop": build_weights("drop", rows, name2id), "centroid": build_weights("centroid", rows, name2id)}
    tr = np.load("data/slices/train.npz"); X = torch.tensor(tr["images"].astype(np.float32)); Y = torch.tensor(tr["labels"].astype(np.int64))
    te = np.load("data/slices/test.npz"); Xte = torch.tensor(te["images"].astype(np.float32)); Yte = torch.tensor(te["labels"].astype(np.int64))
    if args.max_slices and X.shape[0] > args.max_slices:                    # tail-stratified oversampling (same as b1, fair across arms)
        Yg = Y.to(torch.int16).to(dev); ws = torch.ones(Yg.shape[0], device=dev)
        for oid, b in [(6, 25.), (4, 2.), (12, 2.), (13, 2.), (17, .5)]:
            ws[(Yg == oid).flatten(1).any(1)] += b
        idx = torch.multinomial(ws, args.max_slices, replacement=True).cpu(); del Yg; torch.cuda.empty_cache(); X, Y = X[idx], Y[idx]
    Xtr = X.to(dev); Ytr = remap_labels(Y.to(dev)).to(torch.uint8); Yte = remap_labels(Yte.to(dev)).cpu()
    print(f"{Xtr.shape[0]} train / {Xte.shape[0]} test slices, {NCLS} classes, eval@R{args.Reval}, dev={dev}", flush=True)

    PART = "outputs/results/fgseg_control_partial.json"; res = {a: {} for a in W}; runlist = [(a, s) for a in W for s in args.seeds]
    if os.path.exists(PART):
        d = json.load(open(PART)); res = d["res"]; print(f"RESUMING — have {sum(len(v) for v in res.values())} runs", flush=True)
    for a, s in runlist:
        if str(s) in res[a]: continue
        per = run_arm(W[a], Xtr, Ytr, Xte, Yte, args.epochs, s, args.Reval, args.train_R)
        res[a][str(s)] = per
        tail = np.nanmean([per[L.ABDO[o]] for o in L.TAIL]); large = np.nanmean([per[L.ABDO[o]] for o in L.ABDO if o not in L.TAIL])
        print(f"  {a:9} seed{s}: TAIL {tail:.3f}  LARGE {large:.3f}", flush=True)
        json.dump({"res": res}, open(PART, "w"))
    # aggregate + the make-or-break stats (paired over seed x tail-organ)
    from scipy.stats import wilcoxon
    def tailvec(a): return np.array([res[a][str(s)][L.ABDO[o]] for s in args.seeds for o in L.TAIL])   # (seeds*tail,)
    def meantail(a): return float(np.nanmean(tailvec(a)))
    def wil(a, b):
        x, y = tailvec(a), tailvec(b); ok = ~(np.isnan(x) | np.isnan(y))
        return float(wilcoxon(x[ok], y[ok]).pvalue) if ok.sum() > 1 and (x[ok] != y[ok]).any() else None
    out = {"Reval": args.Reval, "seeds": args.seeds,
           "tail_mean": {a: round(meantail(a), 4) for a in W},
           "large_mean": {a: round(float(np.nanmean([res[a][str(s)][L.ABDO[o]] for s in args.seeds for o in L.ABDO if o not in L.TAIL])), 4) for a in W},
           "MAKE_OR_BREAK_centroid_vs_drop": {"delta_tail": round(meantail("centroid") - meantail("drop"), 4), "p": wil("centroid", "drop")},
           "centroid_vs_uniform": {"delta_tail": round(meantail("centroid") - meantail("uniform"), 4), "p": wil("centroid", "uniform")},
           "drop_vs_uniform": {"delta_tail": round(meantail("drop") - meantail("uniform"), 4), "p": wil("drop", "uniform")},
           "per_organ": {a: {o: round(float(np.nanmean([res[a][str(s)][o] for s in args.seeds])), 4) for o in res[a][str(args.seeds[0])]} for a in W}}
    json.dump(out, open("outputs/results/fgseg_control.json", "w"), indent=2)
    if os.path.exists(PART): os.remove(PART)
    print(f"\n=== FG-Seg make-or-break @R{args.Reval} (abdominal proxy, TAIL Dice, {len(args.seeds)} seeds) ===")
    for a in ["uniform", "drop", "centroid"]: print(f"  {a:9} TAIL {out['tail_mean'][a]:.4f}  LARGE {out['large_mean'][a]:.4f}")
    mb = out["MAKE_OR_BREAK_centroid_vs_drop"]
    print(f"  >>> LAW vs CONTROL:  centroid - drop = {mb['delta_tail']:+.4f} (p={mb['p']})  <- if <=0 & n.s., the fragility-law novelty is DEAD")
    print("wrote fgseg_control.json")


if __name__ == "__main__":
    main()
