"""EQUAL-ENERGY INTERVENTION v3 — the width-matched design. LAST attempt at a causal test; if this is null, the
frequency claim is reported as predictive-only and we stop.

Why v1 and v2 failed (both are quarantined, not deleted):
  v1  budget = min(E_own, E_control) with control = FARTHEST annulus. That outer ring holds ~0.03% of image energy, so
      the shared budget was ~0.03% and only ~0.5% of the centroid band was removed -> every ablation a no-op (all
      Dice changes exactly 0.0000).
  v2  budget raised to 1.49%, matched by ATTENUATING each band. But matched energy != matched surgery: the
      energy-poor band was fully deleted (beta=0) while the energy-rich band was barely touched (beta=0.86). The
      significant result therefore reflected HOW COMPLETELY a band was removed, not WHERE it sat.

v3 fixes the confound: EVERY ablation is a FULL deletion (beta=0), and the energy is matched by varying the band
WIDTH instead — a thin ring near DC, a wide ring further out, both removing exactly E_target. Same surgery, same
energy, only LOCATION differs. That is the only contrast that isolates "where".

Test: for each organ, delete at its OWN spectral-centroid radius vs at OTHER organs' centroid radii. Because it is
within-organ, baseline difficulty / size / shape are constant by construction.

  python scripts/intervention_v3.py --E_pct 1.5
-> outputs/results/intervention_v3.json
"""
import json, argparse, numpy as np, torch, sys
sys.path.insert(0, "scripts")
from b1_joint import UNet2D, dice_ce, remap_labels, NCLS, ABDO_IDS
import labels as L
from scipy.stats import wilcoxon

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def radial(n, device):
    c = n // 2; yy, xx = np.mgrid[:n, :n]
    return torch.tensor((np.sqrt((yy - c) ** 2 + (xx - c) ** 2) / c).astype(np.float32), device=device)


def ring_mask(rho, lo, hi):                                  # bool ring lo<=r<hi
    return (rho >= lo) & (rho < hi)


def delete(x, keep):                                         # FULL deletion of the ring (keep is a 0/1 float mask)
    K = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1)) * keep[None]
    return torch.fft.ifft2(torch.fft.ifftshift(K, dim=(-2, -1)), dim=(-2, -1)).abs()


def per_organ_dice(net, X, Y, keep, bs=16):
    inter = torch.zeros(NCLS); den = torch.zeros(NCLS)
    with torch.no_grad():
        for i in range(0, X.shape[0], bs):
            xb, yb = X[i:i + bs], Y[i:i + bs]
            pred = net((delete(xb, keep) if keep is not None else xb)[:, None]).argmax(1)
            for c in range(1, NCLS):
                inter[c] += (2 * ((pred == c) & (yb == c)).sum()).item(); den[c] += ((pred == c).sum() + (yb == c).sum()).item()
    return {L.ABDO[ABDO_IDS[c - 1]]: (inter[c] / den[c]).item() if den[c] > 0 else float("nan") for c in range(1, NCLS)}


def solve_halfwidth(P, rho, r0, E_t, hi=0.5):
    """smallest h with energy(ring [r0-h, r0+h]) == E_t  (monotone in h -> bisection)."""
    def eng(h):
        m = ring_mask(rho, max(0.0, r0 - h), min(1.5, r0 + h))
        return float(P[:, m].sum())
    if eng(hi) < E_t: return None                            # cannot reach the budget at this radius
    lo = 0.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if eng(mid) < E_t: lo = mid
        else: hi = mid
    return (lo + hi) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=45); ap.add_argument("--E_pct", type=float, default=1.5)
    ap.add_argument("--max_slices", type=int, default=6000); ap.add_argument("--min_sep", type=float, default=0.015)
    args = ap.parse_args()

    tr = np.load("data/slices/train.npz"); X = torch.tensor(tr["images"].astype(np.float32)); Y = torch.tensor(tr["labels"].astype(np.int64))
    te = np.load("data/slices/test.npz"); Xt = torch.tensor(te["images"].astype(np.float32)).to(dev); Yt = torch.tensor(te["labels"].astype(np.int64))
    if args.max_slices and X.shape[0] > args.max_slices:
        Yg = Y.to(torch.int16).to(dev); ws = torch.ones(Yg.shape[0], device=dev)
        for oid, b in [(6, 25.), (4, 2.), (12, 2.), (13, 2.), (17, .5)]: ws[(Yg == oid).flatten(1).any(1)] += b
        idx = torch.multinomial(ws, args.max_slices, replacement=True).cpu(); del Yg; torch.cuda.empty_cache(); X, Y = X[idx], Y[idx]
    X = X.to(dev); Y = remap_labels(Y.to(dev)).to(torch.uint8); Yt = remap_labels(Yt.to(dev)).to(torch.uint8)
    N = X.shape[-1]
    torch.manual_seed(0); net = UNet2D(NCLS).to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3)
    w = torch.ones(NCLS, device=dev); w[1:] = 3.0; bs = 16
    for _ in range(args.epochs):
        perm = torch.randperm(X.shape[0], device=dev)
        for i in range(0, X.shape[0], bs):
            b = perm[i:i + bs]; loss = dice_ce(net(X[b].unsqueeze(1)), Y[b].long(), w)
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()

    rho = radial(N, dev)
    with torch.no_grad():
        P = (torch.fft.fftshift(torch.fft.fft2(Xt[:200]), dim=(-2, -1)).abs() ** 2)
    E_t = float(P.sum()) * args.E_pct / 100.0
    law = {r["organ"]: r["centroid"] for r in json.load(open("outputs/results/m0_law_v2.json"))["rows"]}

    # distinct target radii = organs' centroids (merged when closer than min_sep)
    targets = []
    for r in sorted(set(round(v, 4) for v in law.values())):
        if not targets or r - targets[-1] >= args.min_sep: targets.append(r)
    keeps, widths = {}, {}
    for r0 in targets:
        h = solve_halfwidth(P, rho, r0, E_t)
        if h is None: print(f"  skip r={r0}: cannot reach budget", flush=True); continue
        m = ring_mask(rho, max(0.0, r0 - h), r0 + h)
        keeps[r0] = (~m).float(); widths[r0] = h
        print(f"  target r={r0:.3f}: full-delete ring +-{h:.4f} ({100*float(P[:, m].sum())/float(P.sum()):.3f}% of energy)", flush=True)
    if len(keeps) < 2: raise SystemExit("need >=2 target radii")

    clean = per_organ_dice(net, Xt, Yt, None)
    dice_at = {r0: per_organ_dice(net, Xt, Yt, keeps[r0]) for r0 in keeps}
    res = {"E_pct": args.E_pct, "targets": {str(r): round(widths[r], 4) for r in keeps},
           "clean_dice": {o: round(clean[L.ABDO[o]], 3) for o in L.ABDO}, "organs": {}}
    dall, dfrag = [], []
    for oid in ABDO_IDS:
        nm = L.ABDO[oid]; c = law.get(nm)
        if c is None or np.isnan(clean[nm]): continue
        own = min(keeps, key=lambda r: abs(r - c))                       # nearest target to this organ's centroid
        others = [r for r in keeps if r != own]
        d_own = clean[nm] - dice_at[own][nm]
        d_oth = float(np.mean([clean[nm] - dice_at[r][nm] for r in others]))
        res["organs"][nm] = {"centroid": round(c, 3), "own_target": own, "drop_own": round(d_own, 4),
                             "drop_others_mean": round(d_oth, 4), "own_minus_other": round(d_own - d_oth, 4),
                             "by_target": {str(r): round(clean[nm] - dice_at[r][nm], 4) for r in keeps},
                             "fragile": oid in L.TAIL}
        dall.append(d_own - d_oth)
        if oid in L.TAIL: dfrag.append(d_own - d_oth)

    a, af = np.array(dall), np.array(dfrag)
    allc = [v for o in res["organs"].values() for v in o["by_target"].values()]
    mx = float(max(abs(x) for x in allc)) if allc else 0.0
    def wil(x): return float(wilcoxon(x, np.zeros_like(x)).pvalue) if len(x) > 1 and (x != 0).any() else None
    res["SUMMARY"] = {"n_organs": len(a), "mean_own_minus_other": round(float(a.mean()), 4), "p_wilcoxon": wil(a),
                      "n_own_worse": int((a > 0).sum()),
                      "fragile_mean": round(float(af.mean()), 4) if len(af) else None, "fragile_p": wil(af),
                      "max_abs_dice_change": round(mx, 4), "ablation_is_a_noop": bool(mx < 0.005)}
    json.dump(res, open("outputs/results/intervention_v3.json", "w"), indent=2)
    s = res["SUMMARY"]
    print(f"\n=== INTERVENTION v3 (width-matched, FULL deletion, {args.E_pct}% energy everywhere) ===")
    print(f"  sanity: max |dDice| = {s['max_abs_dice_change']} | no-op = {s['ablation_is_a_noop']}")
    if s["ablation_is_a_noop"]:
        print("  !! INVALID: no-op, raise --E_pct and re-run.")
    else:
        print(f"  ALL organs   : own-radius minus other-radii dDice = {s['mean_own_minus_other']:+.4f} "
              f"(p={s['p_wilcoxon']}), {s['n_own_worse']}/{s['n_organs']} hurt most at their OWN radius")
        print(f"  FRAGILE      : {s['fragile_mean']} (p={s['fragile_p']})")
        print("  -> CAUSAL support only if >0 AND significant. Otherwise: report predictive-only and STOP.")


if __name__ == "__main__":
    main()
