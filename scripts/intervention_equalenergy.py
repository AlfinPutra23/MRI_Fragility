"""EQUAL-ENERGY CONTROL-BAND INTERVENTION — the causal linchpin of the fragility discovery. For each organ, zero a
k-space annulus at the organ's OWN spectral-centroid radius vs a CONTROL annulus of MATCHED total energy at a different
radius; re-segment with a fixed clean segmenter; the organ's Dice must drop MORE for its centroid band than the
equal-energy control. Because total removed energy is held fixed and it is WITHIN-ORGAN, a positive result is causal,
non-tautological, and immune to the size/shape/contrast confound. Abdominal proxy. -> outputs/results/intervention.json"""
import os, json, argparse, numpy as np, torch
import sys; sys.path.insert(0, "scripts")
from b1_joint import UNet2D, dice_ce, remap_labels, REMAP, NCLS, ABDO_IDS
import labels as L
from scipy.stats import wilcoxon

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def annulus(n, r_lo, r_hi):                                   # 2D fftshift-centered radial band (bool)
    c = n // 2; yy, xx = np.mgrid[:n, :n]; rho = np.sqrt((yy - c) ** 2 + (xx - c) ** 2) / c
    return (rho >= r_lo) & (rho < r_hi)


def ablate(x, band_t, beta=0.0):                             # scale k-space coeffs INSIDE the band by beta (0 = full removal)
    K = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
    m = torch.ones_like(band_t, dtype=torch.float32); m[band_t] = beta   # attenuation lets us match removed energy EXACTLY
    return torch.fft.ifft2(torch.fft.ifftshift(K * m[None], dim=(-2, -1)), dim=(-2, -1)).abs()


def per_organ_dice(net, X, Y, band_t, beta=0.0, bs=16):
    inter = torch.zeros(NCLS); den = torch.zeros(NCLS)
    with torch.no_grad():
        for i in range(0, X.shape[0], bs):
            xb = X[i:i + bs]; yb = Y[i:i + bs]
            xin = (ablate(xb, band_t, beta) if band_t is not None else xb)[:, None]
            pred = net(xin).argmax(1)
            for c in range(1, NCLS):
                inter[c] += (2 * ((pred == c) & (yb == c)).sum()).item(); den[c] += ((pred == c).sum() + (yb == c).sum()).item()
    return {L.ABDO[ABDO_IDS[c - 1]]: (inter[c] / den[c]).item() if den[c] > 0 else float("nan") for c in range(1, NCLS)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=45); ap.add_argument("--width", type=float, default=0.06); ap.add_argument("--max_slices", type=int, default=6000); args = ap.parse_args()
    tr = np.load("data/slices/train.npz"); X = torch.tensor(tr["images"].astype(np.float32)); Y = torch.tensor(tr["labels"].astype(np.int64))
    te = np.load("data/slices/test.npz"); Xt = torch.tensor(te["images"].astype(np.float32)).to(dev); Yt = torch.tensor(te["labels"].astype(np.int64))
    if args.max_slices and X.shape[0] > args.max_slices:
        Yg = Y.to(torch.int16).to(dev); ws = torch.ones(Yg.shape[0], device=dev)
        for oid, b in [(6, 25.), (4, 2.), (12, 2.), (13, 2.), (17, .5)]: ws[(Yg == oid).flatten(1).any(1)] += b
        idx = torch.multinomial(ws, args.max_slices, replacement=True).cpu(); del Yg; torch.cuda.empty_cache(); X, Y = X[idx], Y[idx]
    X = X.to(dev); Y = remap_labels(Y.to(dev)).to(torch.uint8); Yt = remap_labels(Yt.to(dev)).to(torch.uint8); N = X.shape[-1]
    torch.manual_seed(0); net = UNet2D(NCLS).to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3); w = torch.ones(NCLS, device=dev); w[1:] = 3.0; bs = 16
    for ep in range(args.epochs):                            # train ONE clean segmenter
        perm = torch.randperm(X.shape[0], device=dev)
        for i in range(0, X.shape[0], bs):
            b = perm[i:i + bs]; loss = dice_ce(net(X[b].unsqueeze(1)), Y[b].long(), w); opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    law = {r["organ"]: r["centroid"] for r in json.load(open("outputs/results/m0_law_v2.json"))["rows"]}
    edges = np.arange(0, 1.0 + 1e-9, args.width)
    bands = [torch.tensor(annulus(N, edges[k], edges[k + 1]), device=dev) for k in range(len(edges) - 1)]
    with torch.no_grad():                                    # avg energy per annulus (for matched-energy control selection)
        P = (torch.fft.fftshift(torch.fft.fft2(Xt[:200]), dim=(-2, -1)).abs() ** 2)
        eng = np.array([float(P[:, b].sum().item()) for b in bands])
    ctr = np.array([(edges[k] + edges[k + 1]) / 2 for k in range(len(bands))])   # band centre radii
    clean = per_organ_dice(net, Xt, Yt, None)
    # --- OWN-BAND vs OTHER-ORGAN'S-BAND, at a matched, MEANINGFUL energy budget -------------------------------
    # v1 BUG (invalidated): budget = min(E_centroid, E_control) with control = FARTHEST band. The outermost annulus
    # holds ~0.03% of image energy, so the shared budget was ~0.03% and only ~0.5% of the centroid band was ever
    # removed -> every ablation was a no-op and every Dice drop came out 0.0000. Fixed here by (a) using the set of
    # organs' OWN centroid annuli as the candidate bands (all energy-rich), and (b) setting the budget to the
    # smallest energy among THOSE bands, so every ablation removes the same, non-trivial amount.
    ci_of = {}
    for oid in ABDO_IDS:
        nm = L.ABDO[oid]; c = law.get(nm)
        if c is None or np.isnan(clean[nm]): continue
        ci_of[oid] = min(len(bands) - 1, int(c / args.width))
    used = sorted(set(ci_of.values()))
    E_t = float(min(eng[k] for k in used))                                  # matched budget: removable from EVERY used band
    tot = float(eng.sum())
    beta = {k: float(np.sqrt(max(0.0, 1 - E_t / (float(eng[k]) + 1e-12)))) for k in used}
    print(f"  candidate bands (organ centroid annuli): {used}  radii={[round(float(ctr[k]),3) for k in used]}", flush=True)
    print(f"  matched energy budget E_t = {E_t:.4e} = {100*E_t/tot:.3f}% of image energy "
          f"(removes {100*E_t/float(eng[min(used)]):.1f}% of the richest candidate band)", flush=True)
    if 100 * E_t / tot < 0.2:
        print("  !! WARNING: budget <0.2% of image energy — ablation may be too weak to move Dice", flush=True)
    # per (organ, ablated band) Dice, so each organ is compared against ITSELF under different band ablations
    dice_ob = {k: per_organ_dice(net, Xt, Yt, bands[k], beta[k]) for k in used}
    res = {"width": args.width, "matched_energy": E_t, "matched_energy_pct_of_image": round(100 * E_t / tot, 4),
           "candidate_bands": {int(k): round(float(ctr[k]), 3) for k in used},
           "clean_dice": {o: round(clean[L.ABDO[o]], 3) for o in L.ABDO}, "organs": {}}
    dall, dfrag = [], []
    for oid, ci in ci_of.items():
        nm = L.ABDO[oid]
        others = [k for k in used if k != ci]
        if not others: continue
        drop_own = clean[nm] - dice_ob[ci][nm]
        drops_other = {int(k): round(clean[nm] - dice_ob[k][nm], 4) for k in others}
        drop_oth = float(np.mean(list(drops_other.values())))
        delta = drop_own - drop_oth                                          # >0 => own centroid band hurts MORE
        res["organs"][nm] = {"centroid": round(law[nm], 3), "own_band": int(ci), "own_radius": round(float(ctr[ci]), 3),
                             "drop_own_band": round(drop_own, 4), "drop_other_bands_mean": round(drop_oth, 4),
                             "drops_by_other_band": drops_other,
                             "own_minus_other": round(delta, 4), "fragile": oid in L.TAIL}
        dall.append(delta)
        if oid in L.TAIL: dfrag.append(delta)
    a = np.array(dall); af = np.array(dfrag)
    def wil(x): return float(wilcoxon(x, np.zeros_like(x)).pvalue) if len(x) > 1 and (x != 0).any() else None
    # SANITY GATE: if the ablation moved nothing at all, the experiment is a no-op (the v1 failure mode), NOT a null.
    all_drops = [v["drop_own_band"] for v in res["organs"].values()] + \
                [d for v in res["organs"].values() for d in v["drops_by_other_band"].values()]
    max_abs = float(max(abs(x) for x in all_drops)) if all_drops else 0.0
    res["SUMMARY"] = {"n_organs": len(a), "mean_own_minus_other": round(float(a.mean()), 4), "p_wilcoxon": wil(a),
                      "n_own_band_worse": int((a > 0).sum()),
                      "fragile_mean_own_minus_other": round(float(af.mean()), 4) if len(af) else None, "fragile_p": wil(af),
                      "max_abs_dice_change": round(max_abs, 4),
                      "ablation_is_a_noop": bool(max_abs < 0.005)}
    json.dump(res, open("outputs/results/intervention.json", "w"), indent=2)
    s = res["SUMMARY"]
    print(f"\n=== EQUAL-ENERGY INTERVENTION — own centroid band vs other organs' bands, matched energy ===")
    print(f"  budget {res['matched_energy_pct_of_image']}% of image energy | max |ΔDice| anywhere = {s['max_abs_dice_change']}")
    if s["ablation_is_a_noop"]:
        print("  !! INVALID: the ablation barely changed Dice anywhere -> this is a NO-OP, not a null result.")
        print("     Increase the energy budget (coarser --width, or drop the weakest candidate band) and re-run.")
    else:
        print(f"  ALL organs:     own-band vs other-band Δdrop = {s['mean_own_minus_other']:+.4f} (p={s['p_wilcoxon']}), "
              f"{s['n_own_band_worse']}/{s['n_organs']} hurt more by their OWN band")
        print(f"  FRAGILE (tail): Δdrop = {s['fragile_mean_own_minus_other']} (p={s['fragile_p']})")
        print("  -> CAUSAL support if Δ>0 & significant: same energy removed, but WHERE it is removed matters.")


if __name__ == "__main__":
    main()
