"""M1 mechanism de-risk: per-organ seg-loss GRADIENT MASS on the trained M0 nnU-Net.

The v2 method premise (NEXT_PROJECT_PLAN §0): in a task loss, large organs dominate the
per-organ gradient (liver has ~448x the voxels of an adrenal -> ~448x the CE gradient mass),
so an organ-agnostic loss implicitly under-serves small organs. The fix re-weights per organ.

BUT nnU-Net's loss is Dice+CE and **Dice is volume-normalized** -> it may already cancel much
of the imbalance. This probe measures, on the ACTUAL trained model + real preprocessed patches:
  per-organ |d L / d logits| mass, split into CE-term and Dice-term,
and reports the liver/adrenal mass ratio for each. That ratio == how much room the method has.

Gate read:
  - CE mass strongly liver-dominated (ratio >> 1) AND total (CE+Dice) still imbalanced
        -> gradient-imbalance mechanism is REAL on this loss -> rebalancing has room (proceed M2).
  - Dice term already ~balanced AND total ~balanced
        -> nnU-Net's Dice already rebalances -> the room (if any) is on the RECON side, not the
           seg loss -> re-scope the method honestly.

Run:
  python m1_gradient_probe.py --self_test                 # no model needed (validates the math)
  python m1_gradient_probe.py --dataset_id 501 --tr nnUNetTrainer_250epochs --n_cases 8
"""
import os, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from paths import RESULTS as OUT_R, PLOTS as OUT_P
import labels as L


def per_organ_grad_mass(logits, gt, organs, eps=1e-5):
    """logits: (1,C,...) leaf-or-grad tensor; gt: (...) int64. Returns per-organ CE/Dice grad mass
    (sum of |d term / d logits| over the whole logits tensor) for each organ present."""
    out = {}
    log_p = F.log_softmax(logits, dim=1)
    p = log_p.exp()
    for o in organs:
        g = (gt == o)
        n = int(g.sum())
        if n == 0:
            continue
        # ---- CE term for organ o: total NLL over its voxels (mass ~ N_o) ----
        if logits.grad is not None:
            logits.grad = None
        ce_o = -log_p[0, o][g].sum()
        ce_o.backward(retain_graph=True)
        ce_mass = float(logits.grad.abs().sum())
        # ---- soft-Dice term for organ o (volume-normalized) ----
        logits.grad = None
        p_o = p[0, o]
        g_f = g.float()
        inter = (p_o * g_f).sum()
        denom = p_o.sum() + g_f.sum()
        dc_o = 1.0 - (2 * inter + eps) / (denom + eps)
        dc_o.backward(retain_graph=True)
        dc_mass = float(logits.grad.abs().sum())
        out[o] = dict(n_vox=n, ce_mass=ce_mass, dc_mass=dc_mass, total_mass=ce_mass + dc_mass)
    return out


def _aggregate_and_report(acc, tag):
    """acc[o] = list of dicts -> mean per organ + liver/adrenal ratios + verdict."""
    organs = L.ABDO
    rows = []
    for o, nm in organs.items():
        if o not in acc or not acc[o]:
            continue
        ce = np.mean([d["ce_mass"] for d in acc[o]])
        dc = np.mean([d["dc_mass"] for d in acc[o]])
        nv = np.mean([d["n_vox"] for d in acc[o]])
        rows.append(dict(id=o, organ=nm, tail=o in L.TAIL, n_vox=nv,
                         ce_mass=ce, dc_mass=dc, total_mass=ce + dc))
    rows.sort(key=lambda r: -r["n_vox"])
    print(f"\n=== per-organ gradient mass ({tag}) — sorted large->small ===")
    print(f"{'organ':12} {'tail':5} {'n_vox':>9} {'CE_mass':>11} {'Dice_mass':>11} {'total':>11}")
    for r in rows:
        print(f"{r['organ']:12} {'*' if r['tail'] else ' ':5} {r['n_vox']:9.0f} "
              f"{r['ce_mass']:11.3e} {r['dc_mass']:11.3e} {r['total_mass']:11.3e}")

    def ratio(field):
        big = next((r[field] for r in rows if r["organ"] == "liver"), np.nan)
        sm = np.nanmean([r[field] for r in rows if r["id"] in (12, 13)])  # adrenals
        return big / sm if sm else np.nan

    r_ce, r_dc, r_tot = ratio("ce_mass"), ratio("dc_mass"), ratio("total_mass")
    nvr = ratio("n_vox")
    print(f"\nliver/adrenal  volume(n_vox) ratio = {nvr:7.1f}x")
    print(f"liver/adrenal  CE-grad   ratio = {r_ce:7.1f}x  (expect ~volume if CE is volume-dominated)")
    print(f"liver/adrenal  Dice-grad ratio = {r_dc:7.1f}x  (expect ~1 if Dice fully rebalances)")
    print(f"liver/adrenal  TOTAL     ratio = {r_tot:7.1f}x  <-- the room the method has")
    if r_tot >= 5:
        verdict = "MECHANISM REAL: seg-loss gradient still liver-dominated -> rebalancing has room (M2)."
    elif r_ce >= 5 and r_dc < 2:
        verdict = "PARTIAL: Dice already rebalances; imbalance lives in the CE term -> reweight CE / lean recon side."
    else:
        verdict = "WEAK on the seg loss: nnU-Net's Dice+CE already ~balanced -> method room is on the RECON side."
    print(f"=> {verdict}")
    return rows, dict(vol_ratio=nvr, ce_ratio=r_ce, dc_ratio=r_dc, total_ratio=r_tot, verdict=verdict)


def self_test():
    """Synthetic validation (no model): a big 'liver' blob + tiny 'adrenal' blob.
    Confirms CE mass scales with organ size while soft-Dice mass does NOT."""
    print("=== SELF TEST (synthetic logits, no model) ===")
    torch.manual_seed(0)
    C, D, H, W = 63, 16, 64, 64
    logits = torch.randn(1, C, D, H, W, requires_grad=True)
    gt = torch.zeros(D, H, W, dtype=torch.long)
    gt[2:14, 8:56, 8:56] = 5        # liver: large block  (~27k vox)
    gt[6:9, 30:34, 30:34] = 12      # adrenal_R: tiny block (~48 vox)
    acc = {o: [] for o in L.ABDO}
    m = per_organ_grad_mass(logits, gt, list(L.ABDO))
    for o, d in m.items():
        acc[o].append(d)
    _, summ = _aggregate_and_report(acc, "self-test")
    # correct assertion: CE tracks volume; Dice meaningfully rebalances RELATIVE to CE
    # (absolute Dice ratio on random logits is noisy, so compare against CE, not a fixed value)
    ok = (summ["ce_ratio"] > 20) and (summ["dc_ratio"] < summ["ce_ratio"] / 5)
    print(f"\nSELF TEST {'PASS' if ok else 'FAIL'}: CE mass tracks volume "
          f"(CE {summ['ce_ratio']:.0f}x vs volume {summ['vol_ratio']:.0f}x), "
          f"Dice rebalances ~{summ['ce_ratio']/summ['dc_ratio']:.0f}x relative to CE "
          f"(Dice {summ['dc_ratio']:.1f}x).")
    return ok


def run_real(args):
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2 as DS
    results = os.environ["nnUNet_results"]
    preproc = os.environ["nnUNet_preprocessed"]
    dsname = [d for d in os.listdir(results) if d.startswith(f"Dataset{args.dataset_id:03d}")][0]
    model_dir = f"{results}/{dsname}/{args.tr}__nnUNetPlans__{args.config}"
    print(f"loading model: {model_dir}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred = nnUNetPredictor(device=dev, allow_tqdm=False)
    pred.initialize_from_trained_model_folder(model_dir, use_folds=(args.fold,),
                                              checkpoint_name=args.checkpoint)
    net = pred.network
    net.load_state_dict(pred.list_of_parameters[0])
    net.eval().to(dev)
    try:
        net.decoder.deep_supervision = False
    except Exception as e:
        print("note: could not toggle deep_supervision:", e)

    pp_dir = f"{preproc}/{dsname}/nnUNetPlans_{args.config}"
    ids = DS.get_identifiers(pp_dir)
    ps = pred.configuration_manager.patch_size
    print(f"{len(ids)} preprocessed cases; patch_size={ps}; probing {args.n_cases} on {dev}")

    acc = {o: [] for o in L.ABDO}
    ds = DS(pp_dir)
    n_done = 0
    for cid in ids:
        if n_done >= args.n_cases:
            break
        data, seg, _, _ = ds.load_case(cid)
        data = np.asarray(data);  seg = np.asarray(seg)[0]            # (C,Z,Y,X), (Z,Y,X)
        # center a patch on a tail-organ voxel so small organs are actually present
        tail_vox = np.argwhere(np.isin(seg, list(L.TAIL)))
        if len(tail_vox) == 0:
            continue
        cz, cy, cx = tail_vox[len(tail_vox) // 2]
        sl = []
        for c, n, p in zip((cz, cy, cx), seg.shape, ps):
            lo = int(np.clip(c - p // 2, 0, max(n - p, 0)));  sl.append(slice(lo, lo + p))
        patch = data[(slice(None), *sl)]
        gtp = seg[tuple(sl)]
        if patch.shape[1:] != tuple(ps):                              # edge cases: pad
            pad = [(0, 0)] + [(0, p - s) for p, s in zip(ps, patch.shape[1:])]
            patch = np.pad(patch, pad);  gtp = np.pad(gtp, pad[1:])
        x = torch.from_numpy(patch[None]).float().to(dev)
        gt = torch.from_numpy(gtp.astype(np.int64)).to(dev)
        logits = net(x)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        logits = logits.detach().requires_grad_(True)               # probe grad at the logits
        m = per_organ_grad_mass(logits, gt, list(L.ABDO))
        for o, d in m.items():
            acc[o].append(d)
        n_done += 1
        print(f"  case {n_done}/{args.n_cases} ({cid}): organs present = {sorted(m)}")

    rows, summ = _aggregate_and_report(acc, f"real model, {n_done} cases")
    os.makedirs(OUT_R, exist_ok=True)
    json.dump(dict(rows=rows, summary=summ, n_cases=n_done),
              open(f"{OUT_R}/m1_gradient_probe.json", "w"), indent=2, default=float)
    print(f"\nwrote {OUT_R}/m1_gradient_probe.json")

    # figure: CE vs Dice mass per organ (log), tail in red
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rr = [r for r in rows if r["total_mass"] > 0]
    nm = [r["organ"] for r in rr]; x = np.arange(len(rr)); w = 0.4
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w/2, [r["ce_mass"] for r in rr], w, label="CE-term grad mass", color="#f4a259")
    ax.bar(x + w/2, [r["dc_mass"] for r in rr], w, label="Dice-term grad mass", color="#5b8def")
    ax.set_yscale("log"); ax.set_xticks(x)
    ax.set_xticklabels([f"{'*' if r['tail'] else ''}{n}" for n, r in zip(nm, rr)], rotation=45, ha="right")
    ax.set_ylabel("per-organ |dL/dlogits| mass (log)")
    ax.set_title(f"M1 mechanism de-risk: liver/adrenal TOTAL grad ratio = {summ['total_ratio']:.0f}x "
                 f"(CE {summ['ce_ratio']:.0f}x / Dice {summ['dc_ratio']:.1f}x)\n{summ['verdict']}",
                 fontsize=10, fontweight="bold")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT_P}/m1_gradient_probe.png", dpi=140)
    print(f"wrote {OUT_P}/m1_gradient_probe.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--dataset_id", type=int, default=501)
    ap.add_argument("--tr", default="nnUNetTrainer_250epochs")
    ap.add_argument("--config", default="3d_fullres")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--checkpoint", default="checkpoint_final.pth")
    ap.add_argument("--n_cases", type=int, default=8)
    args = ap.parse_args()
    if args.self_test:
        ok = self_test()
        raise SystemExit(0 if ok else 1)
    run_real(args)


if __name__ == "__main__":
    main()
