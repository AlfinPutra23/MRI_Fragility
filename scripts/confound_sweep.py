"""CONFOUND SWEEP — the standing audit that stops "is it just X?" surprises.

Motivation: we shipped a "spectral centroid predicts fragility" claim, then discovered LATE that BASELINE DIFFICULTY
(clean R1 Dice) predicts the drop BETTER (-0.93 vs +0.86) and dissolves the partial. That should have been caught on
day one by testing the claimed predictor against EVERY available rival, in both directions. This script does that
mechanically, so no rival can hide again.

For the outcome (drop) it tests the CLAIMED predictor (centroid) against every rival covariate and reports:
  - marginal Spearman + Pearson for every candidate, ranked (does any rival BEAT the claim?)
  - partial(claim, outcome | rival)  -- does the claim SURVIVE the rival?
  - partial(rival, outcome | claim)  -- does the rival survive the claim?  (asymmetry distinguishes the cases)
  - permutation p for both, and a Pearson-vs-Spearman disagreement flag (method-sensitivity = fragile claim)
  - a VERDICT per rival: SURVIVES / CONFOUNDED / MEDIATOR-LIKE / METHOD-SENSITIVE

Reading the verdicts:
  SURVIVES        claim keeps unique signal controlling for the rival -> safe to claim incremental value
  CONFOUNDED      rival survives, claim does not -> the rival is the better explanation; retreat
  MEDIATOR-LIKE   neither survives / both die -> entangled; if the claim is CAUSALLY PRIOR (anatomy precedes model
                  performance) the rival may be a MEDIATOR, and controlling for it is over-adjustment. Argue the
                  causal ordering explicitly; do not silently drop the claim.
  METHOD-SENSITIVE Pearson and Spearman disagree on significance -> report the conservative one, treat as fragile.

  python scripts/confound_sweep.py                    # default: centroid -> drop, MRISeg + AMOS
-> outputs/results/confound_sweep.json
"""
import json, os, argparse, numpy as np
from scipy import stats
np.random.seed(0)


def z(v): return (v - v.mean()) / (v.std() + 1e-12)


def _partial(x, y, Z, spearman):
    """partial correlation of x,y controlling for columns Z (rank-transformed if spearman)."""
    tf = (lambda v: stats.rankdata(v)) if spearman else (lambda v: v)
    xx, yy = tf(x), tf(y)
    A = np.column_stack([np.ones_like(xx)] + [z(tf(c)) for c in Z])
    rx = xx - A @ np.linalg.lstsq(A, xx, rcond=None)[0]
    ry = yy - A @ np.linalg.lstsq(A, yy, rcond=None)[0]
    return float(stats.pearsonr(rx, ry)[0])


def perm_p(x, y, Z, obs, spearman, n=4000):
    return float(np.mean([abs(_partial(x, np.random.permutation(y), Z, spearman)) >= abs(obs) for _ in range(n)]))


def sweep(name, y, claim_name, claim, rivals, alpha=0.05):
    out = {"dataset": name, "n": int(len(y)), "claim": claim_name, "marginal": {}, "vs_rival": {}}
    # --- marginal leaderboard: does any rival simply beat the claim? ---
    cand = dict(rivals); cand[claim_name] = claim
    for k, v in cand.items():
        rs, ps = stats.spearmanr(v, y); rp, pp = stats.pearsonr(v, y)
        out["marginal"][k] = {"spearman": round(float(rs), 3), "p_spearman": round(float(ps), 4),
                              "pearson": round(float(rp), 3), "abs_spearman": round(abs(float(rs)), 3)}
    rank = sorted(out["marginal"], key=lambda k: -out["marginal"][k]["abs_spearman"])
    out["marginal_ranking"] = rank
    out["claim_is_top_marginal"] = bool(rank[0] == claim_name)
    out["rivals_beating_claim"] = [k for k in rank if k != claim_name
                                   and out["marginal"][k]["abs_spearman"] > out["marginal"][claim_name]["abs_spearman"]]
    # --- head-to-head: both directions, both methods ---
    for rk, rv in rivals.items():
        rec = {}
        for meth, sp in [("spearman", True), ("pearson", False)]:
            c = _partial(claim, y, [rv], sp)          # claim controlling rival
            r = _partial(rv, y, [claim], sp)          # rival controlling claim
            rec[meth] = {"claim_given_rival": round(c, 3), "p_claim": round(perm_p(claim, y, [rv], c, sp), 4),
                         "rival_given_claim": round(r, 3), "p_rival": round(perm_p(rv, y, [claim], r, sp), 4)}
        s = rec["spearman"]
        claim_ok = s["p_claim"] < alpha
        rival_ok = s["p_rival"] < alpha
        method_split = (rec["spearman"]["p_claim"] < alpha) != (rec["pearson"]["p_claim"] < alpha)
        if method_split:                       verdict = "METHOD-SENSITIVE"
        elif claim_ok and not rival_ok:        verdict = "SURVIVES"
        elif rival_ok and not claim_ok:        verdict = "CONFOUNDED"
        elif claim_ok and rival_ok:            verdict = "SURVIVES (both independent)"
        else:                                  verdict = "MEDIATOR-LIKE (neither unique)"
        rec["verdict"] = verdict
        out["vs_rival"][rk] = rec
    worst = [k for k, v in out["vs_rival"].items() if v["verdict"].startswith(("CONFOUNDED", "MEDIATOR"))]
    out["rivals_that_dissolve_the_claim"] = worst
    out["OVERALL"] = ("CLEAN — claim survives every rival" if not worst and not out["rivals_beating_claim"]
                      else f"AT RISK — dissolved by {worst}; out-predicted by {out['rivals_beating_claim']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/results/confound_sweep.json")
    args = ap.parse_args()
    RES = "outputs/results"
    results = {}

    # ---------- MRISegmentator: full rival set INCLUDING baseline difficulty ----------
    rows = json.load(open(f"{RES}/m0_law_v2.json"))["rows"]
    org = [r["organ"] for r in rows]
    y = np.array([r["drop"] for r in rows], float)
    claim = np.array([r["centroid"] for r in rows], float)
    rivals = {k: np.array([r[k] for r in rows], float) for k in ["sav", "contrast", "hf8_img", "fdim"]}
    rig = {r["organ"]: r for r in json.load(open(f"{RES}/m0_rigor.json"))["rows"]}
    if all(o in rig for o in org):                       # THE rival we found late
        rivals["clean_dice_R1"] = np.array([rig[o]["dice_R1"] for o in org], float)
        rivals["nsd_R1"] = np.array([rig[o]["nsd_R1"] for o in org], float)
    results["MRISegmentator"] = sweep("MRISegmentator", y, "centroid", claim, rivals)

    # ---------- AMOS ----------
    rows = json.load(open(f"{RES}/amos_law_v2.json"))["rows"]
    y = np.array([r["drop"] for r in rows], float)
    claim = np.array([r["centroid"] for r in rows], float)
    rivals = {k: np.array([r[k] for r in rows], float) for k in ["sav", "contrast", "hf8_img", "fdim"]}
    results["AMOS"] = sweep("AMOS", y, "centroid", claim, rivals)

    json.dump(results, open(args.out, "w"), indent=2)
    for name, o in results.items():
        print(f"\n{'='*78}\n{name}  (n={o['n']}, claim = {o['claim']})\n{'='*78}")
        print("  MARGINAL leaderboard (|Spearman| vs drop):")
        for k in o["marginal_ranking"]:
            m = o["marginal"][k]; star = "  <-- CLAIM" if k == o["claim"] else ""
            print(f"    {k:16s} rho={m['spearman']:+.3f}  (pearson {m['pearson']:+.3f}){star}")
        if o["rivals_beating_claim"]:
            print(f"  !! rivals OUT-PREDICTING the claim: {o['rivals_beating_claim']}")
        print("\n  HEAD-TO-HEAD (rank-based partials):")
        for rk, rec in o["vs_rival"].items():
            s = rec["spearman"]
            print(f"    vs {rk:16s} claim|rival={s['claim_given_rival']:+.3f} (p={s['p_claim']:.3f}) | "
                  f"rival|claim={s['rival_given_claim']:+.3f} (p={s['p_rival']:.3f})  -> {rec['verdict']}")
        print(f"\n  >> OVERALL: {o['OVERALL']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
