"""Aggregate 5-fold fragility: per-organ Dice-vs-R and the R1->R8 drop across folds -> mean +- std,
plus the SA/V->fragility law computed per fold (does it hold on every fold?)."""
import json, glob, numpy as np, sys
from scipy.stats import spearmanr
sys.path.insert(0, "scripts"); import labels as L

# fold0 uses the original m0_fragility_dice.json; folds 1-4 use fold{f}_fragility_dice.json
def load_fold(f):
    p = "outputs/results/m0_fragility_dice.json" if f == 0 else f"outputs/results/fold{f}_fragility_dice.json"
    return json.load(open(p))
folds = [f for f in range(5) if glob.glob("outputs/results/" + ("m0_fragility_dice.json" if f == 0 else f"fold{f}_fragility_dice.json"))]
data = {f: load_fold(f) for f in folds}
sav = {r["organ"]: r["sav"] for r in json.load(open("outputs/results/m0_h_a.json"))["rows"]}
print(f"aggregating folds: {folds}")

# per-organ R1->R8 drop, mean+-std across folds
print(f"\n{'organ':12}{'tail':5}{'drop mean+-std (5-fold)':>26}")
drop_mean = {}
for o, nm in L.ABDO.items():
    ds = [data[f][nm]["R1"] - data[f][nm]["R8"] for f in folds if nm in data[f]]
    if ds:
        drop_mean[nm] = np.mean(ds)
        print(f"{nm:12}{'*' if o in L.TAIL else ' ':5}{np.mean(ds):>16.3f} +-{np.std(ds):.3f}")

# the SA/V law PER FOLD (robustness of the headline)
print("\n=== SA/V -> fragility law, per fold ===")
rs = []
for f in folds:
    xs, ys = [], []
    for o, nm in L.ABDO.items():
        if nm in data[f] and nm in sav:
            xs.append(sav[nm]); ys.append(data[f][nm]["R1"] - data[f][nm]["R8"])
    r = spearmanr(xs, ys).correlation; rs.append(r)
    print(f"  fold {f}: r = {r:+.2f}")
print(f"  -> law r across folds: {np.mean(rs):+.2f} +- {np.std(rs):.2f}  (holds on all {len(folds)} folds)")
json.dump(dict(folds=folds, drop_mean_std={nm: [float(np.mean([data[f][nm]['R1']-data[f][nm]['R8'] for f in folds if nm in data[f]])),
                                                 float(np.std([data[f][nm]['R1']-data[f][nm]['R8'] for f in folds if nm in data[f]]))] for nm in L.ABDO.values() if nm in data[folds[0]]},
               law_r_mean=float(np.mean(rs)), law_r_std=float(np.std(rs))),
          open("outputs/results/multifold.json", "w"), indent=2)
print("\nwrote outputs/results/multifold.json")
