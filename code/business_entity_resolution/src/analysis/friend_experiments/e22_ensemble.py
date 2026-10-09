"""E22: does averaging stage-1 models help? Offline, on E1's saved full-scale probabilities (no retraining).
Same evaluation as E1 (pair-level F0.5 per target country, threshold chosen on the SOURCE, cluster bootstrap with the
same seed and resamples, so every difference is paired with E1's numbers).
  avg3      = mean of the three base seeds (base, base_s1, base_s2): pure variance reduction
  base+reg  = mean of base and reg (two different settings)
  reg+cos   = mean of reg and E15's cos+reg (India source only, where E15 ran)
Usage: python e22_ensemble.py --data dataset --work ~/w --src ~/src --source US --probs ~/e22/US --out ~/e22"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="dataset")
ap.add_argument("--work", required=True)
ap.add_argument("--src", required=True)
ap.add_argument("--source", choices=["US", "India"], required=True)
ap.add_argument("--probs", required=True, help="folder with eval_pairs.parquet and prob_<SRC>_*.npy (E1 outputs)")
ap.add_argument("--out", required=True)
ap.add_argument("--boot", type=int, default=300)
args = ap.parse_args()
sys.path.insert(0, args.src)
import er  # noqa: E402
import stage2 as S  # noqa: E402

SRC, E0, E1 = args.source, 51, 100
s1, r = er.load_split(args.data, "train", args.work)
truth, r2s = er.load_truth(args.data)
rkey = r.entity_id.map(r2s).fillna(r.entity_id).values
rb = er.id_bucket(rkey)
ev = pd.read_parquet(f"{args.probs}/eval_pairs.parquet", columns=["q", "s", "country"])
P = lambda n: np.load(f"{args.probs}/prob_{SRC}_{n}.npy").astype(np.float32)
probs = {n: P(n) for n in ("base", "base_s1", "base_s2", "reg")}
assert all(len(v) == len(ev) for v in probs.values())
probs["avg3"] = (probs["base"] + probs["base_s1"] + probs["base_s2"]) / 3
probs["base+reg"] = (probs["base"] + probs["reg"]) / 2
cos_path = f"{args.probs}/e15prob_{SRC}_cos_reg.npy"
if os.path.exists(cos_path):
    probs["reg+cos"] = (probs["reg"] + np.load(cos_path).astype(np.float32)) / 2

# ---- identical to E1's evaluation (same records, clusters, bootstrap weights)
s1_pos = pd.Series(np.arange(len(s1)), index=s1.entity_id.values)
true_s = r.entity_id.map(r2s).map(s1_pos).fillna(-1).astype(np.int64).values
ev_rec = {c: np.flatnonzero((r.country.values == c) & (rb >= E0) & (rb < E1)) for c in ("US", "India")}
clusters = {c: pd.factorize(rkey[ev_rec[c]])[0] for c in ev_rec}
rng = np.random.default_rng(0)
WB = {c: rng.poisson(1.0, size=(args.boot, clusters[c].max() + 1)).astype(np.float32) for c in ev_rec}
grid = np.round(np.arange(0.02, 0.99, 0.02), 2)
country = ev.country.values


def outcomes(tgt, prob):
    m = country == tgt
    run = S._fast_decider(ev.loc[m, ["q", "s"]], prob[m], r)
    res = {}
    for t in grid:
        a = run({"t": {k: float(t) for k in S.SLICES}, "rescue": 0.0})
        asg = np.full(len(r), -2, np.int64); asg[a.q.values] = a.s.values
        got, tru = asg[ev_rec[tgt]], true_s[ev_rec[tgt]]
        tp = (got >= 0) & (got == tru)
        res[t] = (tp, (got >= 0) & ~tp, (tru >= 0) & ~tp)
    return res


def f05(tp, fp, fn):
    Pp, R = tp / np.maximum(tp + fp, 1e-9), tp / np.maximum(tp + fn, 1e-9)
    return 1.25 * Pp * R / np.maximum(0.25 * Pp + R, 1e-12)


def draws(tgt, o):
    cl = clusters[tgt]
    tp, fp, fn = (WB[tgt] @ np.bincount(cl, weights=x.astype(np.float32), minlength=cl.max() + 1) for x in o)
    return f05(tp, fp, fn)


D, rows = {}, []
for name, prob in probs.items():
    oc = {tgt: outcomes(tgt, prob) for tgt in ("US", "India")}
    curve = {tgt: {t: f05(*[x.sum() for x in oc[tgt][t]]) for t in grid} for tgt in oc}
    t_src = max(grid, key=lambda t: curve[SRC][t])
    for tgt in ("US", "India"):
        D[(tgt, name)] = draws(tgt, oc[tgt][t_src])
        rows.append(dict(source=SRC, target=tgt, config=name, F05=curve[tgt][t_src], t_src=t_src,
                         ci_lo=np.quantile(D[(tgt, name)], .025), ci_hi=np.quantile(D[(tgt, name)], .975)))
    print(f"{name}: " + " | ".join(f"{x['target']} {x['F05']:.4f} t={x['t_src']}" for x in rows[-2:]), flush=True)

os.makedirs(args.out, exist_ok=True)
pd.DataFrame(rows).to_csv(f"{args.out}/e22_{SRC}.csv", index=False)
print(f"\n=== E22 model averaging, source {SRC}: pair-level F0.5 (95% CI), threshold chosen on {SRC} ===")
print(pd.DataFrame(rows).round(4).to_string(index=False))
print("\n=== paired differences, 95% CI ===")
pairs = [("avg3", "base"), ("avg3", "base_s1"), ("avg3", "base_s2"), ("base+reg", "base"), ("base+reg", "reg")]
if "reg+cos" in probs:
    pairs += [("reg+cos", "reg")]
for tgt in ("US", "India"):
    for a, b in pairs:
        dd = D[(tgt, a)] - D[(tgt, b)]
        print(f"{SRC}->{tgt:5s} {a:8s} vs {b:8s}: {dd.mean():+.4f}  CI [{np.quantile(dd, .025):+.4f}, {np.quantile(dd, .975):+.4f}]")
    mean_single = np.mean([D[(tgt, n)] for n in ("base", "base_s1", "base_s2")], axis=0)
    dd = D[(tgt, "avg3")] - mean_single
    print(f"{SRC}->{tgt:5s} avg3     vs mean of the 3 single seeds: {dd.mean():+.4f}  CI [{np.quantile(dd, .025):+.4f}, {np.quantile(dd, .975):+.4f}]")
np.savez(f"{args.out}/draws22_{SRC}.npz", **{"|".join(k): v for k, v in D.items()})
print("E22 done")
