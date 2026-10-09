"""E27: does self-training (the v6.2a France recipe) raise the LEADERBOARD metric on an unseen country?
E19 measured pair-level F0.5; this converts E19's saved per-pair probabilities into macro F0.5 per S1, singletons
included (an S1 with no true match and no prediction scores 1), on the unlabelled country's EVAL S1 (India plays France).
Decision exactly as E19: v4 decoder, one threshold chosen on the SOURCE country (US) per config.
Usage: python e27_pers1.py --data dataset --work <w with loco_pairs_2_100.parquet> --src <src5> --probs <e19 out>
       --tag _std --configs reg,st1"""
import argparse
import sys

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="dataset"); ap.add_argument("--work", required=True); ap.add_argument("--src", required=True)
ap.add_argument("--probs", required=True); ap.add_argument("--tag", default=""); ap.add_argument("--configs", default="reg,st1")
ap.add_argument("--source", default="US")
a = ap.parse_args()
sys.path.insert(0, a.src)
import er  # noqa: E402
import stage2 as S  # noqa: E402

SRC = a.source; TGT = "India" if SRC == "US" else "US"
s1, r = er.load_split(a.data, "train", a.work)
truth, r2s = er.load_truth(a.data)
rkey = r.entity_id.map(r2s).fillna(r.entity_id).values
rb = er.id_bucket(rkey)
p = pd.read_parquet(f"{a.work}/loco_pairs_2_100.parquet", columns=["q", "s", "y"])
country = r.country.values[p.q.values]
evm = rb[p.q.values] >= 51                                   # same EVAL definition as E19 (buckets [51,100))
grid = np.round(np.arange(0.30, 0.991, 0.02), 2)

# EVAL S1 of each country: S1 whose bucket is in [51,100) (a matched record's bucket is its S1's bucket)
sb = er.id_bucket(s1.entity_id.values)
ev_s1 = {c: np.flatnonzero((sb >= 51) & (s1.country.values == c)) for c in ("US", "India")}
sid, rid = s1.entity_id.values, r.entity_id.values


def per_s1(c, prob_ev, t):
    m = evm & (country == c)
    pe = np.zeros(len(p), np.float32); pe[evm] = prob_ev
    run = S._fast_decider(p.loc[m, ["q", "s"]], pe[m], r)
    asg = run({"t": {k: float(t) for k in S.SLICES}, "rescue": 0.0})
    pred = {}
    for q, s in zip(asg.q.values, asg.s.values):
        pred.setdefault(s, set()).add(rid[q])
    out = np.empty(len(ev_s1[c]))
    for k, s in enumerate(ev_s1[c]):
        P, G = pred.get(s, set()), truth.get(sid[s], set())
        if not P and not G: out[k] = 1.0; continue
        tp = len(P & G)
        if tp == 0: out[k] = 0.0; continue
        pr, rc = tp / len(P), tp / len(G); out[k] = 1.25 * pr * rc / (0.25 * pr + rc)
    return out


res = {}
for name in a.configs.split(","):
    prob_ev = np.load(f"{a.probs}/prob{a.tag}_{SRC}_{name}.npy")
    # threshold chosen on the SOURCE country, per-S1 metric (no target labels used)
    curve = {t: per_s1(SRC, prob_ev, t).mean() for t in grid}
    t_src = max(curve, key=curve.get)
    res[name] = {c: per_s1(c, prob_ev, t_src) for c in ("US", "India")}
    print(f"{name}: t chosen on {SRC} = {t_src} | " + " | ".join(f"{c} per-S1 {res[name][c].mean():.4f}" for c in res[name]),
          flush=True)

rng = np.random.default_rng(0)
names = a.configs.split(",")
base = names[0]
print(f"\n=== E27 per-S1 macro F0.5 (singletons included), EVAL S1, threshold from {SRC}; paired bootstrap 1000 ===")
for c in (TGT, SRC):
    n = len(ev_s1[c]); B = rng.integers(0, n, size=(1000, n))
    for name in names[1:]:
        d = res[name][c] - res[base][c]; bm = d[B].mean(1)
        print(f"{c:5s} ({'unlabelled' if c == TGT else 'source'}) {n:,} S1: {base} {res[base][c].mean():.4f} -> {name} "
              f"{res[name][c].mean():.4f} | paired {d.mean():+.5f} [{np.quantile(bm, .025):+.5f}, {np.quantile(bm, .975):+.5f}]"
              f" | S1 improved {(d > 1e-9).sum():,}, worsened {(d < -1e-9).sum():,}")
sing = np.array([len(truth.get(sid[s], ())) == 0 for s in ev_s1[TGT]])
for name in names:
    print(f"{name}: {TGT} singletons {sing.mean():.4f} of S1, mean F0.5 on singletons {res[name][TGT][sing].mean():.4f}, "
          f"on matched S1 {res[name][TGT][~sing].mean():.4f}")
