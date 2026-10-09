"""E26 (Angle 1): per-S1 expected-F0.5 decisions vs v4's threshold decision, on v4's exact held-out data.
Metric = macro F0.5 per S1, singletons included (empty prediction on an S1 without matches scores 1).
  1. v4 decision (train_stage2 p2a + meta_v4 decision)                                  -> per-S1 F0.5 on half B
  2. calibration: isotonic regression of y on p2a, fitted on held-out half A pairs only
  3. expected-F0.5 decision: each record keeps only its single most likely S1 (v4 does the same); for each S1, sort
     its candidate records by calibrated p and pick the prefix k (k = 0..n) maximising
       E[F0.5] ~ 1.25*TP / (1.25*TP + 0.25*FN + FP)  with TP = sum p(chosen), FN = sum p(not chosen), FP = k - TP,
     and E[F0.5 | k=0] = prod(1 - p) (all candidates false -> empty is perfect)
  -> per-S1 F0.5 on half B; paired bootstrap 95% CI vs v4, by country. Also: singleton diagnostics.
Usage: python e26_expf05.py <work: train parquets, train_stage2.parquet, meta_v4.json> <src v5> [dataset dir]"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

W, SRC = sys.argv[1], sys.argv[2]
DS = sys.argv[3] if len(sys.argv) > 3 else "dataset"
sys.path.insert(0, SRC)
import er  # noqa: E402
import stage2 as S  # noqa: E402

s1, r = er.load_split(DS, "train", W)
truth, r2s = er.load_truth(DS)
dec = json.load(open(f"{W}/meta_v4.json"))["decision"]
d = pd.read_parquet(f"{W}/train_stage2.parquet", columns=["q", "s", "p2a", "y"])
h_ids = s1.entity_id.values[er.split_name(er.id_bucket(s1.entity_id.values)) == "H"]
half = er.id_bucket(np.array([i[::-1] for i in h_ids])) < 50
ids_a, ids_b = h_ids[half], h_ids[~half]
th = {i: truth[i] for i in h_ids}
s1_pos = pd.Series(np.arange(len(s1)), index=s1.entity_id.values)
inA = np.zeros(len(s1), bool); inA[s1_pos[ids_a].values] = True
inB = np.zeros(len(s1), bool); inB[s1_pos[ids_b].values] = True

# 1. v4 decision
pred_v4 = er.to_sets(S.decide(d, d.p2a.values, dec, r), s1, r)
f_v4 = np.array([er.macro_f05(pred_v4, th, [i]) for i in ids_b])

# 2. calibration on half A only (pairs whose S1 is in half A)
mA = inA[d.s.values]
iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(d.p2a.values[mA], d.y.values[mA])
pc = iso.predict(d.p2a.values).astype(np.float32)

# 3. expected-F0.5 decision per S1 on half B (each record -> its most likely S1 first)
x = d.assign(pc=pc)[inB[d.s.values]].sort_values(["q", "pc"], ascending=[True, False])
x = x[~x.q.duplicated()]                                     # one S1 per record
pred_ef = {}
for s, g in x.groupby("s"):
    p = np.sort(g.pc.values)[::-1]; recs = g.sort_values("pc", ascending=False).q.values
    tot = p.sum(); best_k, best = 0, float(np.prod(1 - p))
    tp = 0.0
    for k in range(1, len(p) + 1):
        tp += p[k - 1]
        e = 1.25 * tp / (1.25 * tp + 0.25 * (tot - tp) + (k - tp))
        if e > best:
            best, best_k = e, k
    pred_ef[s1.entity_id.values[s]] = set(r.entity_id.values[recs[:best_k]])
f_ef = np.array([er.macro_f05(pred_ef, th, [i]) for i in ids_b])

cty = s1.set_index("entity_id").country.reindex(ids_b).values
rng = np.random.default_rng(0)
print(f"=== E26 expected-F0.5 per-S1 decision vs v4 decision, held-out half B ({len(ids_b):,} S1) ===")
for c in ("US", "India", "all"):
    m = np.ones(len(ids_b), bool) if c == "all" else cty == c
    dd = f_ef[m] - f_v4[m]; B = rng.integers(0, m.sum(), size=(1000, m.sum()))
    print(f"{c:5s} {m.sum():,} S1: v4 {f_v4[m].mean():.4f} -> expected-F0.5 {f_ef[m].mean():.4f} | paired "
          f"{dd.mean():+.5f} [{np.quantile(dd[B].mean(1), .025):+.5f}, {np.quantile(dd[B].mean(1), .975):+.5f}]", flush=True)
# singleton diagnostics
sing = np.array([len(th[i]) == 0 for i in ids_b])
e_v4 = np.array([len(pred_v4.get(i, ())) == 0 for i in ids_b]); e_ef = np.array([len(pred_ef.get(i, ())) == 0 for i in ids_b])
print(f"true singletons {sing.mean():.4f} | predicted empty: v4 {e_v4.mean():.4f}, expected-F0.5 {e_ef.mean():.4f} | "
      f"singletons wrongly given matches: v4 {(sing & ~e_v4).sum():,}, expected-F0.5 {(sing & ~e_ef).sum():,} | "
      f"mean F0.5 on singletons: v4 {f_v4[sing].mean():.4f}, expected-F0.5 {f_ef[sing].mean():.4f}")
