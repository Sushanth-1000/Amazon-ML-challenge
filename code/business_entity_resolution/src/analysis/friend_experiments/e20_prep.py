"""E20 step 1 (GCP CPU): cross-encoder France test, training groups for the PSEUDO arm.
E3 compared bge-reranker trained on US-only groups ('us') with the same number of US+India groups ('both', India with real
labels): India AP 0.9843 vs 0.9992. Here the 'pseudo' arm = the 'both' arm's US groups + the same number of India groups
whose labels come from the US-only stage-1 model (E1/E19 'reg', India never seen), i.e. what we can do on France:
  group = a record's top-8 candidates by the US-only probability (ce.py TOP_K); no-address groups x3 (as ce.py export)
  pseudo-labels (E19 rule, fixed a priori): best candidate = 1 if p >= 0.9 and p - second >= 0.3, others 0;
  records whose best p <= 0.05 -> all 0; other records skipped
India records come from split B (E1 EVAL buckets), EXCLUDING every record of E3's evaluation set, so train and
evaluation records never overlap. Real India labels are used only to report pseudo-label precision.
Usage: python e20_prep.py --work ~/w --src ~/src --e3 ~/e20/e3 --e1 ~/e20/e1_US --out ~/e20/pseudo"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True)
ap.add_argument("--src", required=True)
ap.add_argument("--e3", required=True, help="folder with both/ce_groups.parquet and eval/ce_score_train.parquet")
ap.add_argument("--e1", required=True, help="folder with E1 US-source eval_pairs.parquet and prob_US_reg.npy")
ap.add_argument("--out", required=True)
args = ap.parse_args()
sys.path.insert(0, args.src)
from ce import _text  # noqa: E402

TOP_K, HI, LO, MARGIN = 8, 0.9, 0.05, 0.3
s1 = pd.read_parquet(f"{args.work}/train_s1.parquet")
r = pd.read_parquet(f"{args.work}/train_r.parquet")
both = pd.read_parquet(f"{args.e3}/both/ce_groups.parquet")
both["q"] = both.gid.str.split("_").str[0].astype(np.int64)
both["country"] = r.country.values[both.q.values]
n_in = both.loc[both.country == "India", "q"].nunique()          # distinct India records (x2/x3 copies excluded)
ev = pd.read_parquet(f"{args.e3}/eval/ce_score_train.parquet", columns=["q"])
excl = set(ev.q.unique())
e1 = pd.read_parquet(f"{args.e1}/eval_pairs.parquet", columns=["q", "s", "y", "country"])
e1["p"] = np.load(f"{args.e1}/prob_US_reg.npy").astype(np.float32)
d = e1[(e1.country == "India") & ~e1.q.isin(excl)].sort_values(["q", "p"], ascending=[True, False])
d["rk"] = d.groupby("q").cumcount()
g = d[d.rk < TOP_K].copy()
top = g[g.rk == 0].set_index("q")
sec = g[g.rk == 1].set_index("q").p.reindex(top.index).fillna(0.0)
pos_q = top.index[(top.p >= HI) & (top.p - sec >= MARGIN)]
neg_q = top.index[top.p <= LO]
g = g[g.q.isin(set(pos_q) | set(neg_q))].copy()
g["ypseudo"] = ((g.rk == 0) & g.q.isin(set(pos_q))).astype(np.int8)
qs = g.q.unique()
rng = np.random.default_rng(0)
keep = rng.choice(qs, size=min(n_in, len(qs)), replace=False)
g = g[g.q.isin(set(keep))].copy()
pp = g[g.ypseudo == 1]
print(f"India pseudo groups {g.q.nunique():,} (target {n_in:,} = India groups of E3 'both') | pseudo positives {len(pp):,}, "
      f"precision vs real labels {pp.y.mean():.4f} | pseudo negatives truly negative {1 - g.y[g.ypseudo == 0].mean():.5f}", flush=True)
g["gid"] = g.q.astype(str) + "_p"
g["text_a"] = [_text(s1, i) for i in g.s.values]
g["text_b"] = [_text(r, i) for i in g.q.values]
na = g[r.addr.values[g.q.values] == ""]
g = pd.concat([g, na.assign(gid=na.gid + "_x2"), na.assign(gid=na.gid + "_x3")])     # no-address groups x3 (ce.py)
x = pd.concat([both[both.country == "US"][["gid", "text_a", "text_b", "y"]],
               g[["gid", "text_a", "text_b", "ypseudo"]].rename(columns={"ypseudo": "y"})], ignore_index=True)
os.makedirs(args.out, exist_ok=True)
x.to_parquet(f"{args.out}/ce_groups.parquet")
print(f"pseudo arm: {x.gid.nunique():,} groups ({len(x):,} pairs, positive share {x.y.mean():.3f}) | "
      f"E3 both arm: {both.gid.nunique():,} groups ({len(both):,} pairs, positive share {both.y.mean():.3f})", flush=True)
