"""v6.2b: cross-encoder training groups = v6.1's real groups (ce.py export: US + India, split A) + French pseudo-labelled
groups built exactly like E20's pseudo arm (which passed: unseen-country AP +0.0067 [+0.0061, +0.0073], source -0.0001):
  group = a French record's top-8 candidates by v6.1 stage-1 p1 (ce.py TOP_K); labels from fr_pseudo_labels.npy
  (E19/E20 rule); only records with a pseudo-match or whose best p1 <= 0.05; no-address groups x3 (ce.py export);
  262,264 French records sampled (seed 0) = E20's count.
Usage: python fr_ce_groups.py --work ~/w61 --data ~/dataset --out ~/v62b"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import er  # noqa: E402
from ce import TOP_K, _text  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True)
ap.add_argument("--data", default="dataset")
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=262264)
a = ap.parse_args()
W = a.work
s1t, rt_ = er.load_split(a.data, "test", W)
sc = pd.read_parquet(f"{W}/test_scored.parquet", columns=["q", "s", "p1"])
lab = np.load(f"{W}/fr_pseudo_labels.npy")
assert len(lab) == len(sc)
fr = rt_.country.values[sc.q.values] == "France"
d = sc[fr].assign(lab=lab[fr]).sort_values(["q", "p1"], ascending=[True, False])
d["rk"] = d.groupby("q").cumcount()
g = d[d.rk < TOP_K].copy()
ok = g.groupby("q").lab.transform(lambda x: (x >= 0).all() and ((x == 1).any() or (x == 0).all()))
g = g[ok.astype(bool)]
qs = g.q.unique()
keep = np.random.default_rng(0).choice(qs, size=min(a.n, len(qs)), replace=False)
g = g[g.q.isin(set(keep))].copy()
g["y"] = (g.lab == 1).astype(np.int8)
g["gid"] = "fr" + g.q.astype(str)
g["text_a"] = [_text(s1t, i) for i in g.s.values]
g["text_b"] = [_text(rt_, i) for i in g.q.values]
na = g[rt_.addr.values[g.q.values] == ""]
g = pd.concat([g, na.assign(gid=na.gid + "_x2"), na.assign(gid=na.gid + "_x3")])
real = pd.read_parquet(f"{W}/ce_groups.parquet")[["gid", "text_a", "text_b", "y"]]
x = pd.concat([real, g[["gid", "text_a", "text_b", "y"]]], ignore_index=True)
os.makedirs(a.out, exist_ok=True)
x.to_parquet(f"{a.out}/ce_groups.parquet")
print(f"French pseudo groups {g.gid.nunique():,} ({len(g):,} pairs, positive share {g.y.mean():.3f}) from "
      f"{len(qs):,} eligible French records | real groups {real.gid.nunique():,} | total {x.gid.nunique():,} groups, "
      f"{len(x):,} pairs", flush=True)
