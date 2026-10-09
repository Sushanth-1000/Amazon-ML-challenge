"""Diagnoses 6 + 7 on the FULL test set (no labels; test is only read): how confident v4 is per country, and how
the cross-encoder and stage 2 treat same-address / different-name pairs (co-located businesses) per country.
Run from student_resource/:  python test_profile.py --data dataset --work ~/w4 --src ~/src"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="dataset")
ap.add_argument("--work", required=True)
ap.add_argument("--src", required=True)
args = ap.parse_args()
sys.path.insert(0, args.src)
import lightgbm as lgb  # noqa: E402
from rapidfuzz import fuzz  # noqa: E402
from rapidfuzz.process import cpdist  # noqa: E402

import stage2 as S  # noqa: E402
from er import load_split  # noqa: E402

W = args.work
meta = json.load(open(f"{W}/meta.json"))
s1, r = load_split(args.data, "test", W)
d = pd.read_parquet(f"{W}/test_scored.parquet")
d = d[d.p1.values >= S.MIN_P1].reset_index(drop=True)
d = S.load_side(W, "test", d)
edges = S.load_edges(W, "test")
X1 = S.stage2_features(d, d.p1.values, s1, r, edges)[meta["stage2_features_1"]]
prob = np.mean([lgb.Booster(model_file=f"{W}/s2_pass1_{k}.txt").predict(X1, num_threads=0) for k in (0, 1)], axis=0)
a = S.decide(d, prob, meta["decision"], r)  # v4's test decisions (pass 1)
d["p2a"], d["country"] = prob, r.country.values[d.q.values]
d["n_tset"] = cpdist(list(r.name.values[d.q.values]), list(s1.name.values[d.s.values]), scorer=fuzz.token_set_ratio, workers=-1)
d["a_tset"] = cpdist(list(r.addr.values[d.q.values]), list(s1.addr.values[d.s.values]), scorer=fuzz.token_set_ratio, workers=-1)
d["noaddr"] = r.addr.values[d.q.values] == ""
d["assigned"] = pd.MultiIndex.from_frame(d[["q", "s"]]).isin(pd.MultiIndex.from_frame(a[["q", "s"]]))
best = d.sort_values(["q", "p2a"], ascending=[True, False]).drop_duplicates("q")
pd.set_option("display.width", 250)

print("=== D6: record-level confidence by country (each record's best candidate, pass-1 probability) ===")
print(best.groupby("country").p2a.agg(records="size", mean="mean", band_02_08=lambda v: ((v >= 0.2) & (v < 0.8)).mean(),
                                      over_09=lambda v: (v >= 0.9).mean(), assigned_share=lambda v: np.nan).drop(columns="assigned_share").round(4).to_string())
g = d[d.assigned]
print("\n=== D6: accepted matches by country ===")
print(g.groupby("country").agg(accepted=("q", "size"), p_below_08=("p2a", lambda v: (v < 0.8).mean()),
                               weak_name_lt60=("n_tset", lambda v: (v < 60).mean()),
                               colocated=("n_tset", lambda v: np.nan), noaddr=("noaddr", "mean")).drop(columns="colocated").round(4).to_string())
g = g.assign(coloc=(g.n_tset < 60) & (g.a_tset >= 90))
print("   same address (a_tset >= 90) + different name (n_tset < 60) share:",
      g.groupby("country").coloc.mean().round(4).to_dict())

print("\n=== D7: cross-encoder on co-located pairs (same address, different name), all scored test pairs ===")
has = d.ce.notna() if "ce" in d.columns else pd.Series(False, index=d.index)
c = d[has & (d.a_tset >= 90) & (d.n_tset < 60)]
print(c.groupby("country").agg(pairs=("q", "size"), ce_median=("ce", "median"), ce_yes_share=("ce", lambda v: (v > 0).mean()),
                               p2a_median=("p2a", "median"), accepted=("assigned", "mean")).round(4).to_string())
c = d[has & (d.n_tset >= 95)]
print("\n   reference: near-identical names (n_tset >= 95)")
print(c.groupby("country").agg(pairs=("q", "size"), ce_median=("ce", "median"), ce_yes_share=("ce", lambda v: (v > 0).mean()),
                               accepted=("assigned", "mean")).round(4).to_string())
