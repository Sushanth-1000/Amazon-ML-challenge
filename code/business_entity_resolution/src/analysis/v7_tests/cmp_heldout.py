"""Paired held-out comparison of two stage-2 runs (per-S1 F0.5 on held-out half B, bootstrap 95% CI), by country.
Each run = a work folder with train_stage2.parquet + meta.json (decision + stage2_choice) from stage2.py fit.
Usage: python cmp_heldout.py --data dataset --work <any folder with train s1/r parquets> --a <run A> --b <run B>"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import er  # noqa: E402
import stage2 as S  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="dataset"); ap.add_argument("--work", required=True)
ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
a = ap.parse_args()
s1, r = er.load_split(a.data, "train", a.work)
truth, _ = er.load_truth(a.data)
h_ids = s1.entity_id.values[er.split_name(er.id_bucket(s1.entity_id.values)) == "H"]
half = er.id_bucket(np.array([i[::-1] for i in h_ids])) < 50
ids_b = h_ids[~half]
th = {i: truth[i] for i in h_ids}
COL = {"stage1": "p1", "pass1": "p2a", "pass2": "p2", "pass2+noaddr": "p2n"}


def per_s1(run):
    meta = json.load(open(f"{run}/meta.json"))
    d = pd.read_parquet(f"{run}/train_stage2.parquet")
    col = COL[meta.get("stage2_choice", "pass1")]
    pred = er.to_sets(S.decide(d, d[col].values, meta["decision"], r), s1, r)
    return np.array([er.macro_f05(pred, th, [i]) for i in ids_b]), meta.get("stage2_choice")


fa, ca = per_s1(a.a); fb, cb = per_s1(a.b)
cty = s1.set_index("entity_id").country.reindex(ids_b).values
rng = np.random.default_rng(0)
print(f"A = {a.a} ({ca}) | B = {a.b} ({cb})")
for c in ("US", "India", "all"):
    m = np.ones(len(ids_b), bool) if c == "all" else cty == c
    dd = fb[m] - fa[m]; B = rng.integers(0, m.sum(), size=(1000, m.sum()))
    lo, hi = np.quantile(dd[B].mean(1), [.025, .975])
    print(f"{c:5s} {m.sum():,} S1: A {fa[m].mean():.4f} | B {fb[m].mean():.4f} | B-A {dd.mean():+.5f} [{lo:+.5f}, {hi:+.5f}]"
          + ("  PASS" if lo > -0.0005 else "  FAIL (B worse)") if c == "all" else
          f"{c:5s} {m.sum():,} S1: A {fa[m].mean():.4f} | B {fb[m].mean():.4f} | B-A {dd.mean():+.5f} [{lo:+.5f}, {hi:+.5f}]")
