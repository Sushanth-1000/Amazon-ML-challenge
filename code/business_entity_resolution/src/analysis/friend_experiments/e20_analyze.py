"""E20 step 3: cross-encoder France test, same analysis as E3 (e3_analyze.py) + the pseudo arm.
AP of each cross-encoder on E3's split-B evaluation pairs per country; bootstrap over records (200 resamples, seed 0);
paired differences. Decision (fixed before looking): pass if pseudo - us on India is clearly > 0 (CI lower > 0) and
pseudo - us on US is not worse than -0.002 AP.
Usage: python e20_analyze.py <dir with ce_score_train.parquet, bge_us.parquet, bge_both.parquet, bge_pseudo.parquet>"""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as aps

D = sys.argv[1]
ev = pd.read_parquet(f"{D}/ce_score_train.parquet")[["q", "s", "y", "country", "p1"]]
names = ("bge_us", "bge_both", "bge_pseudo")
for n in names:
    ev = ev.merge(pd.read_parquet(f"{D}/{n}.parquet")[["q", "s", "ce"]].rename(columns={"ce": n}), on=["q", "s"], how="left")
assert ev[list(names)].notna().all().all(), "every evaluation pair must be scored by every model"
rng = np.random.default_rng(0)
for c in ("US", "India"):
    g = ev[ev.country == c].reset_index(drop=True)
    qs = g.q.unique(); grp = g.q.map({q: i for i, q in enumerate(qs)}).values
    W = [np.bincount(rng.integers(0, len(qs), len(qs)), minlength=len(qs))[grp] for _ in range(200)]
    ap = {n: aps(g.y, g[n]) for n in ("p1", *names)}
    bd = {n: np.array([aps(g.y, g[n], sample_weight=w) for w in W]) for n in ("p1", *names)}
    print(f"\n=== {c} ({len(g):,} split-B pairs, {len(qs):,} records; true share {g.y.mean():.3f}) ===")
    for n in ("p1", *names):
        print(f"  {n:11s} AP {ap[n]:.4f} [{np.quantile(bd[n], .025):.4f}, {np.quantile(bd[n], .975):.4f}]")
    for a, b in (("bge_pseudo", "bge_us"), ("bge_pseudo", "bge_both"), ("bge_us", "bge_both")):
        d = bd[a] - bd[b]
        print(f"  {a} - {b}: {ap[a] - ap[b]:+.4f}  CI [{np.quantile(d, .025):+.4f}, {np.quantile(d, .975):+.4f}]")
    if c == "India":
        gap = ap["bge_both"] - ap["bge_us"]
        print(f"  share of the US-only -> both gap closed by pseudo-labels: {(ap['bge_pseudo'] - ap['bge_us']) / gap:.2f}")
