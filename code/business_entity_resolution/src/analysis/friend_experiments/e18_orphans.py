"""E18: test-like orphans by REMOVING S1 (design from the teammate's sim_orphans.py), on v4's exact held-out decisions.
Test has ~5.75 records per S1 vs 4.68 in train while exact name+address copies per S1 are the same, i.e. ~19% of the
businesses behind test records have no S1. Here a share of NON-held-out train S1 is removed from the index (all held-out
S1 kept); their records become orphans that can attach to held-out S1, as in test. Reported on held-out half B:
  harm      = v4 decision under removal - v4 decision normal                (paired per S1)
  re-tuning = decision re-tuned on half A under removal - v4 decision, both under removal (paired per S1)
Uses train labels + test's unlabelled record counts only. Caveat (teammate's): features were computed with the true S1
present, so the harm is understated.  Usage: python e18_orphans.py <work> <src>"""
import json
import sys
import zlib

import numpy as np
import pandas as pd

W = sys.argv[1]
sys.path.insert(0, sys.argv[2])
import er  # noqa: E402
import stage2 as S  # noqa: E402

s1, r = er.load_split("dataset", "train", W)
truth, _ = er.load_truth("dataset")
meta = json.load(open(f"{W}/meta_v4.json"))
dec = meta["decision"]
d = pd.read_parquet(f"{W}/train_stage2.parquet", columns=["q", "s", "p2a"])
h_ids = s1.entity_id.values[er.split_name(er.id_bucket(s1.entity_id.values)) == "H"]
half = er.id_bucket(np.array([i[::-1] for i in h_ids])) < 50
ids_a, ids_b = h_ids[half], h_ids[~half]
th = {i: truth[i] for i in h_ids}


def per_s1(dd, dc, ids):
    pred = er.to_sets(S.decide(dd, dd.p2a.values, dc, r), s1, r)
    return np.array([er.macro_f05(pred, th, [i]) for i in ids])


rng = np.random.default_rng(0)
B = rng.integers(0, len(ids_b), size=(1000, len(ids_b)))
ci = lambda v: (v.mean(), np.quantile(v[B].mean(1), .025), np.quantile(v[B].mean(1), .975))
base_b = per_s1(d, dec, ids_b)
m, lo, hi = ci(base_b)
print(f"v4 normal: half B {m:.4f} [{lo:.4f}, {hi:.4f}]  ({len(ids_b):,} S1)", flush=True)
is_h = er.split_name(er.id_bucket(s1.entity_id.values)) == "H"
key = np.fromiter((zlib.crc32((i + "#orphan").encode()) % 1000 for i in s1.entity_id.values), np.int32, len(s1))
for share in (0.10, 0.19, 0.25):
    gone = (key < share * 1000) & ~is_h
    dd = d[~gone[d.s.values]].reset_index(drop=True)
    newly = r.entity_id.isin(set(x for i in s1.entity_id.values[gone] for x in truth.get(i, ()))).mean()
    f_b = per_s1(dd, dec, ids_b)
    m, lo, hi = ci(f_b - base_b)
    print(f"remove {share:.0%} of S1 ({newly:.1%} of records orphaned): v4 decision half B {f_b.mean():.4f} | "
          f"harm vs normal {m:+.4f} [{lo:+.4f}, {hi:+.4f}]", flush=True)
    if share == 0.19:
        dec2, fa2 = S.tune(dd, dd.p2a.values, r, s1, th, ids_a)
        f2 = per_s1(dd, dec2, ids_b)
        m, lo, hi = ci(f2 - f_b)
        print(f"   re-tuned on half A under removal: half B {f2.mean():.4f} | vs v4 decision {m:+.4f} [{lo:+.4f}, {hi:+.4f}]"
              f" | decision {dec2}", flush=True)
        g2 = per_s1(d, dec2, ids_b)
        m, lo, hi = ci(g2 - base_b)
        print(f"   same re-tuned decision on NORMAL data: {g2.mean():.4f} | vs v4 {m:+.4f} [{lo:+.4f}, {hi:+.4f}]", flush=True)
