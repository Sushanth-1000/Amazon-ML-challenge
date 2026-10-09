"""E23: does the teammates' France look-alike filter (leaderboard +0.00055 on France) also help US / India?
Applies their exact decoy_pairs() rule (france_lookalike_filter.py, unchanged) per country to the TRAIN split, removes the
flagged (S1, record) pairs from v4's exact held-out decisions (train_stage2 p2a + v4 decision, as E18), and reports
per-S1 F0.5 on held-out half B before/after, paired bootstrap 95% CI, plus how many removed pairs were true matches.
Usage: python e23_lookalike.py <work with train parquets, train_stage2.parquet, meta_v4.json> <src> <filter dir>"""
import json
import sys

import numpy as np
import pandas as pd

W, SRC, FD = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, SRC); sys.path.insert(0, FD)
import er  # noqa: E402
import stage2 as S  # noqa: E402
from france_lookalike_filter import decoy_pairs  # noqa: E402

s1, r = er.load_split("dataset", "train", W)
truth, r2s = er.load_truth("dataset")
dec = json.load(open(f"{W}/meta_v4.json"))["decision"]
d = pd.read_parquet(f"{W}/train_stage2.parquet", columns=["q", "s", "p2a"])
h_ids = s1.entity_id.values[er.split_name(er.id_bucket(s1.entity_id.values)) == "H"]
half = er.id_bucket(np.array([i[::-1] for i in h_ids])) < 50
ids_b = h_ids[~half]
th = {i: truth[i] for i in h_ids}
pred = er.to_sets(S.decide(d, d.p2a.values, dec, r), s1, r)
rd = lambda f: er.read_tsv(f"dataset/train/{f}")
S1raw = rd("train_source1.tsv"); Rraw = pd.concat([rd("train_source2.tsv"), rd("train_source3.tsv")], ignore_index=True)
bad = set()
for c in ("US", "India"):
    b = decoy_pairs(S1raw[S1raw.country == c].reset_index(drop=True), Rraw[Rraw.country == c].reset_index(drop=True))
    tr = sum(x in truth.get(sid, ()) for sid, x in b)
    print(f"{c}: flagged pairs {len(b):,} | of them true matches {tr:,} ({tr / max(len(b), 1):.3f})", flush=True)
    bad |= b
after = {sid: {x for x in ms if (sid, x) not in bad} for sid, ms in pred.items()}
removed = [(sid, x) for sid, ms in pred.items() if sid in th for x in ms if (sid, x) in bad]
print(f"held-out predictions removed {len(removed):,} (true matches among them {sum(x in th[s] for s, x in removed):,})")
f0 = np.array([er.macro_f05(pred, th, [i]) for i in ids_b])
f1 = np.array([er.macro_f05(after, th, [i]) for i in ids_b])
cty = s1.set_index("entity_id").country.reindex(ids_b).values
rng = np.random.default_rng(0)
for c in ("US", "India", "all"):
    m = np.ones(len(ids_b), bool) if c == "all" else cty == c
    dd = f1[m] - f0[m]; B = rng.integers(0, m.sum(), size=(1000, m.sum()))
    print(f"{c:5s} half B {m.sum():,} S1: v4 {f0[m].mean():.4f} -> filtered {f1[m].mean():.4f} | paired diff {dd.mean():+.5f} "
          f"[{np.quantile(dd[B].mean(1), .025):+.5f}, {np.quantile(dd[B].mean(1), .975):+.5f}]", flush=True)
