"""E16: would deeper cross-encoder scoring fix no-address ranking errors? (held-out H, v4's exact decisions)
prep  : for held-out no-address records whose true S1 is NOT v4's top-1 (ranking errors) and a control group of
        no-address records v4 got RIGHT, export each record's top-8 candidates (by v4 pass-1 p2a) with cross-encoder text.
eval  : after scoring with v4's cross-encoder: share of errors where the cross-encoder ranks the true S1 first (fixable)
        vs share of control records where it would push the true S1 off first place (breakable). Bootstrap 95% CIs.
Usage: python e16.py prep <work> <src>   |   python e16.py eval <work> <scored.parquet>"""
import json
import sys

import numpy as np
import pandas as pd

mode, W = sys.argv[1], sys.argv[2]
if mode == "prep":
    sys.path.insert(0, sys.argv[3])
    import er
    import stage2 as S
    from ce import _text
    s1, r = er.load_split("dataset", "train", W)
    truth, r2s = er.load_truth("dataset")
    meta = json.load(open(f"{W}/meta_v4.json"))
    d = pd.read_parquet(f"{W}/train_stage2.parquet", columns=["q", "s", "p1", "p2a"])
    s1_pos = pd.Series(np.arange(len(s1)), index=s1.entity_id.values)
    true_s = r.entity_id.map(r2s).map(s1_pos).fillna(-1).astype(np.int64).values
    h_ids = s1.entity_id.values[er.split_name(er.id_bucket(s1.entity_id.values)) == "H"]
    in_h = np.zeros(len(s1), bool); in_h[s1_pos[h_ids].values] = True
    keep = (true_s >= 0) & in_h[np.maximum(true_s, 0)] & (r.addr.values == "")
    dq = d[keep[d.q.values]].sort_values(["q", "p2a"], ascending=[True, False])
    dq["rank"] = dq.groupby("q").cumcount() + 1
    tr = dq[dq.s.values == true_s[dq.q.values]].set_index("q")["rank"]
    err_q, ok_q = tr[tr > 1].index.values, tr[tr == 1].index.values
    rng = np.random.default_rng(0)
    ctrl = rng.choice(ok_q, size=min(3000, len(ok_q)), replace=False)
    x = dq[dq.q.isin(set(err_q) | set(ctrl)) & (dq["rank"] <= 8)].copy()
    x["group"] = np.where(x.q.isin(set(err_q)), "error", "control")
    x["true"] = (x.s.values == true_s[x.q.values]).astype(np.int8)
    x["text_a"] = [_text(s1, i) for i in x.s.values]
    x["text_b"] = [_text(r, i) for i in x.q.values]
    x.to_parquet("/tmp/e16_pairs.parquet")
    print(f"no-address H records: ranking errors {len(err_q):,} (true S1 within top-8: "
          f"{int((tr[tr > 1] <= 8).sum()):,}) | control {len(ctrl):,} | pairs {len(x):,}")
else:
    x = pd.read_parquet(sys.argv[3])
    x = x.sort_values(["q", "ce"], ascending=[True, False])
    top_ce = x.drop_duplicates("q").set_index("q")
    rng = np.random.default_rng(0)
    for grp in ("error", "control"):
        g = top_ce[top_ce.group == grp]
        v = g.true.values.astype(float)
        bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(1000)]
        word = "cross-encoder ranks the true S1 first (fixable)" if grp == "error" else "true S1 stays first (not broken)"
        print(f"{grp:7s} records {len(g):,}: {word}: {v.mean():.3f}  95% CI [{np.quantile(bs, .025):.3f}, {np.quantile(bs, .975):.3f}]")
