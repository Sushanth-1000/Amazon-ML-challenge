"""Record matcher (S2/S3 record <-> S2/S3 record), used as graph evidence in stage 2.

A record without an address can hardly be told apart from same-named look-alikes by comparing it with S1 alone,
but it usually has a sibling (same business, other source) that carries an address and is matched confidently.
This model learns, from TRAIN labels only (two records match if they share their true S1), whether two records that
compete for the same S1 are the same business. Trained on split A; applied to all train pairs and to test.

Record pairs: for every S1, its top candidate records (stage-1 p1 >= P1_MIN, at most TOP_PER_S1) are paired.
Output: work/{train,test}_edges.parquet with columns q1, q2, prob (max over shared S1).

Usage (from student_resource/):  python src/matcher.py --data dataset --work work
"""
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

from er import id_bucket, load_split, load_truth, log, split_name

P1_MIN, TOP_PER_S1 = 0.05, 15
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
              feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=0)


def record_pairs(d):
    """Pairs of records that are top candidates of the same S1."""
    d = d[d.p1.values >= P1_MIN].sort_values(["s", "p1"], ascending=[True, False])
    d = d[d.groupby("s").cumcount().values < TOP_PER_S1][["q", "s", "p1"]]
    m = d.merge(d, on="s", suffixes=("1", "2"))
    m = m[m.q1.values < m.q2.values]
    agg = m.groupby(["q1", "q2"]).agg(p1_min=("p11", "min"), p1_max=("p11", "max"),
                                      p2_min=("p12", "min"), n_shared=("s", "size")).reset_index()
    return agg


def pair_features(e, r):
    f = pd.DataFrame(index=e.index)
    a, b = e.q1.values, e.q2.values
    for col, scorer, name in (("name", fuzz.token_set_ratio, "name_tset"), ("name_ns", fuzz.ratio, "ns_ratio"),
                              ("name_ns", JaroWinkler.normalized_similarity, "ns_jw"),
                              ("name_ph", fuzz.ratio, "ph_ratio"), ("addr", fuzz.token_set_ratio, "addr_tset"),
                              ("nums", fuzz.token_set_ratio, "nums_tset"), ("tail", fuzz.token_set_ratio, "tail_tset")):
        f[name] = cpdist(list(r[col].values[a]), list(r[col].values[b]), scorer=scorer, workers=-1, dtype=np.float32)
    f["codes_eq"] = [(-1.0 if not x or not y else float(set(x.split()) == set(y.split())))
                     for x, y in zip(r.codes.values[a], r.codes.values[b])]
    src = r.entity_id.str[:2].values
    f["same_src"] = (src[a] == src[b]).astype(np.float32)
    f["noaddr_1"] = (r.addr.values[a] == "").astype(np.float32)
    f["noaddr_2"] = (r.addr.values[b] == "").astype(np.float32)
    f["nonlatin_any"] = (r.nonlatin.values[a] | r.nonlatin.values[b]).astype(np.float32)
    for c in ("p1_min", "p1_max", "p2_min", "n_shared"):
        f[c] = e[c].values
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--work", default="work")
    args = ap.parse_args()
    s1, r = load_split(args.data, "train", args.work)
    truth, r2s = load_truth(args.data)
    d = pd.read_parquet(f"{args.work}/train_scored.parquet", columns=["q", "s", "p1", "split"])
    e = record_pairs(d)
    tr = r.entity_id.map(r2s).values
    rsplit = split_name(id_bucket(pd.Series(tr).fillna(pd.Series(r.entity_id.values)).values))
    y = ((tr[e.q1.values] == tr[e.q2.values]) & pd.notna(tr[e.q1.values])).astype(np.int8)
    in_a = (rsplit[e.q1.values] == "A") & (rsplit[e.q2.values] == "A")
    X = pair_features(e, r)
    hold = in_a & (id_bucket(r.entity_id.values[e.q1.values]) % 10 == 0)
    tra = in_a & ~hold
    m = lgb.train(PARAMS, lgb.Dataset(X[tra], y[tra]), num_boost_round=2000,
                  valid_sets=[lgb.Dataset(X[hold], y[hold])], callbacks=[lgb.early_stopping(50, verbose=False)])
    m.save_model(f"{args.work}/matcher.txt")
    prob = m.predict(X, num_threads=0)
    from sklearn.metrics import average_precision_score, roc_auc_score
    ev = ~in_a
    log(f"record pairs {len(e):,} (train on A {tra.sum():,}); positives {y.mean():.3f}; "
        f"not-A AUC {roc_auc_score(y[ev], prob[ev]):.4f} AP {average_precision_score(y[ev], prob[ev]):.4f}")
    e[["q1", "q2"]].assign(prob=prob.astype(np.float32)).to_parquet(f"{args.work}/train_edges.parquet")

    s1t, rt = load_split(args.data, "test", args.work)
    dt = pd.read_parquet(f"{args.work}/test_scored.parquet", columns=["q", "s", "p1"])
    et = record_pairs(dt)
    et[["q1", "q2"]].assign(prob=m.predict(pair_features(et, rt), num_threads=0).astype(np.float32)).to_parquet(
        f"{args.work}/test_edges.parquet")
    log(f"test record pairs {len(et):,}")


if __name__ == "__main__":
    main()
