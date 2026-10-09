"""Stage 2 (v4): collective re-scoring with sibling context, source coverage, record-graph edges and cross-encoder
scores, plus the final decision layer.

Training: split B of the train S1 entities (see er.py). Features only use stage-1 probabilities, cross-encoder /
record-matcher scores and text, never labels, so they are computed identically on train and test.
  pass 1  features from stage-1 p1 -> model p2a (2 entity-grouped folds inside B -> out-of-fold p2a on B)
  pass 2  sibling / coverage / graph features recomputed from pass-1 assignments -> model p2
  no-address records get their own pass-2 model if it wins on held-out half A
Decision (tuned on held-out H half A, reported on H half B): per-slice thresholds (address x script), one S1 per
record, zero-prediction rescue for S1 entities left empty, cap of 11 records per S1.

Usage (from student_resource/):  python src/stage2.py --stage fit    # train + tune + report on H
                                 python src/stage2.py --stage apply  # test -> output/matching_results.tsv
"""
import argparse
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from er import (MAX_PER_S1, STAGE2_B, _first_int, id_bucket, load_split, load_truth, log, macro_f05, split_name, to_sets,
                write_lists)

MIN_P1 = 0.002   # v3: 0.005 (audit: 57 true val pairs lost below it)
ANCHOR = 0.5     # a record "confidently claims" its best S1 above this probability
PARAMS2 = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
               feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=0)
ROUNDS2 = 500
EXTRA = ["n_exact_cands", "is_unique_exact", "s1_name_freq", "s1_addr_freq", "codes_eq", "num_small_shift",
         "n_tset", "a_tset"]


# ---------------------------------------------------------------- inputs

def load_side(work, split, d):
    """Attach optional cross-encoder scores (pointwise, listwise, symmetric) to pairs d."""
    path = f"{work}/{split}_ce.parquet"
    if os.path.exists(path):
        ce = pd.read_parquet(path)
        d = d.merge(ce, on=["q", "s"], how="left")
    return d


def load_edges(work, split):
    path = f"{work}/{split}_edges.parquet"
    return pd.read_parquet(path) if os.path.exists(path) else None


# ---------------------------------------------------------------- features

def _claims(d, prob):
    """Each record's best S1 if its probability clears ANCHOR: DataFrame (q, s, p)."""
    t = d.assign(_p=prob).sort_values(["q", "_p"], ascending=[True, False]).drop_duplicates("q")
    return t[t._p >= ANCHOR][["q", "s", "_p"]].rename(columns={"_p": "p"})


def stage2_features(d, prob, s1, r, edges):
    """d: pairs (q, s, p1, extras, optional ce columns); prob: the probability defining 'confident claims'."""
    f = pd.DataFrame({"p": prob}, index=d.index)
    g = f.groupby(d.q.values).p
    f["q_rank"] = g.rank(ascending=False, method="first").values
    f["q_max"] = g.transform("max").values
    f["q_n"] = g.transform("size").values
    order = np.lexsort((-prob, d.q.values))  # second-best probability of each record, vectorized
    qs, ps = d.q.values[order], prob[order]
    start = np.flatnonzero(np.r_[True, qs[1:] != qs[:-1]])
    nxt = np.minimum(start + 1, len(qs) - 1)
    second = pd.Series(np.where((nxt > start) & (qs[nxt] == qs[start]), ps[nxt], 0.0), index=qs[start])
    f["q_gap"] = np.where(f.q_rank.values == 1, prob - d.q.map(second).values, prob - f.q_max.values)
    for c in [c for c in d.columns if c.startswith("ce")]:
        v = d[c].values.astype(np.float32)
        f[c] = v
        f[c + "_rank"] = pd.Series(v).groupby(d.q.values).rank(ascending=False, method="first").values
        f[c + "_gap"] = v - pd.Series(v).groupby(d.q.values).transform("max").values
    for c in EXTRA:
        if c in d.columns:
            f[c] = d[c].values

    cl = _claims(d, prob)
    src = r.entity_id.str[:2].values
    has_addr = r.addr.values != ""
    cl = cl.assign(src=src[cl.q.values], addr=has_addr[cl.q.values])
    own = (f.q_rank.values == 1) & (prob >= ANCHOR)
    by = cl.groupby("s")
    n_all = d.s.map(by.size()).fillna(0).values - own
    f["s_claims"] = n_all
    f["s_claim_sum"] = d.s.map(by.p.sum()).fillna(0).values - np.where(own, prob, 0)
    same = cl.groupby(["s", "src"]).size()
    key_same = pd.MultiIndex.from_arrays([d.s.values, src[d.q.values]])
    n_same = same.reindex(key_same).fillna(0).values - own
    f["s_claims_same_src"] = n_same
    f["s_claims_other_src"] = n_all - n_same
    f["s_lacks_this_src"] = (n_same <= 0).astype(np.float32)
    f["s_claims_with_addr"] = d.s.map(cl[cl.addr].groupby("s").size()).fillna(0).values - (own & has_addr[d.q.values])

    # sibling agreement with the confident claimants of the same S1
    a = d[["q", "s"]].reset_index().merge(cl[["q", "s"]].rename(columns={"q": "qa"}), on="s")
    a = a[a.qa.values != a.q.values]
    rq, ra = a.q.values, a.qa.values
    a["sib_addr"] = cpdist(list(r.addr.values[rq]), list(r.addr.values[ra]), scorer=fuzz.token_set_ratio, workers=-1)
    a["sib_name"] = cpdist(list(r.name.values[rq]), list(r.name.values[ra]), scorer=fuzz.token_set_ratio, workers=-1)
    rnum, snum = _first_int(r.nums.values), _first_int(s1.nums.values)
    fq, fa, fs = rnum[rq], rnum[ra], snum[a.s.values]
    a["sib_num_eq"] = ((fq >= 0) & (fq == fa)).astype(np.float32)
    a["sib_num_is_s1"] = ((fa >= 0) & (fa == fs)).astype(np.float32)
    agg = a.groupby("index").agg(sib_addr_max=("sib_addr", "max"), sib_name_max=("sib_name", "max"),
                                 sib_num_eq_n=("sib_num_eq", "sum"), sib_num_is_s1_frac=("sib_num_is_s1", "mean"))
    f = f.join(agg)
    f[agg.columns] = f[agg.columns].fillna(-1)

    # record graph (learned S2<->S3 matcher): support from confident siblings of this S1 vs pull to another S1
    f["graph_support"], f["graph_conflict"] = -1.0, -1.0
    if edges is not None and len(edges):
        e = pd.concat([edges, edges.rename(columns={"q1": "q2", "q2": "q1"})], ignore_index=True)
        best = cl.set_index("q").s
        e["s_other"] = e.q2.map(best)
        e = e[e.s_other.notna()]
        sup = e.groupby(["q1", "s_other"]).prob.max()
        k = pd.MultiIndex.from_arrays([d.q.values, d.s.values])
        f["graph_support"] = sup.reindex(k).fillna(0).values
        tot = e.groupby("q1").prob.max()
        f["graph_conflict"] = np.where(f.graph_support.values < d.q.map(tot).fillna(0).values,
                                       d.q.map(tot).fillna(0).values, 0.0)

    fr, fs1 = rnum[d.q.values], snum[d.s.values]
    f["num_eq_s1"] = np.where((fr < 0) | (fs1 < 0), -1, fr == fs1)
    f["noaddr"] = ~has_addr[d.q.values]
    f["nonlatin"] = r.nonlatin.values[d.q.values]
    f["p1"] = d.p1.values
    return f.drop(columns=["p"]).astype(np.float32)


# ---------------------------------------------------------------- decision

SLICES = ("noaddr", "native", "other")


def slice_of(r, q):
    na = r.addr.values[q] == ""
    nat = r.nonlatin.values[q]
    return np.where(na, "noaddr", np.where(nat, "native", "other"))


def decide(d, prob, dec, r):
    """Each record -> its best S1 if prob passes its slice threshold; zero-prediction rescue; cap per S1."""
    t = d[["q", "s"]].assign(p=prob, sl=slice_of(r, d.q.values)).sort_values(["q", "p"], ascending=[True, False])
    best = t.drop_duplicates("q")
    thr = best.sl.map(dec["t"]).values
    out = best[best.p.values >= thr]
    if dec.get("rescue"):  # S1 entities left with nothing: take their best unassigned record if likely enough
        empty = ~t.s.isin(set(out.s.values))
        free = ~t.q.isin(set(out.q.values))
        cand = t[empty & free & (t.p.values >= dec["rescue"])].sort_values("p", ascending=False)
        cand = cand.drop_duplicates("s").drop_duplicates("q")
        out = pd.concat([out, cand])
    out = out.sort_values(["s", "p"], ascending=[True, False])
    return out[out.groupby("s").cumcount().values < dec.get("cap", MAX_PER_S1)]


def tune(d, prob, r, s1, truth, ids):
    f = lambda dec: macro_f05(to_sets(decide(d, prob, dec, r), s1, r), truth, ids)
    dec = {"t": {k: 0.5 for k in SLICES}, "rescue": 0.0}
    for _ in range(2):  # coordinate descent over slice thresholds, then the rescue threshold
        for k in SLICES:
            best = max(np.arange(0.1, 0.96, 0.025), key=lambda v: f({**dec, "t": {**dec["t"], k: float(v)}}))
            dec["t"][k] = float(best)
        dec["rescue"] = float(max([0.0] + list(np.arange(0.1, 0.7, 0.05)), key=lambda v: f({**dec, "rescue": v})))
    return dec, f(dec)


# ---------------------------------------------------------------- fit / apply

def _fit_crossfit(X, y, fold):
    p = np.zeros(len(y))
    models = []
    for k in (0, 1):
        m = lgb.train(PARAMS2, lgb.Dataset(X[fold != k], y[fold != k]), num_boost_round=ROUNDS2)
        p[fold == k] = m.predict(X[fold == k], num_threads=0)
        models.append(m)
    return p, models


def fit(args):
    s1, r = load_split(args.data, "train", args.work)
    truth, r2s = load_truth(args.data)
    d = pd.read_parquet(f"{args.work}/train_scored.parquet")
    d = d[d.p1.values >= MIN_P1].reset_index(drop=True)
    d = load_side(args.work, "train", d)
    edges = load_edges(args.work, "train")
    y = d.y.values.astype(np.int8)
    ekey = id_bucket(r.entity_id.map(r2s).fillna(r.entity_id).values)[d.q.values]
    is_b = (d.split.values == "B") & (ekey >= STAGE2_B[0]) & (ekey < STAGE2_B[1])  # the scored slice of B
    fold = ekey % 2
    log(f"stage-2 pairs {len(d):,} (B {is_b.sum():,}); ce columns {[c for c in d.columns if c.startswith('ce')]}; "
        f"edges {'yes' if edges is not None else 'no'}")

    # pass 1: context from stage-1 p1
    X1 = stage2_features(d, d.p1.values, s1, r, edges)
    p2a = np.zeros(len(d))
    pb, m1 = _fit_crossfit(X1[is_b], y[is_b], fold[is_b])
    p2a[is_b] = pb
    p2a[~is_b] = np.mean([m.predict(X1[~is_b], num_threads=0) for m in m1], axis=0)
    # pass 2: context recomputed from pass-1 probabilities (collective)
    X2 = stage2_features(d, p2a, s1, r, edges)
    X2["p2a"] = p2a
    p2 = np.zeros(len(d))
    pb, m2 = _fit_crossfit(X2[is_b], y[is_b], fold[is_b])
    p2[is_b] = pb
    p2[~is_b] = np.mean([m.predict(X2[~is_b], num_threads=0) for m in m2], axis=0)
    # no-address specialist
    na = X2.noaddr.values == 1
    p2n = p2.copy()
    mb = is_b & na
    pb, mn = _fit_crossfit(X2[mb], y[mb], fold[mb])
    p2n[mb] = pb
    p2n[~is_b & na] = np.mean([m.predict(X2[~is_b & na], num_threads=0) for m in mn], axis=0)

    # evaluate on held-out H: half A tunes, half B reports
    h_ids = s1.entity_id.values[split_name(id_bucket(s1.entity_id.values)) == "H"]
    half = id_bucket(np.array([i[::-1] for i in h_ids])) < 50
    ids_a, ids_b = h_ids[half], h_ids[~half]
    th = {i: truth[i] for i in h_ids}
    res = {}
    for name, prob in (("stage1", d.p1.values), ("pass1", p2a), ("pass2", p2), ("pass2+noaddr", p2n)):
        dec, fa = tune(d, prob, r, s1, th, ids_a)
        fb = macro_f05(to_sets(decide(d, prob, dec, r), s1, r), th, ids_b)
        res[name] = (fb, dec, fa)
        pr, rc = _pair_pr(decide(d, prob, dec, r), s1, r, th, ids_b)
        log(f"{name:13s} H half A {fa:.4f} | half B {fb:.4f} (pair precision {pr:.4f}, recall {rc:.4f}) | decision {dec}")
    best = max(res, key=lambda k: res[k][2])  # chosen on half A only: half B stays a clean report
    boot = _bootstrap(d, {"stage1": d.p1.values, "pass1": p2a, "pass2": p2, "pass2+noaddr": p2n}[best],
                      res[best][1], r, s1, th, ids_b)
    log(f"chosen: {best} -> H half B {res[best][0]:.4f} (bootstrap 95% CI {boot[0]:.4f}-{boot[1]:.4f})")

    for k, m in enumerate(m1):
        m.save_model(f"{args.work}/s2_pass1_{k}.txt")
    for k, m in enumerate(m2):
        m.save_model(f"{args.work}/s2_pass2_{k}.txt")
    for k, m in enumerate(mn):
        m.save_model(f"{args.work}/s2_noaddr_{k}.txt")
    meta = json.load(open(f"{args.work}/meta.json"))
    meta.update(stage2_choice=best, decision=res[best][1], stage2_features_1=list(X1.columns),
                stage2_features_2=list(X2.columns), h_half_b=res[best][0])
    json.dump(meta, open(f"{args.work}/meta.json", "w"))
    d.assign(p2=p2, p2n=p2n)[["q", "s", "p1", "p2", "p2n", "y", "split"]].to_parquet(f"{args.work}/train_stage2.parquet")


def _pair_pr(a, s1, r, truth, ids):
    """Pair-level precision and recall of the predicted matches over the S1 entities `ids`."""
    pred = to_sets(a, s1, r)
    tp = sum(len(pred.get(i, set()) & truth[i]) for i in ids)
    n_pred = sum(len(pred.get(i, set())) for i in ids)
    n_true = sum(len(truth[i]) for i in ids)
    return tp / max(n_pred, 1), tp / max(n_true, 1)


def _bootstrap(d, prob, dec, r, s1, truth, ids, n=200, seed=0):
    pred = to_sets(decide(d, prob, dec, r), s1, r)
    per = np.array([macro_f05(pred, truth, [i]) for i in ids])
    rng = np.random.default_rng(seed)
    means = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(n)]
    return np.quantile(means, [0.025, 0.975])


def apply(args):
    meta = json.load(open(f"{args.work}/meta.json"))
    s1, r = load_split(args.data, "test", args.work)
    d = pd.read_parquet(f"{args.work}/test_scored.parquet")
    d = d[d.p1.values >= MIN_P1].reset_index(drop=True)
    d = load_side(args.work, "test", d)
    edges = load_edges(args.work, "test")
    load = lambda name: [lgb.Booster(model_file=f"{args.work}/{name}_{k}.txt") for k in (0, 1)]
    choice = meta["stage2_choice"]
    trained_ce = any(c.startswith("ce") for c in meta.get("stage2_features_1", []))
    has_ce = any(c.startswith("ce") for c in d.columns)
    assert trained_ce == has_ce, "stage 2 was trained with/without cross-encoder scores; test must match"
    if choice == "stage1":
        prob = d.p1.values
    else:
        X1 = stage2_features(d, d.p1.values, s1, r, edges)[meta["stage2_features_1"]]
        prob = np.mean([m.predict(X1, num_threads=0) for m in load("s2_pass1")], axis=0)
        if choice != "pass1":
            X2 = stage2_features(d, prob, s1, r, edges)
            X2["p2a"] = prob
            X2 = X2[meta["stage2_features_2"]]
            prob = np.mean([m.predict(X2, num_threads=0) for m in load("s2_pass2")], axis=0)
            if choice == "pass2+noaddr":
                na = X2.noaddr.values == 1
                prob[na] = np.mean([m.predict(X2[na], num_threads=0) for m in load("s2_noaddr")], axis=0)
    a = decide(d, prob, meta["decision"], r)
    os.makedirs(args.out, exist_ok=True)
    write_lists(f"{args.out}/matching_results.tsv", "matched_entity_ids", s1.entity_id.values, to_sets(a, s1, r))
    log(f"wrote {args.out}/matching_results.tsv: {len(a):,} matched records ({choice}, {meta['decision']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--work", default="work")
    ap.add_argument("--out", default="output")
    ap.add_argument("--stage", choices=["fit", "apply"], required=True)
    args = ap.parse_args()
    {"fit": fit, "apply": apply}[args.stage](args)
