"""E35 = ranker v2: same as E31 apply, but trained on ~10x more no-address records: held-out half A PLUS v6.1's
stage-2 training slice (split B, cross-fitted probabilities, not held-out). Scored on held-out half B only.
(E31 apply: (1) leakage re-check on held-out half B — swaps restricted to records NOT used for training;
(2) train the no-address ranker on ALL held-out records (half A + B) and apply it to TEST (US/India no-address records),
exactly as validated: a record's confidence stays max_s p2a, only which S1 receives it can change.
Writes <out>/matching_results.tsv (all countries from v6.1's stage 2 + ranker; France rows are replaced later by merge).
Usage: python e31_apply.py --data dataset --work w61 --src src61 --out out_rk"""
import argparse
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
for k in ("data", "work", "src", "out", "frame_out"):
    ap.add_argument(f"--{k}", required=True)
a = ap.parse_args()
sys.path.insert(0, a.src)
import er  # noqa: E402
import stage2 as S  # noqa: E402

t0 = time.time()
log = lambda *x: print(f"[{time.time() - t0:6.0f}s]", *x, flush=True)
meta = json.load(open(f"{a.work}/meta.json"))
F1 = list(meta["stage2_features_1"])
PARAMS = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[1], learning_rate=0.05, num_leaves=31,
              min_data_in_leaf=100, feature_fraction=0.7, lambda_l2=10, verbose=-1, num_threads=0)


def with_extras(d, split):
    have = set(pq.ParquetFile(f"{a.work}/{split}_scored.parquet").schema_arrow.names)
    extra = [f for f in F1 if f in have and f not in d.columns and f not in ("q", "s")]
    if extra:
        d = d.merge(pd.read_parquet(f"{a.work}/{split}_scored.parquet", columns=["q", "s"] + extra), on=["q", "s"], how="left")
    return d


def rank_feats(d, s1, r, edges):
    X = S.stage2_features(d, d.p1.values, s1, r, edges)[F1].reset_index(drop=True)
    for c in ("p1", "p2a", "p2", "p2n"):
        X[f"x_{c}"] = d[c].values
    g = d.groupby("q").p2a
    X["x_p2a_rank"] = g.rank(ascending=False, method="first").values
    X["x_p2a_gap"] = (g.transform("max") - d.p2a).values
    X["x_ncand"] = g.transform("size").values
    return X


def swap(d, cand, sc, allowed=None):
    p_new = d.p2a.values.copy()
    dd = d[cand].assign(sc=sc[cand], p=d.p2a.values[cand])
    tr = dd.sort_values(["q", "sc"], ascending=[True, False]).drop_duplicates("q")
    tp = dd.sort_values(["q", "p"], ascending=[True, False]).drop_duplicates("q")
    i_new = pd.Series(tr.index.values, index=tr.q.values); i_old = pd.Series(tp.index.values, index=tp.q.values).reindex(i_new.index)
    ch = i_new.values != i_old.values
    if allowed is not None:
        ch &= pd.Series(i_new.index).isin(allowed).values
    n, o = i_new.values[ch], i_old.values[ch]
    p_new[n], p_new[o] = d.p2a.values[o], d.p2a.values[n]
    return p_new, int(ch.sum())


# ---------- train side (held-out H)
s1, r = er.load_split(a.data, "train", a.work)
truth, r2s = er.load_truth(a.data)
d = with_extras(S.load_side(a.work, "train", pd.read_parquet(f"{a.work}/train_stage2.parquet")), "train")
X = rank_feats(d, s1, r, S.load_edges(a.work, "train"))
sid, rid = s1.entity_id.values, r.entity_id.values
h_ids = sid[er.split_name(er.id_bucket(sid)) == "H"]
half = er.id_bucket(np.array([i[::-1] for i in h_ids])) < 50
A_set, B_set = set(h_ids[half]), set(h_ids[~half]); H_all = A_set | B_set
ids_b = h_ids[~half]; th = {i: truth[i] for i in h_ids}
true_s1 = pd.Series(rid[d.q.values]).map(r2s).fillna("").values
cand = (S.slice_of(r, d.q.values) == "noaddr") & (pd.Series(true_s1).isin(H_all).values | pd.Series(sid[d.s.values]).isin(H_all).values) \
       & (X.x_ncand.values >= 2)


def fit(mask):
    pos = set(d.q.values[mask & (d.y.values == 1)])
    mask = mask & pd.Series(d.q.values).isin(pos).values
    dt = d[mask].sort_values("q"); Xt = X.loc[dt.index]
    grp = dt.groupby("q", sort=False).size().values
    gi = (er.id_bucket(rid[dt.groupby("q", sort=False).q.first().values]) % 10 == 0)
    rows_in = np.repeat(gi, grp)
    m = lgb.train(PARAMS, lgb.Dataset(Xt[~rows_in], dt.y.values[~rows_in], group=grp[~gi]), 2000,
                  valid_sets=[lgb.Dataset(Xt[rows_in], dt.y.values[rows_in], group=grp[gi])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    return m, pos


def per_s1(prob):
    P = er.to_sets(S._fast_decider(d, prob, r)(meta["decision"]), s1, r)
    return np.array([er.macro_f05(P, th, [i]) for i in ids_b])


# (1) leakage re-check: train on half A, swap only records NOT in the training set
candB = (S.slice_of(r, d.q.values) == "noaddr") & (d.split.values == "B") & (X.x_ncand.values >= 2)         & ~pd.Series(true_s1).isin(B_set).values & ~pd.Series(sid[d.s.values]).isin(B_set).values   # never touches half B
log(f"training rows: held-out half A {int((cand & pd.Series(true_s1).isin(A_set).values).sum()):,} + split-B slice {int(candB.sum()):,}")
mA, posA = fit((cand & pd.Series(true_s1).isin(A_set).values) | candB)
log(f"ranker v2 trained: {mA.best_iteration} trees")
sc = np.full(len(d), -np.inf); sc[cand] = mA.predict(X[cand], num_threads=0)
allowed = set(d.q.values[cand]) - posA
p_clean, nch = swap(d, cand, sc, allowed)
f0, f1 = per_s1(d.p2a.values), per_s1(p_clean)
cty = s1.set_index("entity_id").country.reindex(ids_b).values
rng = np.random.default_rng(0)
print(f"\n=== leakage re-check: swaps only on records not used for training ({nch:,} records changed) ===")
for c in ("US", "India", "all"):
    mm = np.ones(len(ids_b), bool) if c == "all" else cty == c
    dl = f1[mm] - f0[mm]; B = rng.integers(0, mm.sum(), size=(1000, mm.sum())); bm = dl[B].mean(1)
    print(f"CLEAN {c:5s} {f0[mm].mean():.4f} -> {f1[mm].mean():.4f} | paired {dl.mean():+.5f} "
          f"[{np.quantile(bm, .025):+.5f}, {np.quantile(bm, .975):+.5f}]", flush=True)

# (2) final ranker on all held-out records -> test
mF = mA   # apply exactly the model that passed the leakage-free held-out check (trained on half A)
log(f"test ranker = the validated half-A model: {mA.best_iteration} trees")
del d, X
s1t, rt = er.load_split(a.data, "test", a.work)
dtst = pd.read_parquet(f"{a.work}/test_scored.parquet")
dtst = dtst[dtst.p1.values >= S.MIN_P1].reset_index(drop=True)
dtst = with_extras(S.load_side(a.work, "test", dtst), "test")
edges = S.load_edges(a.work, "test")
load = lambda name: [lgb.Booster(model_file=f"{a.work}/{name}_{k}.txt") for k in (0, 1)]
X1 = S.stage2_features(dtst, dtst.p1.values, s1t, rt, edges)[F1]
dtst["p2a"] = np.mean([m.predict(X1, num_threads=0) for m in load("s2_pass1")], axis=0)
X2 = S.stage2_features(dtst, dtst.p2a.values, s1t, rt, edges); X2["p2a"] = dtst.p2a.values
X2 = X2[meta["stage2_features_2"]]
dtst["p2"] = np.mean([m.predict(X2, num_threads=0) for m in load("s2_pass2")], axis=0)
dtst["p2n"] = dtst["p2"].values.copy()
na = X2.noaddr.values == 1
dtst.loc[na, "p2n"] = np.mean([m.predict(X2[na], num_threads=0) for m in load("s2_noaddr")], axis=0)
log(f"test stage-2 probabilities recomputed ({meta.get('stage2_choice')} is used for the decision)")
d = dtst
Xt = rank_feats(d, s1t, rt, edges)
ctry = rt.country.values[d.q.values]
candt = (S.slice_of(rt, d.q.values) == "noaddr") & (Xt.x_ncand.values >= 2) & np.isin(ctry, ["US", "India"])
sct = np.full(len(d), -np.inf); sct[candt] = mF.predict(Xt[candt], num_threads=0)
p_new, nch = swap(d, candt, sct)
log(f"test: US/India no-address records whose chosen S1 changed: {nch:,}")
base_asg = S._fast_decider(d, d.p2a.values, rt)(meta["decision"])
new_asg = S._fast_decider(d, p_new, rt)(meta["decision"])
os.makedirs(a.out, exist_ok=True)
er.write_lists(f"{a.out}/matching_results_v61_recomputed.tsv", "matched_entity_ids", s1t.entity_id.values, er.to_sets(base_asg, s1t, rt))
er.write_lists(f"{a.out}/matching_results.tsv", "matched_entity_ids", s1t.entity_id.values, er.to_sets(new_asg, s1t, rt))
# ---- X12 export (added): every US/India candidate row, same column names as heldout_stage2.parquet
usin = np.isin(ctry, ["US", "India"])
fx = Xt[usin].reset_index(drop=True)
fx.insert(0, "record_id", rt.entity_id.values[d.q.values[usin]])
fx.insert(1, "s1_id", s1t.entity_id.values[d.s.values[usin]])
fx.insert(2, "slice", np.asarray(S.slice_of(rt, d.q.values[usin])))
fx["ranker_v2"] = np.where(candt, sct, np.nan)[usin]
fx.to_parquet(a.frame_out)
log(f"frame written: {len(fx):,} rows, {fx.record_id.nunique():,} records, slices {fx.slice.value_counts().to_dict()}")
log("E31_APPLY DONE")
