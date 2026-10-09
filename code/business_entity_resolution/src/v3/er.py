"""Business entity resolution: normalize -> block (TF-IDF top-k) -> prefilter -> LightGBM pair scorer -> assign.

Key structural facts from the training labels that shape this pipeline:
  * every S2/S3 record matches at most ONE S1 entity (and ~26% match none), so we query from the
    S2/S3 side: each record looks up its top-k most similar S1 entities, the model scores those
    pairs, and the record is assigned to its single best S1 entity if the score clears a threshold;
  * matches never cross countries, so blocking runs per country (country is an open set of labels).

v3: bigger top-k per channel (blocking recall was the ceiling), optional dense (embedding) channel,
a cheap prefilter model whose survivors are candidate_pairs.tsv, new features (legal-form agreement,
city/state tail, postcode, rare shared / extra name tokens), 2-fold stage-1 models with cross-fitted
token log-odds (gives out-of-fold probabilities on the training pairs for the cross-encoder).

Stages (run from student_resource/):
    python src/er.py --stage train      # block + prefilter + features + 2 fold models; val + OOF predictions
    python src/decode.py                # decision rule on validation
    python src/er.py --stage test       # block + prefilter + stage-1 scores on test, candidate_pairs.tsv
    python src/ce.py ...                # cross-encoder: export / train / score (GPU)
    python src/stage2.py                # sibling + cross-encoder re-scoring, validated
    python src/er.py --stage final      # decision -> matching_results.tsv
"""
import argparse
import json
import math
import os
import time
import zlib
from collections import Counter
from multiprocessing import Pool

import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from normalize import NORM_COLS, normalize_record

LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=0)
PRE_PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=0)
PRE_KEEP = 0.999  # prefilter keeps this share of true pairs (measured on held-out training queries)
N_FOLDS = 2


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)


def id_bucket(ids):
    """Stable 0-99 bucket per entity id (for reproducible splits)."""
    return np.fromiter((zlib.crc32(i.encode()) % 100 for i in ids), dtype=np.int16, count=len(ids))


# ---------------------------------------------------------------- loading + normalization

def _norm_chunk(rows):
    return [normalize_record(*r) for r in rows]


def _normalize(df):
    rows = list(zip(df.business_name, df.business_address, df.country))
    chunks = [rows[i:i + 50000] for i in range(0, len(rows), 50000)]
    with Pool() as pool:
        out = [x for c in pool.map(_norm_chunk, chunks) for x in c]
    norm = pd.DataFrame(out, columns=NORM_COLS, index=df.index)
    raw = df[["business_name", "business_address"]].rename(columns={"business_name": "raw_name",
                                                                      "business_address": "raw_addr"})
    return pd.concat([df[["entity_id", "country"]], norm, raw], axis=1)


def load_split(data, split, work):
    """Returns (s1, r): normalized Source-1 table and Source-2+3 table. Cached as parquet."""
    paths = [f"{work}/{split}_s1.parquet", f"{work}/{split}_r.parquet"]
    if all(map(os.path.exists, paths)):
        return tuple(pd.read_parquet(p) for p in paths)
    s1 = _normalize(read_tsv(f"{data}/{split}/{split}_source1.tsv"))
    r = _normalize(pd.concat([read_tsv(f"{data}/{split}/{split}_source{i}.tsv") for i in (2, 3)],
                             ignore_index=True))
    s1.to_parquet(paths[0]); r.to_parquet(paths[1])
    return s1, r


# ---------------------------------------------------------------- blocking

# channel -> (field, top-k, only native-script queries?, TfidfVectorizer kwargs).
# v2 (k = 5/5/10/5, skeleton for native-script queries only) reached 97.1% recall, ceiling 0.987.
CHANNELS = {
    "name": ("name_ns", 20, False, dict(analyzer="char", ngram_range=(3, 3), max_df=0.005)),
    "addr": ("addr", 20, False, dict(analyzer="word", token_pattern=r"\S+", max_df=0.002)),
    "both": ("both", 30, False, dict(analyzer="word", token_pattern=r"\S+", ngram_range=(1, 2), max_df=0.005)),
    "skel": ("name_sk", 20, False, dict(analyzer="char", ngram_range=(3, 3), max_df=0.005)),
}


def _vectorize(s_text, q_text, kw):
    vec = TfidfVectorizer(dtype=np.float32, sublinear_tf=True, **kw)
    try:
        vec.fit(pd.concat([s_text, q_text]))
    except ValueError:  # vocabulary empty after pruning (tiny country)
        return None, None
    return vec.transform(s_text), vec.transform(q_text)


def _rowdot(Q, S, qi, si, chunk=2_000_000):
    """Cosine of TF-IDF rows Q[qi] . S[si] for aligned index arrays (chunked to bound memory)."""
    out = np.zeros(len(qi), np.float32)
    if Q is None:
        return out
    for a in range(0, len(qi), chunk):
        b = a + chunk
        out[a:b] = np.asarray(Q[qi[a:b]].multiply(S[si[a:b]]).sum(axis=1)).ravel()
    return out


def block(s1, r, dense=None):
    """Candidate pairs (q = row in r, s = row in s1): union of top-k per channel, per country.
    dense: optional DataFrame (q, s, sc_dense) of extra pairs from the embedding channel (row indices)."""
    for d in (s1, r):
        d["both"] = d.name + " " + d.addr
    parts = []
    for c in r.country.unique():
        si = np.flatnonzero(s1.country.values == c)
        qi = np.flatnonzero(r.country.values == c)
        if not len(si) or not len(qi):
            continue
        mats, found = {}, []
        for ch, (field, k, native_only, kw) in CHANNELS.items():
            S, Q = _vectorize(s1[field].iloc[si], r[field].iloc[qi], kw)
            mats[ch] = (S, Q)
            if S is None:
                continue
            rows = np.flatnonzero(r.nonlatin.values[qi]) if native_only else np.arange(len(qi))
            C = sp_matmul_topn(Q[rows], S.T.tocsr(), top_n=k, threshold=0.05, n_threads=-1).tocoo()
            found.append(pd.DataFrame({"q": rows[C.row], "s": C.col}))
        if dense is not None and len(dense):
            pos_q = np.full(len(r), -1); pos_q[qi] = np.arange(len(qi))
            pos_s = np.full(len(s1), -1); pos_s[si] = np.arange(len(si))
            dq, ds = pos_q[dense.q.values], pos_s[dense.s.values]
            ok = (dq >= 0) & (ds >= 0)
            found.append(pd.DataFrame({"q": dq[ok], "s": ds[ok]}))
        if not found:
            continue
        pc = pd.concat(found).drop_duplicates()
        for ch, (S, Q) in mats.items():  # exact score in every channel for every candidate
            pc["sc_" + ch] = _rowdot(Q, S, pc.q.values, pc.s.values)
        pc["q"], pc["s"] = qi[pc.q.values], si[pc.s.values]
        parts.append(pc)
        log(f"  blocked {c}: {len(qi):,} queries x {len(si):,} S1 -> {len(pc):,} pairs")
    p = pd.concat(parts, ignore_index=True)
    p["q"], p["s"] = p.q.astype(np.int32), p.s.astype(np.int32)
    if dense is not None:
        p = p.merge(dense[["q", "s", "sc_dense"]], on=["q", "s"], how="left")
        p["sc_dense"] = p.sc_dense.astype(np.float32)  # NaN = not found by the dense channel
    return p.sort_values(["q", "s"], ignore_index=True)


def reverse_queries(s1, r, s_mask, k):
    """Rows of r likely to have an S1 from s_mask among their candidates: the top-k most similar
    queries of each masked S1, per channel. Lets validation block only those queries instead of all."""
    for d in (s1, r):
        d["both"] = d.name + " " + d.addr
    found = []
    for c in r.country.unique():
        si = np.flatnonzero((s1.country.values == c) & s_mask)
        qi = np.flatnonzero(r.country.values == c)
        if not len(si) or not len(qi):
            continue
        for ch, (field, _, native_only, kw) in CHANNELS.items():
            qq = qi[r.nonlatin.values[qi]] if native_only else qi
            if not len(qq):
                continue
            S, Q = _vectorize(s1[field].iloc[si], r[field].iloc[qq], kw)
            if S is None:
                continue
            C = sp_matmul_topn(S, Q.T.tocsr(), top_n=k, threshold=0.05, n_threads=-1)
            found.append(qq[np.unique(C.indices)])
    # a query can still rank a val S1 in its top-k while missing that S1's reverse top-k
    # (hub names with many near-duplicates); raise --reverse-k if val looks optimistic.
    return np.unique(np.concatenate(found))


def load_dense(work, split, s1, r):
    """Pairs from the embedding channel (src/dense.py), mapped to row indices; None if not computed."""
    path = f"{work}/{split}_dense.parquet"
    if not os.path.exists(path):
        return None
    d = pd.read_parquet(path)  # columns: r_id, s1_id, cos
    qmap = pd.Series(np.arange(len(r)), index=r.entity_id.values)
    smap = pd.Series(np.arange(len(s1)), index=s1.entity_id.values)
    d = d[d.r_id.isin(qmap.index) & d.s1_id.isin(smap.index)]
    return pd.DataFrame({"q": qmap[d.r_id].values, "s": smap[d.s1_id].values, "sc_dense": d.cos.values})


def cached_block(s1, r, path, dense=None):
    if os.path.exists(path):
        return pd.read_parquet(path)
    p = block(s1, r, dense)
    p = pd.concat([p, _competition_feats(p)], axis=1)
    p.to_parquet(path)
    return p


# ---------------------------------------------------------------- prefilter

def sc_cols(p):
    """Raw channel scores (not the derived _rank/_gap/_margin columns)."""
    return [c for c in p.columns if c.startswith("sc_") and not c.endswith(("_rank", "_gap", "_margin"))]


def cheap_cols(p):
    """Prefilter inputs: channel scores + the record-side competition features computed at blocking."""
    return sc_cols(p) + [c for c in p.columns if c == "n_cand" or c.endswith(("_rank", "_gap", "_margin"))]


def fit_prefilter(p, y, hold):
    X = p[cheap_cols(p)]
    dtr = lgb.Dataset(X[~hold], y[~hold])
    m = lgb.train(PRE_PARAMS, dtr, num_boost_round=1000, valid_sets=[lgb.Dataset(X[hold], y[hold], reference=dtr)],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
    ph = m.predict(X[hold], num_threads=0)
    pos = np.sort(ph[y[hold] == 1])
    tau = float(pos[int((1 - PRE_KEEP) * len(pos))]) if len(pos) else 0.0
    log(f"prefilter: tau={tau:.2e} keeps {np.mean(ph >= tau):.3f} of held-out pairs, "
        f"{np.mean(pos >= tau):.4f} of their true pairs")
    return m, tau


def apply_prefilter(p, m, tau):
    p0 = m.predict(p[cheap_cols(p)], num_threads=0)
    keep = p0 >= tau
    return p[keep].reset_index(drop=True).assign(p0=p0[keep].astype(np.float32))


# ---------------------------------------------------------------- pair features

def _competition_feats(p):
    """How each candidate's channel scores compare to the other candidates of the same query.
    p must be sorted by q. Vectorized (numpy) because test has hundreds of millions of pairs."""
    q = p.q.values
    n = len(q)
    assert n == 0 or (q[1:] >= q[:-1]).all(), "pairs must be sorted by q"
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]]) if n else np.zeros(0, int)
    counts = np.diff(np.r_[starts, n])
    grp = np.repeat(np.arange(len(starts)), counts)
    f = pd.DataFrame(index=p.index)
    f["n_cand"] = counts[grp].astype(np.float32)
    pos = np.arange(n)
    for c in sc_cols(p):
        v = np.nan_to_num(p[c].values.astype(np.float32), nan=0.0)
        order = np.lexsort((-v, q))  # by query, then score descending
        sv = v[order]
        new = np.r_[True, (sv[1:] != sv[:-1]) | (grp[1:] != grp[:-1])] if n else np.zeros(0, bool)
        first_eq = np.maximum.accumulate(np.where(new, pos, 0))  # ties share the smallest rank ("min")
        rank = np.empty(n, np.float32)
        rank[order] = first_eq - starts[grp] + 1
        best = sv[starts]
        second = np.where(counts > 1, sv[np.minimum(starts + 1, n - 1)], 0.0)
        f[c + "_rank"] = rank
        f[c + "_gap"] = (v - best[grp]).astype(np.float32)  # 0 for the best candidate
        f[c + "_margin"] = (best - second)[grp].astype(np.float32)
    return f


def _competition_feats_pandas(p):  # reference implementation (v2), used by the self-test only
    f = pd.DataFrame(index=p.index)
    g = p.groupby("q")
    f["n_cand"] = g.s.transform("size").astype(np.float32)
    for c in sc_cols(p):
        v = p[c].fillna(0)
        best = v.groupby(p.q).transform("max")
        second = v.to_frame().assign(q=p.q.values).sort_values(["q", c], ascending=[True, False]).groupby("q")[c].nth(1)
        second = pd.Series(second.values, index=p.q.loc[second.index].values)
        f[c + "_rank"] = v.groupby(p.q).rank(ascending=False, method="min").astype(np.float32)
        f[c + "_gap"] = (v - best).astype(np.float32)
        f[c + "_margin"] = (best - p.q.map(second).fillna(0)).astype(np.float32)
    return f


# Decoys are synthetic near-clones of real S1 businesses: same street with the house number shifted
# by a few units, and often one added business word ("holdings", "partners", "downtown", ...).
# Token log-odds: per added name token, log P(added | lookalike non-match) - log P(added | true match).
def fit_token_logodds(rn, sn, y, tset, min_count=30):
    pos, neg = Counter(), Counter()
    hard = (y == 0) & (tset >= 80)  # lookalike non-matches, not random candidates
    for a, b, is_pos, is_hard in zip(rn, sn, y == 1, hard):
        if is_pos or is_hard:
            (pos if is_pos else neg).update(set(a.split()) - set(b.split()))
    n_pos, n_neg = int((y == 1).sum()) + 1, int(hard.sum()) + 1
    return {w: float(np.log((neg[w] + 1) / n_neg) - np.log((pos[w] + 1) / n_pos))
            for w in pos.keys() | neg.keys() if pos[w] + neg[w] >= min_count}


def fit_idf(s1, r):
    """Name-token IDF on the split's own (unlabeled) text: rare shared words signal a true match."""
    df = Counter()
    for col in (s1.name.values, r.name.values):
        for x in col:
            df.update(set(x.split()))
    n = len(s1) + len(r)
    # words seen once can never be shared; they get the default (= rarest possible) IDF
    return {"idf": {w: math.log(n / c) for w, c in df.items() if c > 1}, "default": math.log(n)}


def _first_int(xs):
    return np.array([int(x.split(" ", 1)[0][:9]) if x else -1 for x in xs], dtype=np.int64)


_TOK = {}  # token log-odds / IDF shared with worker processes (set once per pool, not per task)


def _tok_init(lo, idf, idf_default):
    _TOK.update(lo=lo, idf=idf, d=idf_default)


def _token_chunk(args):
    rn, sn = args
    lo, idf, d = _TOK["lo"], _TOK["idf"], _TOK["d"]
    n = len(rn)
    out = np.zeros((n, 7), np.float32)
    for i, (x, y) in enumerate(zip(rn, sn)):
        tx, ty = set(x.split()), set(y.split())
        extra, shared = tx - ty, tx & ty
        out[i, 0], out[i, 1] = len(extra), len(ty - tx)
        if extra:
            v = [lo.get(w, 0.0) for w in extra]
            out[i, 2], out[i, 3] = sum(v), max(v)
            out[i, 6] = max(idf.get(w, d) for w in extra)
        out[i, 4] = max((idf.get(w, d) for w in shared), default=0.0)
        u = len(tx | ty)
        out[i, 5] = len(shared) / u if u else 0.0
    return out


def _token_feats(f, rn, sn, lo, idf, chunk=200_000):
    args = [(rn[i:i + chunk], sn[i:i + chunk]) for i in range(0, len(rn), chunk)]
    init = (lo, idf["idf"], idf["default"])
    if len(args) > 1:
        with Pool(initializer=_tok_init, initargs=init) as pool:
            out = np.vstack(pool.map(_token_chunk, args))
    else:
        _tok_init(*init)
        out = _token_chunk(args[0]) if args else np.zeros((0, 7), np.float32)
    for j, c in enumerate(["n_extra_tok", "n_missing_tok", "extra_tok_logodds_sum", "extra_tok_logodds_max",
                           "rare_shared_max", "n_jacc", "rare_extra_max"]):
        f[c] = out[:, j]


def _decoy_num_feats(f, rnum, snum):
    a, b = _first_int(rnum), _first_int(snum)
    miss = (a < 0) | (b < 0)
    delta = np.abs(a - b).astype(np.float32)
    f["num_first_delta"] = np.where(miss, np.nan, delta).astype(np.float32)
    f["num_first_rel"] = np.where(miss, np.nan, delta / np.maximum(np.maximum(a, b), 1)).astype(np.float32)
    f["num_small_shift"] = np.where(miss, -1, (delta >= 1) & (delta <= 20)).astype(np.float32)
    f["num_r_first_in_s"] = np.array([bool(x) and x.split(" ", 1)[0] in y.split() for x, y in zip(rnum, snum)],
                                     dtype=np.float32)


def _eq3(a, b, overlap=False):
    """-1 if either side missing, else 1 if equal (or overlapping token sets), 0 if different."""
    out = np.full(len(a), -1, np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        if x and y:
            out[i] = bool(set(x.split()) & set(y.split())) if overlap else (x == y)
    return out


def features(p, s1, r, lo, idf):
    """p: prefiltered pairs carrying channel scores + competition features; lo: token log-odds; idf: token IDF."""
    q, s = p.q.values, p.s.values
    f = p[cheap_cols(p)].reset_index(drop=True).copy()  # p0 (prefilter score) is not a feature: in-sample on train

    def pair(col):
        return list(r[col].values[q]), list(s1[col].values[s])

    def sim(col, scorer, name):
        a, b = pair(col)
        f[name] = cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)
        return a, b

    rn, sn = sim("name", fuzz.token_set_ratio, "n_tset")
    sim("name", fuzz.token_sort_ratio, "n_tsort")
    sim("name", fuzz.partial_ratio, "n_partial")
    sim("name_ns", fuzz.ratio, "n_ns_ratio")
    sim("name_ns", JaroWinkler.normalized_similarity, "n_ns_jw")
    sim("name_sk", fuzz.ratio, "n_sk_ratio")
    sim("name_sk", fuzz.token_set_ratio, "n_sk_tset")
    ra, sa = sim("addr", fuzz.token_set_ratio, "a_tset")
    sim("addr", fuzz.token_sort_ratio, "a_tsort")
    sim("addr", fuzz.ratio, "a_ratio")
    sim("tail", fuzz.token_set_ratio, "tail_tset")
    rnum, snum = sim("nums", fuzz.token_set_ratio, "num_tset")
    first = lambda xs: np.array([x.split(" ", 1)[0] for x in xs], dtype=object)
    fr, fs = first(rnum), first(snum)
    f["num_first_eq"] = np.where((fr == "") | (fs == ""), -1, (fr == fs).astype(np.int8)).astype(np.float32)
    _decoy_num_feats(f, rnum, snum)
    _token_feats(f, rn, sn, lo, idf)
    f["suffix_eq"] = _eq3(*pair("suffix"), overlap=True)
    f["pcode_eq"] = _eq3(*pair("pcode"))
    lens = lambda xs: np.fromiter(map(len, xs), np.float32, len(xs))
    f["r_name_len"], f["s_name_len"] = lens(rn), lens(sn)
    f["r_addr_len"], f["s_addr_len"] = lens(ra), lens(sa)
    f["name_len_ratio"] = np.minimum(f.r_name_len, f.s_name_len) / np.maximum(np.maximum(f.r_name_len, f.s_name_len), 1)
    f["r_nonlatin"] = r.nonlatin.values[q].astype(np.float32)
    f["r_is_s3"] = r.entity_id.str.startswith("S3").values[q].astype(np.float32)
    return f


# ---------------------------------------------------------------- assignment + metric

MAX_PER_S1 = 11  # no train S1 entity has more than 11 matched records


def decode(p, prob, dec, noaddr):
    """Final matches from pair probabilities. dec = {"kind": "threshold", "t"} | {"kind": "margin", "t", "m"}
    | {"kind": "expf", "floor"}, each optionally with "t_na": a stricter cut for records without an address
    (noaddr, aligned with p), and "cap": max records per S1 entity (default MAX_PER_S1, lowest prob dropped).
    margin: record -> best S1 only if prob >= t and it beats the record's 2nd-best S1 by >= m.
    expf:   record -> best S1 (prob >= floor); then per S1 keep the top-k of its records maximizing the
            plug-in expected F0.5 = 1.25*sum(top-k p) / (0.25*E[#true] + k), or none when P(no true match)
            is larger. This optimizes the metric per entity, including singletons."""
    d = pd.DataFrame({"q": p.q.values, "s": p.s.values, "prob": prob, "na": noaddr})
    d = d.sort_values(["q", "prob"], ascending=[True, False])
    first = ~d.q.duplicated()
    best = d[first]
    cut = lambda key: np.where(best.na.values, dec.get("t_na", dec[key]), dec[key])
    if dec["kind"] == "threshold":
        out = best[best.prob.values >= cut("t")]
    elif dec["kind"] == "margin":
        second = d[~first].drop_duplicates("q").set_index("q").prob
        gap = best.prob.values - best.q.map(second).fillna(0).values
        out = best[(best.prob.values >= cut("t")) & (gap >= dec["m"])]
    else:
        e_true = d.groupby("s").prob.sum()
        p_none = np.exp(np.log1p(-d.prob.clip(upper=1 - 1e-6)).groupby(d.s).sum())
        b = best[best.prob.values >= cut("floor")].sort_values(["s", "prob"], ascending=[True, False])
        k = b.groupby("s").cumcount().values + 1
        ef = 1.25 * b.groupby("s").prob.cumsum().values / (0.25 * b.s.map(e_true).values + k)
        b = b.assign(k=k, ef=ef)
        top = b.loc[b.groupby("s").ef.idxmax(), ["s", "k", "ef"]].set_index("s")
        keep = (b.k.values <= b.s.map(top.k).values) & (b.s.map(top.ef).values > b.s.map(p_none).values)
        out = b[keep]
    cap = int(dec.get("cap", MAX_PER_S1))
    out = out.sort_values(["s", "prob"], ascending=[True, False])
    out = out[out.groupby("s").cumcount().values < cap]
    return out[["q", "s", "prob"]]


def macro_f05(pred, truth, ids):
    """Macro F0.5 over S1 ids; pred/truth: dict s1_id -> set of matched ids. Singletons score 1 iff empty."""
    tot = 0.0
    for i in ids:
        P, T = pred.get(i, set()), truth.get(i, set())
        if not T:
            tot += not P
            continue
        tp = len(P & T)
        if tp:
            prec, rec = tp / len(P), tp / len(T)
            tot += 1.25 * prec * rec / (0.25 * prec + rec)
    return tot / len(ids)


def to_sets(a, s1, r):
    out = {}
    for s_id, r_id in zip(s1.entity_id.values[a.s.values], r.entity_id.values[a.q.values]):
        out.setdefault(s_id, set()).add(r_id)
    return out


def write_lists(path, col, s1_ids, sets):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"source1_entity_id\t{col}\n")
        for i in s1_ids:
            fh.write(f"{i}\t{','.join(sorted(sets.get(i, ())))}\n")


def load_truth(data):
    gt = read_tsv(f"{data}/train/train_ground_truth.tsv")
    truth = {i: set(x for x in ms.split(",") if x) for i, ms in zip(gt.source1_entity_id, gt.matched_entity_ids)}
    r2s = {m: s for s, ms in truth.items() for m in ms}
    return truth, r2s


def fold_of(r, r2s, q):
    """Fold of a training pair = hash of the record's true S1 (or of the record if unmatched), so all
    records of one entity (and all candidates of one record) fall in the same fold."""
    key = r.entity_id.map(r2s).fillna(r.entity_id).values
    return (id_bucket(key) % N_FOLDS)[q]


def train_lgb(X, y, hold, params=LGB_PARAMS):
    dtr = lgb.Dataset(X[~hold], y[~hold])
    dho = lgb.Dataset(X[hold], y[hold], reference=dtr)
    return lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dho],
                     callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)])


def predict_models(models, X):
    return np.mean([m.predict(X, num_threads=0) for m in models], axis=0)


# ---------------------------------------------------------------- main

def train_and_validate(args):
    s1, r = load_split(args.data, "train", args.work)
    log(f"train: {len(s1):,} S1, {len(r):,} S2+S3")
    # Block only the train queries we use: a hash sample for fitting + the queries that can reach the
    # validation S1 set V (found by reverse search).
    s_val = id_bucket(s1.entity_id.values) < args.val_pct
    keep_path = f"{args.work}/train_keep.npy"
    if os.path.exists(keep_path):
        keep = np.load(keep_path)
    else:
        q_rev = reverse_queries(s1, r, s_val, args.reverse_k)
        keep = np.union1d(np.flatnonzero(id_bucket(r.entity_id.values) < args.train_pct), q_rev)
        np.save(keep_path, keep)
    r = r.iloc[keep].reset_index(drop=True)
    log(f"train queries kept: {len(r):,}")
    p = cached_block(s1, r, f"{args.work}/train_pairs.parquet", load_dense(args.work, "train", s1, r))

    truth, r2s = load_truth(args.data)
    true_s = r.entity_id.map(r2s)
    y = (s1.entity_id.values[p.s.values] == true_s.values[p.q.values]).astype(np.int8)
    n_true = true_s.notna().sum()
    log(f"pairs {len(p):,} ({len(p) / len(r):.1f}/query) | blocking recall {y.sum() / n_true:.4f}")

    # Validation S1 set V; val queries = every kept record with a candidate in V (all its candidates kept,
    # so the argmax assignment is exact). Training uses the remaining kept queries, pairs outside V.
    q_val = np.unique(p.q.values[s_val[p.s.values]])
    is_val = np.isin(p.q.values, q_val)
    is_tr = ~is_val & ~s_val[p.s.values]
    ptr, pva = p[is_tr].reset_index(drop=True), p[is_val].reset_index(drop=True)
    ytr, yva = y[is_tr], y[is_val]
    del p
    log(f"val: {s_val.sum():,} S1 / {len(pva):,} pairs | train: {len(ptr):,} pairs")

    hold = id_bucket(r.entity_id.values[ptr.q.values]) < 10  # by query
    pre, tau = fit_prefilter(ptr, ytr, hold)
    pre.save_model(f"{args.work}/prefilter.txt")
    k_tr = pre.predict(ptr[cheap_cols(ptr)], num_threads=0) >= tau
    ptr = ptr[k_tr].reset_index(drop=True).assign(p0=pre.predict(ptr[k_tr][cheap_cols(ptr)], num_threads=0))
    ytr = ytr[k_tr]
    k_va = pre.predict(pva[cheap_cols(pva)], num_threads=0) >= tau
    pva_all_y = yva
    pva = pva[k_va].reset_index(drop=True).assign(p0=pre.predict(pva[k_va][cheap_cols(pva)], num_threads=0))
    yva = yva[k_va]
    log(f"after prefilter: train {len(ptr):,} pairs, val {len(pva):,} pairs "
        f"(val true pairs kept {yva.sum() / max(pva_all_y.sum(), 1):.4f})")

    idf = fit_idf(s1, r)
    rn, sn = list(r.name.values[ptr.q.values]), list(s1.name.values[ptr.s.values])
    tset = cpdist(rn, sn, scorer=fuzz.token_set_ratio, workers=-1)
    fold = fold_of(r, r2s, ptr.q.values)
    # cross-fitted token log-odds: rows of fold k get log-odds fitted on the other folds
    Xtr_parts = []
    for k in range(N_FOLDS):
        mk = fold == k
        lo_k = fit_token_logodds([x for x, m in zip(rn, ~mk) if m], [x for x, m in zip(sn, ~mk) if m],
                                 ytr[~mk], tset[~mk])
        Xtr_parts.append(features(ptr[mk], s1, r, lo_k, idf).set_index(np.flatnonzero(mk)))
    Xtr = pd.concat(Xtr_parts).sort_index()
    lo = fit_token_logodds(rn, sn, ytr, tset)  # all training pairs: used for validation and test
    json.dump(lo, open(f"{args.work}/token_logodds.json", "w"))
    json.dump(idf, open(f"{args.work}/train_idf.json", "w"))
    Xva = features(pva, s1, r, lo, idf)
    log("features done", Xtr.shape)

    models, oof = [], np.zeros(len(ptr))
    for k in range(N_FOLDS):
        mk = fold == k
        inner = id_bucket(r.entity_id.values[ptr.q.values[~mk]]) < 10
        m = train_lgb(Xtr[~mk], ytr[~mk], inner)
        oof[mk] = m.predict(Xtr[mk], num_threads=0)
        m.save_model(f"{args.work}/model_f{k}.txt")
        models.append(m)
    prob = predict_models(models, Xva)
    pd.DataFrame({"q": ptr.q.values, "s": ptr.s.values, "p1": oof.astype(np.float32), "y": ytr, "fold": fold}
                 ).to_parquet(f"{args.work}/train_oof.parquet")
    pva.assign(prob=prob.astype(np.float32), y=yva)[["q", "s", "p0", "prob", "y"]].to_parquet(
        f"{args.work}/val_pred.parquet")

    val_ids = s1.entity_id.values[s_val]
    truth = {i: truth[i] for i in val_ids}
    noaddr = r.addr.values[pva.q.values] == ""
    best_t, best_f = 0.5, -1
    for t in np.arange(0.05, 0.96, 0.05):
        f05 = macro_f05(to_sets(decode(pva, prob, {"kind": "threshold", "t": t}, noaddr), s1, r), truth, val_ids)
        if f05 > best_f:
            best_t, best_f = float(t), f05
    ceiling = macro_f05(to_sets(pva[yva == 1], s1, r), truth, val_ids)
    log(f"best t={best_t:.2f}  val F0.5={best_f:.4f}  (perfect-matcher ceiling given candidates: {ceiling:.4f})")
    imp = pd.Series(models[0].feature_importance("gain"), index=Xtr.columns).sort_values(ascending=False)
    log("top features:\n" + imp.head(20).round(0).to_string())
    with open(f"{args.work}/meta.json", "w") as fh:
        json.dump({"threshold": best_t, "val_f05": best_f, "ceiling": ceiling, "prefilter_tau": tau,
                   "n_folds": N_FOLDS, "features": list(Xtr.columns),
                   "decoder": {"kind": "threshold", "t": best_t}}, fh)


def load_models(work):
    meta = json.load(open(f"{work}/meta.json"))
    return meta, [lgb.Booster(model_file=f"{work}/model_f{k}.txt") for k in range(meta["n_folds"])]


def block_test(args):
    """Test normalization + blocking only (no model needed): can run while the training stage runs."""
    s1, r = load_split(args.data, "test", args.work)
    p = cached_block(s1, r, f"{args.work}/test_pairs.parquet", load_dense(args.work, "test", s1, r))
    log(f"test pairs {len(p):,} ({len(p) / len(r):.1f}/query)")


def predict_test(args):
    """Blocking + prefilter + stage-1 scores on test; writes candidate_pairs.tsv and work/test_pred.parquet."""
    meta, models = load_models(args.work)
    lo = json.load(open(f"{args.work}/token_logodds.json"))
    s1, r = load_split(args.data, "test", args.work)
    log(f"test: {len(s1):,} S1, {len(r):,} S2+S3")
    pred_path = f"{args.work}/test_pred.parquet"
    if not os.path.exists(pred_path):
        p = cached_block(s1, r, f"{args.work}/test_pairs.parquet", load_dense(args.work, "test", s1, r))
        log(f"test pairs {len(p):,}")
        pre = lgb.Booster(model_file=f"{args.work}/prefilter.txt")
        p = apply_prefilter(p, pre, meta["prefilter_tau"])
        log(f"after prefilter {len(p):,}")
        idf = fit_idf(s1, r)  # unlabeled test text
        prob = np.empty(len(p), np.float32)
        n_q = int(p.q.max()) + 1
        bounds = np.searchsorted(p.q.values, list(range(0, n_q, args.chunk_queries)) + [n_q])
        for a, b in zip(bounds[:-1], bounds[1:]):  # pairs are sorted by q: a query never straddles chunks
            if b > a:
                X = features(p.iloc[a:b].reset_index(drop=True), s1, r, lo, idf)[meta["features"]]
                prob[a:b] = predict_models(models, X)
                log(f"  scored {b:,}/{len(p):,}")
        p[["q", "s", "p0"]].assign(prob=prob).to_parquet(pred_path)
    p = pd.read_parquet(pred_path)
    os.makedirs(args.out, exist_ok=True)
    write_lists(f"{args.out}/candidate_pairs.tsv", "candidate_entity_ids", s1.entity_id.values, to_sets(p, s1, r))
    log(f"wrote {args.out}/candidate_pairs.tsv ({len(p):,} pairs)")


def finalize_test(args):
    """Stage-2 re-scoring (if adopted on validation) + decision -> matching_results.tsv."""
    meta = json.load(open(f"{args.work}/meta.json"))
    s1, r = load_split(args.data, "test", args.work)
    p = pd.read_parquet(f"{args.work}/test_pred.parquet")
    prob = p.prob.values
    if meta.get("stage2"):
        from stage2 import MIN_P1, stage2_features, load_ce
        keep = prob >= MIN_P1
        p, prob = p[keep].reset_index(drop=True), prob[keep]
        ce = load_ce(args.work, "test", p) if meta.get("use_ce") else None
        m2 = lgb.Booster(model_file=f"{args.work}/model2.txt")
        prob = m2.predict(stage2_features(p.q.values, p.s.values, prob, s1, r, ce)[meta["stage2_features"]],
                          num_threads=0)
        log(f"stage 2 re-scored {len(p):,} pairs")
    dec = meta["decoder"]
    a = decode(p, prob, dec, r.addr.values[p.q.values] == "")
    os.makedirs(args.out, exist_ok=True)
    write_lists(f"{args.out}/matching_results.tsv", "matched_entity_ids", s1.entity_id.values, to_sets(a, s1, r))
    log(f"wrote {args.out}/matching_results.tsv : {len(a):,} matched records, decoder {dec}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--out", default="output")
    ap.add_argument("--work", default="work")
    ap.add_argument("--val-pct", type=int, default=1, help="percent of train S1 entities used for validation")
    ap.add_argument("--train-pct", type=int, default=10, help="percent of train S2/S3 queries (id hash) to fit on")
    ap.add_argument("--reverse-k", type=int, default=60, help="queries kept per validation S1 per channel")
    ap.add_argument("--chunk-queries", type=int, default=500000)
    ap.add_argument("--stage", choices=["train", "testblock", "test", "final"], required=True)
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)
    {"train": train_and_validate, "testblock": block_test, "test": predict_test, "final": finalize_test}[args.stage](args)
