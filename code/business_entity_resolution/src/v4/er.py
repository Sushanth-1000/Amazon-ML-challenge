"""Business entity resolution v4: clean -> shortlist -> prefilter -> features -> stage-1 LightGBM.

Structural facts from the training labels that shape this pipeline:
  * every S2/S3 record matches at most ONE S1 entity (~26% match none), so each record queries the S1 index and is
    finally assigned to at most one S1;
  * matches never cross countries, so the shortlist runs per country (country is an open set of labels).

v4 data design ("train like test"): ALL train records go through exactly the same forward shortlist as test.
Train S1 entities are split by a hash of the entity (a record follows its true S1, or its own id if unmatched):
  H (2%)  held-out evaluation only (half A tunes decisions, half B is reported)
  A (49%) trains stage-1 model A, the cross-encoder and the record matcher
  B (49%) trains stage-1 model B and stage 2
Stage 1 is cross-fitted A<->B (each half scored by the model of the other half); H and test get the average.
Only the training folder and its labels are used for learning; test is read-only.

Stages (from student_resource/):
    python src/er.py --stage train      # shortlist + prefilter + features + stage 1 on train -> work/train_scored.parquet
    python src/er.py --stage test       # same on test -> work/test_scored.parquet + output/candidate_pairs.tsv
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

from normalize import NORM_COLS, core_name, normalize_record

LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=0)
PRE_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=0)
PRE_KEEP = 0.9995     # prefilter keeps this share of true pairs (v3: 0.999) ...
PRE_MAX_SHARE = 0.08  # ... relaxed to 99.9% recall if that keeps more than this share; hard cap at this share
PRE_ROUNDS = 300      # fixed: early stopping on log-loss stopped at ~30 trees (true pairs < 1%) and lost 1.6% of them
GENERIC_DF = 0.005    # a name word in more than 0.5% of the split's names counts as generic (unsupervised)
SPLIT_H, SPLIT_A = 2, 51   # entity hash bucket: [0,2) -> H, [2,51) -> A, [51,100) -> B
STAGE2_B = (51, 61)        # stage 2 trains on this slice of B (~10% of train S1, ~100x v3's stage-2 data)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)


def id_bucket(ids):
    """Stable 0-99 bucket per entity id (for reproducible splits)."""
    return np.fromiter((zlib.crc32(i.encode()) % 100 for i in ids), dtype=np.int16, count=len(ids))


def split_name(bucket):
    return np.where(bucket < SPLIT_H, "H", np.where(bucket < SPLIT_A, "A", "B"))


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


def _add_core(s1, r):
    """Distinctive name = name minus generic words; generic = document frequency > GENERIC_DF on this split's
    names (unsupervised, computed the same way on train and test)."""
    df = Counter()
    for col in (s1.name.values, r.name.values):
        for x in col:
            df.update(set(x.split()))
    n = len(s1) + len(r)
    generic = {w for w, c in df.items() if c / n > GENERIC_DF}
    for d in (s1, r):
        d["name_core"] = [core_name(x, generic) for x in d.name.values]
        d["core_ns"] = d.name_core.str.replace(" ", "", regex=False)
    return generic


def load_split(data, split, work):
    """Returns (s1, r): normalized Source-1 table and Source-2+3 table. Cached as parquet."""
    paths = [f"{work}/{split}_s1.parquet", f"{work}/{split}_r.parquet"]
    if all(map(os.path.exists, paths)):
        return tuple(pd.read_parquet(p) for p in paths)
    s1 = _normalize(read_tsv(f"{data}/{split}/{split}_source1.tsv"))
    r = _normalize(pd.concat([read_tsv(f"{data}/{split}/{split}_source{i}.tsv") for i in (2, 3)],
                             ignore_index=True))
    _add_core(s1, r)
    s1.to_parquet(paths[0]); r.to_parquet(paths[1])
    return s1, r


# ---------------------------------------------------------------- shortlist

# channel -> (field, top-k, TfidfVectorizer kwargs). v3: name/addr/both/skel = 20/20/30/20, recall 98.3%.
CHANNELS = {
    "name": ("name_ns", 20, dict(analyzer="char", ngram_range=(3, 3), max_df=0.005)),
    "addr": ("addr", 20, dict(analyzer="word", token_pattern=r"\S+", max_df=0.002)),
    "both": ("both", 30, dict(analyzer="word", token_pattern=r"\S+", ngram_range=(1, 2), max_df=0.005)),
    "phon": ("name_ph", 20, dict(analyzer="char_wb", ngram_range=(2, 4), max_df=0.05)),  # tiny alphabet: lighter pruning
    "core": ("core_ns", 20, dict(analyzer="char", ngram_range=(3, 3), max_df=0.005)),
}
# v3 audit: 59% of blocking misses were records with an EMPTY address (only the name can find the entity and
# common names have many look-alikes) -> deep search; records whose name finds nothing similar get a fallback
# channel without common-gram pruning and a lower cosine floor, plus a deeper address search.
DEEP_NOADDR = (("name", "phon"), 100)
FALLBACK_K, FALLBACK_ADDR_K, FALLBACK_BELOW, FALLBACK_FLOOR = 20, 40, 0.3, 0.02
S1_REVERSE_K = 20  # S1 entities left without any candidate search the records instead


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


def _topn(Q, ST, rows, k, floor):
    if not len(rows):
        return pd.DataFrame({"q": [], "s": []}, dtype=np.int64)
    C = sp_matmul_topn(Q[rows], ST, top_n=k, threshold=floor, n_threads=-1).tocoo()
    return pd.DataFrame({"q": rows[C.row], "s": C.col})


def block(s1, r, dense=None):
    """Candidate pairs (q = row in r, s = row in s1): union of all channels, per country."""
    for d in (s1, r):
        d["both"] = d.name + " " + d.addr
    parts = []
    for c in r.country.unique():
        si = np.flatnonzero(s1.country.values == c)
        qi = np.flatnonzero(r.country.values == c)
        if not len(si) or not len(qi):
            continue
        mats, found = {}, []
        allq = np.arange(len(qi))
        noaddr = np.flatnonzero(r.addr.values[qi] == "")
        best_name = np.zeros(len(qi), np.float32)
        for ch, (field, k, kw) in CHANNELS.items():
            S, Q = _vectorize(s1[field].iloc[si], r[field].iloc[qi], kw)
            mats[ch] = (S, Q)
            if S is None:
                continue
            ST = S.T.tocsr()
            hit = _topn(Q, ST, allq, k, 0.05)
            found.append(hit)
            if ch == "name" and len(hit):
                cos = _rowdot(Q, S, hit.q.values, hit.s.values)
                np.maximum.at(best_name, hit.q.values, cos)
            if ch in DEEP_NOADDR[0]:
                found.append(_topn(Q, ST, noaddr, DEEP_NOADDR[1], 0.05))
        weak = np.flatnonzero(best_name < FALLBACK_BELOW)
        if len(weak):
            S, Q = _vectorize(s1.name_ns.iloc[si], r.name_ns.iloc[qi], dict(analyzer="char", ngram_range=(3, 3)))
            if S is not None:
                found.append(_topn(Q, S.T.tocsr(), weak, FALLBACK_K, FALLBACK_FLOOR))
            S, Q = mats["addr"]
            if S is not None:
                found.append(_topn(Q, S.T.tocsr(), weak, FALLBACK_ADDR_K, 0.05))
        if dense is not None and len(dense):
            pos_q = np.full(len(r), -1); pos_q[qi] = np.arange(len(qi))
            pos_s = np.full(len(s1), -1); pos_s[si] = np.arange(len(si))
            dq, ds = pos_q[dense.q.values], pos_s[dense.s.values]
            ok = (dq >= 0) & (ds >= 0)
            found.append(pd.DataFrame({"q": dq[ok], "s": ds[ok]}))
        pc = pd.concat(found).drop_duplicates()
        lonely = np.setdiff1d(np.arange(len(si)), pc.s.values)  # S1 with no candidate at all: search the records
        if len(lonely):
            for ch in ("both", "name", "phon"):
                S, Q = mats[ch]
                if S is None:
                    continue
                C = sp_matmul_topn(S[lonely], Q.T.tocsr(), top_n=S1_REVERSE_K, threshold=FALLBACK_FLOOR,
                                   n_threads=-1).tocoo()
                pc = pd.concat([pc, pd.DataFrame({"q": C.col, "s": lonely[C.row]})])
            pc = pc.drop_duplicates()
        for ch, (S, Q) in mats.items():  # exact score in every channel for every candidate
            pc["sc_" + ch] = _rowdot(Q, S, pc.q.values, pc.s.values)
        pc["q"], pc["s"] = qi[pc.q.values], si[pc.s.values]
        parts.append(pc)
        log(f"  blocked {c}: {len(qi):,} queries x {len(si):,} S1 -> {len(pc):,} pairs "
            f"(no-address deep {len(noaddr):,}, weak-name fallback {len(weak):,}, S1 reverse {len(lonely):,})")
    p = pd.concat(parts, ignore_index=True)
    p["q"], p["s"] = p.q.astype(np.int32), p.s.astype(np.int32)
    if dense is not None:
        p = p.merge(dense[["q", "s", "sc_dense"]], on=["q", "s"], how="left")
        p["sc_dense"] = p.sc_dense.astype(np.float32)
    return p.sort_values(["q", "s"], ignore_index=True)


def load_dense(work, split, s1, r):
    """Pairs from an optional embedding channel (src/dense.py), mapped to row indices; None if not computed."""
    path = f"{work}/{split}_dense.parquet"
    if not os.path.exists(path):
        return None
    d = pd.read_parquet(path)
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
    return sc_cols(p) + [c for c in p.columns if c == "n_cand" or c.endswith(("_rank", "_gap", "_margin"))]


def fit_prefilter(p, y, hold):
    X = p[cheap_cols(p)]
    dtr = lgb.Dataset(X[~hold], y[~hold])
    m = lgb.train(PRE_PARAMS, dtr, num_boost_round=PRE_ROUNDS)
    ph = m.predict(X[hold], num_threads=0)
    pos = np.sort(ph[y[hold] == 1])
    at = lambda keep: float(pos[int((1 - keep) * len(pos))]) if len(pos) else 0.0
    tau_budget = float(np.quantile(ph, 1 - PRE_MAX_SHARE))  # a few unrecognisable true pairs must not force ~0
    tau = at(PRE_KEEP)
    if tau < tau_budget:  # 99.95% would keep too many pairs: fall back to v3's 99.9%, then to the hard cap
        tau = max(at(0.999), tau_budget)
    log(f"prefilter: tau={tau:.2e} keeps {np.mean(ph >= tau):.3f} of held-out pairs, {np.mean(pos >= tau):.5f} of true")
    return m, tau


def _competition_feats(p):
    """How each candidate's channel scores compare to the other candidates of the same query (p sorted by q)."""
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
        order = np.lexsort((-v, q))
        sv = v[order]
        new = np.r_[True, (sv[1:] != sv[:-1]) | (grp[1:] != grp[:-1])] if n else np.zeros(0, bool)
        first_eq = np.maximum.accumulate(np.where(new, pos, 0))
        rank = np.empty(n, np.float32)
        rank[order] = first_eq - starts[grp] + 1
        rank = np.where(v > 0, rank, counts[grp]).astype(np.float32)  # a zero score is last, not a tied "first"
        best = sv[starts]
        second = np.where(counts > 1, sv[np.minimum(starts + 1, n - 1)], 0.0)
        f[c + "_rank"] = rank
        f[c + "_gap"] = (v - best[grp]).astype(np.float32)
        f[c + "_margin"] = (best - second)[grp].astype(np.float32)
    return f


# ---------------------------------------------------------------- label-derived tables (cross-fitted)

def fit_token_logodds(rn, sn, y, tset, min_count=30):
    """Decoy words: per added name token, log P(added | look-alike non-match) - log P(added | true match)."""
    pos, neg = Counter(), Counter()
    hard = (y == 0) & (tset >= 80)
    for a, b, is_pos, is_hard in zip(rn, sn, y == 1, hard):
        if is_pos or is_hard:
            (pos if is_pos else neg).update(set(a.split()) - set(b.split()))
    n_pos, n_neg = int((y == 1).sum()) + 1, int(hard.sum()) + 1
    return {w: float(np.log((neg[w] + 1) / n_neg) - np.log((pos[w] + 1) / n_pos))
            for w in pos.keys() | neg.keys() if pos[w] + neg[w] >= min_count}


def fit_locality_pairs(rt, st, y, min_count=20):
    """Locality substitutions learned from TRAIN true pairs: (record tail word, S1 tail word) pairs that differ but
    co-occur in true matches far more than in non-matches (e.g. millcreek <-> salt lake). Returns log-odds."""
    pos, neg = Counter(), Counter()
    for a, b, lab in zip(rt, st, y):
        ta, tb = set(a.split()), set(b.split())
        da, db = ta - tb, tb - ta
        if not da or not db or len(da) > 3 or len(db) > 3:
            continue
        (pos if lab == 1 else neg).update((x, z) for x in da for z in db)
    n_pos, n_neg = sum(pos.values()) + 1, sum(neg.values()) + 1
    return {f"{x}|{z}": float(np.log((pos[(x, z)] + 1) / n_pos) - np.log((neg[(x, z)] + 1) / n_neg))
            for (x, z) in pos if pos[(x, z)] >= min_count}


def fit_idf(s1, r):
    """Name-token IDF on the split's own (unlabelled) text: rare shared words signal a true match."""
    df = Counter()
    for col in (s1.name.values, r.name.values):
        for x in col:
            df.update(set(x.split()))
    n = len(s1) + len(r)
    return {"idf": {w: math.log(n / c) for w, c in df.items() if c > 1}, "default": math.log(n)}


# ---------------------------------------------------------------- pair features

_TOK = {}


def _tok_init(lo, idf, idf_default, loc):
    _TOK.update(lo=lo, idf=idf, d=idf_default, loc=loc)


def _token_chunk(args):
    rn, sn, rc, sc, rt, st = args
    lo, idf, d, loc = _TOK["lo"], _TOK["idf"], _TOK["d"], _TOK["loc"]
    out = np.zeros((len(rn), 15), np.float32)
    for i in range(len(rn)):
        tx, ty = set(rn[i].split()), set(sn[i].split())
        extra, shared, missing = tx - ty, tx & ty, ty - tx
        out[i, 0], out[i, 1] = len(extra), len(missing)
        if extra:
            v = [lo.get(w, 0.0) for w in extra]
            out[i, 2], out[i, 3] = sum(v), max(v)
            out[i, 6] = max(idf.get(w, d) for w in extra)
        out[i, 4] = max((idf.get(w, d) for w in shared), default=0.0)
        u = len(tx | ty)
        out[i, 5] = len(shared) / u if u else 0.0
        out[i, 7] = len(shared) / len(tx) if tx else 0.0          # containment: record inside S1 name
        out[i, 8] = len(shared) / len(ty) if ty else 0.0          # S1 inside record name
        out[i, 9] = float(sorted(tx) == sorted(ty) and bool(tx))  # same words, any order
        cx, cy = set(rc[i].split()), set(sc[i].split())            # distinctive (non-generic) words
        out[i, 10] = len(cx & cy)
        out[i, 11] = len(cy - cx)
        out[i, 12] = len(cx - cy)
        out[i, 13] = max((idf.get(w, d) for w in (cy - cx)), default=0.0)  # rarest distinctive S1 word missing
        ta, tb = set(rt[i].split()) - set(st[i].split()), set(st[i].split()) - set(rt[i].split())
        out[i, 14] = max((loc.get(f"{x}|{z}", 0.0) for x in ta for z in tb), default=0.0) if ta and tb else 0.0
    return out


TOKEN_FEATS = ["n_extra_tok", "n_missing_tok", "extra_tok_logodds_sum", "extra_tok_logodds_max", "rare_shared_max",
               "n_jacc", "rare_extra_max", "contain_r_in_s", "contain_s_in_r", "same_words", "core_shared",
               "core_s_only", "core_r_only", "core_missing_idf", "loc_sub_logodds"]


def _token_feats(f, cols, tables, chunk=200_000):
    n = len(cols[0])
    args = [tuple(c[i:i + chunk] for c in cols) for i in range(0, n, chunk)]
    if len(args) > 1:
        with Pool(initializer=_tok_init, initargs=tables) as pool:
            out = np.vstack(pool.map(_token_chunk, args))
    else:
        _tok_init(*tables)
        out = _token_chunk(args[0]) if args else np.zeros((0, len(TOKEN_FEATS)), np.float32)
    for j, c in enumerate(TOKEN_FEATS):
        f[c] = out[:, j]


def _first_int(xs):
    return np.array([int(x.split(" ", 1)[0][:9]) if x else -1 for x in xs], dtype=np.int64)


def _number_feats(f, rnum, snum):
    a, b = _first_int(rnum), _first_int(snum)
    miss = (a < 0) | (b < 0)
    delta = np.abs(a - b).astype(np.float32)
    f["num_first_delta"] = np.where(miss, np.nan, delta).astype(np.float32)
    f["num_first_rel"] = np.where(miss, np.nan, delta / np.maximum(np.maximum(a, b), 1)).astype(np.float32)
    f["num_small_shift"] = np.where(miss, -1, (delta >= 1) & (delta <= 20)).astype(np.float32)
    f["num_r_first_in_s"] = np.array([bool(x) and x.split(" ", 1)[0] in y.split() for x, y in zip(rnum, snum)],
                                     dtype=np.float32)
    # house number truncated / prefix ("582" vs "5828") and the closest aligned number
    pref, mind = np.full(len(a), -1, np.float32), np.full(len(a), np.nan, np.float32)
    for i, (x, y) in enumerate(zip(rnum, snum)):
        if x and y:
            fx, fy = x.split(" ", 1)[0], y.split(" ", 1)[0]
            pref[i] = float(fx != fy and (fy.startswith(fx) or fx.startswith(fy)))
            xs = [int(t[:9]) for t in x.split()]
            ys = [int(t[:9]) for t in y.split()]
            mind[i] = min(abs(u - v) for u in xs for v in ys)
    f["num_prefix"], f["num_min_delta"] = pref, mind


def _set_eq3(a, b):
    """-1 missing, 1 equal sets, 0.5 overlapping, 0 disjoint."""
    out = np.full(len(a), -1, np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        if x and y:
            sx, sy = set(x.split()), set(y.split())
            out[i] = 1.0 if sx == sy else (0.5 if sx & sy else 0.0)
    return out


def split_context(s1, r):
    """Unsupervised per-split counts used as features (same computation on train and test)."""
    key = lambda d, col: d.country + "|" + d[col]
    return {"s1_name_freq": key(s1, "name").map(key(s1, "name").value_counts()).values.astype(np.float32),
            "r_name_freq": key(r, "name").map(key(r, "name").value_counts()).values.astype(np.float32),
            "s1_addr_freq": key(s1, "addr").map(key(s1, "addr").value_counts()).values.astype(np.float32)}


def features(p, s1, r, lo, idf, loc, ctx):
    """p: prefiltered pairs with channel scores + competition features. lo/loc: cross-fitted label tables."""
    q, s = p.q.values, p.s.values
    f = p[cheap_cols(p)].reset_index(drop=True).copy()

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
    sim("name_ph", fuzz.ratio, "n_ph_ratio")
    rc, sc = sim("name_core", fuzz.token_set_ratio, "core_tset")
    ra, sa = sim("addr", fuzz.token_set_ratio, "a_tset")
    sim("addr", fuzz.token_sort_ratio, "a_tsort")
    sim("addr", fuzz.ratio, "a_ratio")
    rt, st = sim("tail", fuzz.token_set_ratio, "tail_tset")
    rnum, snum = sim("nums", fuzz.token_set_ratio, "num_tset")
    first = lambda xs: np.array([x.split(" ", 1)[0] for x in xs], dtype=object)
    fr, fs = first(rnum), first(snum)
    f["num_first_eq"] = np.where((fr == "") | (fs == ""), -1, (fr == fs).astype(np.int8)).astype(np.float32)
    _number_feats(f, rnum, snum)
    _token_feats(f, (rn, sn, rc, sc, rt, st), (lo, idf["idf"], idf["default"], loc))
    f["suffix_eq"] = _set_eq3(*pair("suffix"))
    f["pcode_eq"] = _set_eq3(*pair("pcode"))
    f["codes_eq"] = _set_eq3(*pair("codes"))
    f["store_eq"] = _set_eq3(*pair("store_no"))
    f["suffix_differs_only"] = ((f.n_tset.values >= 100) & (f.suffix_eq.values == 0)).astype(np.float32)
    # ambiguity: how many of this record's candidates have (almost) the same name, and name frequencies
    exact = (f.n_tset.values >= 95).astype(np.float32)
    n_exact = pd.Series(exact).groupby(q).transform("sum").values
    f["n_exact_cands"] = n_exact
    f["is_unique_exact"] = ((exact == 1) & (n_exact == 1)).astype(np.float32)
    f["s1_name_freq"] = ctx["s1_name_freq"][s]
    f["r_name_freq"] = ctx["r_name_freq"][q]
    f["s1_addr_freq"] = ctx["s1_addr_freq"][s]  # co-location: S1 sharing this exact address
    lens = lambda xs: np.fromiter(map(len, xs), np.float32, len(xs))
    f["r_name_len"], f["s_name_len"] = lens(rn), lens(sn)
    f["r_addr_len"], f["s_addr_len"] = lens(ra), lens(sa)
    f["name_len_ratio"] = np.minimum(f.r_name_len, f.s_name_len) / np.maximum(np.maximum(f.r_name_len, f.s_name_len), 1)
    f["r_nonlatin"] = r.nonlatin.values[q].astype(np.float32)
    f["r_is_s3"] = r.entity_id.str.startswith("S3").values[q].astype(np.float32)
    return f


# ---------------------------------------------------------------- assignment + metric

MAX_PER_S1 = 11  # no train S1 entity has more than 11 matched records


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


def train_lgb(X, y, hold, params=LGB_PARAMS, w=None):
    dtr = lgb.Dataset(X[~hold], y[~hold], weight=None if w is None else w[~hold])
    dho = lgb.Dataset(X[hold], y[hold], reference=dtr)
    return lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dho],
                     callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)])


# ---------------------------------------------------------------- stages

def _label_tables(rn, sn, rt, st, y, tset):
    return fit_token_logodds(rn, sn, y, tset), fit_locality_pairs(rt, st, y)


def train_stage(args):
    s1, r = load_split(args.data, "train", args.work)
    truth, r2s = load_truth(args.data)
    log(f"train: {len(s1):,} S1, {len(r):,} S2+S3 (all records, forward shortlist like test)")
    p = cached_block(s1, r, f"{args.work}/train_pairs.parquet", load_dense(args.work, "train", s1, r))
    true_s = r.entity_id.map(r2s)
    y = (s1.entity_id.values[p.s.values] == true_s.values[p.q.values]).astype(np.int8)
    n_true = int(true_s.notna().sum())
    log(f"pairs {len(p):,} ({len(p) / len(r):.1f}/record) | shortlist recall {y.sum() / n_true:.4f}")

    rkey = true_s.fillna(r.entity_id).values             # a record follows its true S1 (or itself)
    rsplit = split_name(id_bucket(rkey))
    split = rsplit[p.q.values]
    is_a = split == "A"
    hold = is_a & (id_bucket(r.entity_id.values[p.q.values]) % 10 == 0)
    pre, tau = fit_prefilter(p[is_a].reset_index(drop=True), y[is_a], hold[is_a])
    pre.save_model(f"{args.work}/prefilter.txt")
    p0 = pre.predict(p[cheap_cols(p)], num_threads=0)
    keep = p0 >= tau
    lost = {sp: 1 - y[keep & (split == sp)].sum() / max(y[split == sp].sum(), 1) for sp in "HAB"}
    p, y, split = p[keep].reset_index(drop=True), y[keep], split[keep]
    log(f"after prefilter {len(p):,} pairs; true pairs lost by prefilter: {lost}")

    idf, ctx = fit_idf(s1, r), split_context(s1, r)
    rn, sn = r.name.values[p.q.values], s1.name.values[p.s.values]
    rt, st = r["tail"].values[p.q.values], s1["tail"].values[p.s.values]
    tset = cpdist(list(rn), list(sn), scorer=fuzz.token_set_ratio, workers=-1)
    tables = {}
    for part, src in (("A", "B"), ("B", "A"), ("H", "AB")):  # rows of `part` get tables fitted on `src`
        m = np.isin(split, list(src))
        tables[part] = _label_tables(rn[m], sn[m], rt[m], st[m], y[m], tset[m])
    json.dump({"lo": tables["H"][0], "loc": tables["H"][1]}, open(f"{args.work}/label_tables.json", "w"))
    json.dump(idf, open(f"{args.work}/train_idf.json", "w"))
    X = pd.concat([features(p[split == sp], s1, r, tables[sp][0], idf, tables[sp][1], ctx).set_index(np.flatnonzero(split == sp))
                   for sp in "ABH"]).sort_index()
    log(f"features {X.shape}")

    models, p1 = {}, np.zeros(len(p), np.float32)
    for part, other in (("A", "B"), ("B", "A")):  # all training pairs: no negative subsampling (accuracy first)
        m = split == part
        inner = id_bucket(r.entity_id.values[p.q.values[m]]) % 10 == 0
        models[part] = train_lgb(X[m], y[m], inner)
        models[part].save_model(f"{args.work}/stage1_{part}.txt")
        p1[split == other] = models[part].predict(X[split == other], num_threads=0)
    mh = split == "H"
    p1[mh] = np.mean([mdl.predict(X[mh], num_threads=0) for mdl in models.values()], axis=0)
    out = p[["q", "s"]].assign(p0=p0[keep].astype(np.float32), p1=p1, y=y, split=split)
    out = pd.concat([out, X[["n_exact_cands", "is_unique_exact", "s1_name_freq", "s1_addr_freq", "codes_eq",
                             "num_small_shift", "n_tset", "a_tset"]].reset_index(drop=True)], axis=1)
    out.to_parquet(f"{args.work}/train_scored.parquet")
    imp = pd.Series(models["A"].feature_importance("gain"), index=X.columns).sort_values(ascending=False)
    log("top features:\n" + imp.head(25).round(0).to_string())
    json.dump({"prefilter_tau": tau, "features": list(X.columns), "splits": [SPLIT_H, SPLIT_A]},
              open(f"{args.work}/meta.json", "w"))
    for sp in "HAB":
        ids = s1.entity_id.values[split_name(id_bucket(s1.entity_id.values)) == sp]
        log(f"split {sp}: {len(ids):,} S1, {int((split == sp).sum()):,} pairs")


def testblock_stage(args):
    """Test cleaning + shortlist only (no model needed): runs on a second machine in parallel with training."""
    s1, r = load_split(args.data, "test", args.work)
    p = cached_block(s1, r, f"{args.work}/test_pairs.parquet", load_dense(args.work, "test", s1, r))
    log(f"test pairs {len(p):,} ({len(p) / len(r):.1f}/record)")


def test_stage(args):
    meta = json.load(open(f"{args.work}/meta.json"))
    models = [lgb.Booster(model_file=f"{args.work}/stage1_{k}.txt") for k in "AB"]
    lt = json.load(open(f"{args.work}/label_tables.json"))
    s1, r = load_split(args.data, "test", args.work)
    log(f"test: {len(s1):,} S1, {len(r):,} S2+S3")
    path = f"{args.work}/test_scored.parquet"
    if not os.path.exists(path):
        p = cached_block(s1, r, f"{args.work}/test_pairs.parquet", load_dense(args.work, "test", s1, r))
        pre = lgb.Booster(model_file=f"{args.work}/prefilter.txt")
        p0 = pre.predict(p[cheap_cols(p)], num_threads=0)
        keep = p0 >= meta["prefilter_tau"]
        p = p[keep].reset_index(drop=True).assign(p0=p0[keep].astype(np.float32))
        log(f"test pairs after prefilter {len(p):,}")
        idf, ctx = fit_idf(s1, r), split_context(s1, r)
        p1 = np.empty(len(p), np.float32)
        extra = []
        n_q = int(p.q.max()) + 1
        bounds = np.searchsorted(p.q.values, list(range(0, n_q, args.chunk_queries)) + [n_q])
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b > a:
                X = features(p.iloc[a:b].reset_index(drop=True), s1, r, lt["lo"], idf, lt["loc"], ctx)
                p1[a:b] = np.mean([m.predict(X[meta["features"]], num_threads=0) for m in models], axis=0)
                extra.append(X[["n_exact_cands", "is_unique_exact", "s1_name_freq", "s1_addr_freq", "codes_eq",
                                "num_small_shift", "n_tset", "a_tset"]])
                log(f"  scored {b:,}/{len(p):,}")
        out = pd.concat([p[["q", "s", "p0"]].assign(p1=p1), pd.concat(extra, ignore_index=True)], axis=1)
        out.to_parquet(path)
    p = pd.read_parquet(path)
    os.makedirs(args.out, exist_ok=True)
    write_lists(f"{args.out}/candidate_pairs.tsv", "candidate_entity_ids", s1.entity_id.values, to_sets(p, s1, r))
    log(f"wrote {args.out}/candidate_pairs.tsv ({len(p):,} pairs)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--out", default="output")
    ap.add_argument("--work", default="work")
    ap.add_argument("--chunk-queries", type=int, default=500000)
    ap.add_argument("--stage", choices=["train", "testblock", "test"], required=True)
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)
    {"train": train_stage, "testblock": testblock_stage, "test": test_stage}[args.stage](args)
