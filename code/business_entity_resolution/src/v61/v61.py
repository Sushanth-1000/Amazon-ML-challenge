"""v6.1 CPU stages. Every stage writes its outputs to --work, so a failure re-runs only that stage.
Recipe (v6/V6_1.md): v4 candidates UNION bge-m3 top-20 (E13) -> v4 prefilter -> v4's 74 features + bge-m3 similarity
(E15: mm_cos, mm_cos_gap, mm_cos_rank) -> stage 1 = reg (E1) x 3 seeds (E22) with French-style copies of the US
training pairs (E14) -> the rest of v4 unchanged (cross-encoder scores with v4's model, record matcher, stage 2).
Stages:
  union   --split train|test : v4 pairs (--v4) + {split}_dense.parquet -> channel scores for the new pairs, per-record
                               competition features for affected records, v4 prefilter -> {split}_pairs61.parquet,
                               {split}_qs.npy (pairs for the GPU cosine step)
  feats   --split train      : label tables (cross-fitted as v4), 77 features, French-style copies (US, splits A/B)
                               -> train_X.parquet, train_Xc.parquet, label_tables.json, meta.json
  feats   --split test       : 77 features with label tables from train (H tables = fitted on A+B, as v4)
  stage1  --part A|B         : reg x 3 seeds on part rows + their copies -> stage1_{P}_s{k}.txt, p1 of the other part/H
  finish                     : train_scored.parquet (v4 format)
  score                      : test p1 = mean of all stage-1 models -> test_scored.parquet + candidate_pairs.tsv
"""
import argparse
import glob
import json
import math
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import er  # noqa: E402
from er import log  # noqa: E402
from frstyle import fr_addr, fr_name  # noqa: E402
from normalize import core_name, normalize_record  # noqa: E402

REG = dict(num_leaves=31, min_data_in_leaf=1000, feature_fraction=0.5, lambda_l2=10.0)   # E1 'reg'
SEEDS = (0, 1, 2)                                                                          # E22
COSF = ["mm_cos", "mm_cos_gap", "mm_cos_rank"]                                             # E15
EXTRA = ["n_exact_cands", "is_unique_exact", "s1_name_freq", "s1_addr_freq", "codes_eq", "num_small_shift",
         "n_tset", "a_tset"]                                                               # v4 train/test_scored extras
_G = {}


def key(q, s):
    return (q.astype(np.int64) << 22) | s.astype(np.int64)     # S1 rows < 4.19M in both splits


def _chan_worker(task):
    c, ch = task
    s1, r = _G["s1"], _G["r"]
    field, _, kw = er.CHANNELS[ch]
    S, Q = er._vectorize(s1[field].iloc[_G["si"][c]], r[field].iloc[_G["qi"][c]], kw)
    return c, ch, er._rowdot(Q, S, _G["lq"][c], _G["ls"][c])


def cos_feats(q, cos):
    """E15's definitions, per record over its candidate list (pairs sorted by q)."""
    cos = cos.astype(np.float32)
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]]); counts = np.diff(np.r_[starts, len(q)])
    grp = np.repeat(np.arange(len(starts)), counts)
    best = np.maximum.reduceat(cos, starts)[grp]
    order = np.lexsort((-cos, q)); rank = np.empty(len(cos), np.float32)
    rank[order] = np.arange(len(cos)) - np.repeat(starts, counts)[np.argsort(order)] + 1
    return pd.DataFrame({"mm_cos": cos, "mm_cos_gap": cos - best, "mm_cos_rank": rank})


# ---------------------------------------------------------------- union
def union(args):
    sp, W = args.split, args.work
    s1, r = er.load_split(args.data, sp, W)
    d = er.load_dense(W, sp, s1, r)
    qs = pd.read_parquet(f"{args.v4}/{sp}_pairs.parquet", columns=["q", "s"])
    kp = key(qs.q.values, qs.s.values)
    assert (kp[1:] > kp[:-1]).all(), "v4 pairs must be sorted by (q, s)"
    kd = key(d.q.values, d.s.values)
    pos = np.minimum(np.searchsorted(kp, kd), len(kp) - 1)
    dn = d.loc[kp[pos] != kd, ["q", "s"]].drop_duplicates().reset_index(drop=True)
    n_v4 = len(kp); del qs, kp, kd, pos
    log(f"{sp}: v4 pairs {n_v4:,} | dense {len(d):,} | new pairs {len(dn):,} for {dn.q.nunique():,} records")
    if args.smoke:
        dn = dn[dn.q.values % 50 == 0].reset_index(drop=True)
    for x in (s1, r):
        x["both"] = x.name + " " + x.addr                       # as er.block
    cn = r.country.values[dn.q.values]
    si, qi, lq, ls = {}, {}, {}, {}
    for c in np.unique(cn):
        si[c], qi[c] = np.flatnonzero(s1.country.values == c), np.flatnonzero(r.country.values == c)
        pq = np.full(len(r), -1, np.int64); pq[qi[c]] = np.arange(len(qi[c]))
        ps = np.full(len(s1), -1, np.int64); ps[si[c]] = np.arange(len(si[c]))
        m = cn == c
        lq[c], ls[c] = pq[dn.q.values[m]], ps[dn.s.values[m]]
        assert (ls[c] >= 0).all(), "dense pair across countries"
    _G.update(s1=s1, r=r, si=si, qi=qi, lq=lq, ls=ls)
    tasks = [(c, ch) for c in lq for ch in er.CHANNELS]
    for ch in er.CHANNELS:
        dn["sc_" + ch] = np.float32(0)
    with ProcessPoolExecutor(max_workers=min(len(tasks), args.workers)) as ex:
        for c, ch, v in ex.map(_chan_worker, tasks):
            dn.loc[cn == c, "sc_" + ch] = v
            log(f"  channel {ch} scored for {len(v):,} new {c} pairs")
    _G.clear()
    pre = lgb.Booster(model_file=args.prefilter)
    tau = json.load(open(args.v4meta))["prefilter_tau"]
    fn = pre.feature_name()
    p = pd.read_parquet(f"{args.v4}/{sp}_pairs.parquet")
    if args.smoke:
        p = p[p.q.values % 50 == 0].reset_index(drop=True)
    aff = np.zeros(len(r), bool); aff[dn.q.values] = True
    m_aff = aff[p.q.values]
    p0 = np.concatenate([pre.predict(p[fn].iloc[a:a + 50_000_000], num_threads=0) for a in range(0, len(p), 50_000_000)])
    rest = p.loc[~m_aff & (p0 >= tau)].assign(p0=p0[~m_aff & (p0 >= tau)].astype(np.float32))
    pa = pd.concat([p.loc[m_aff, ["q", "s"] + er.sc_cols(p)], dn], ignore_index=True).sort_values(["q", "s"], ignore_index=True)
    del p, p0
    pa = pd.concat([pa, er._competition_feats(pa)], axis=1)
    pa0 = pre.predict(pa[fn], num_threads=0)
    pa = pa.loc[pa0 >= tau].assign(p0=pa0[pa0 >= tau].astype(np.float32))
    out = pd.concat([rest, pa[rest.columns]], ignore_index=True).sort_values(["q", "s"], ignore_index=True)
    out.to_parquet(f"{W}/{sp}_pairs61.parquet")
    np.save(f"{W}/{sp}_qs.npy", np.stack([out.q.values.astype(np.int32), out.s.values.astype(np.int32)]))
    msg = f"{sp}: after v4 prefilter {len(out):,} pairs ({len(pa):,} from affected records)"
    if sp == "train":
        truth, r2s = er.load_truth(args.data)
        ts = r.entity_id.map(r2s)
        y = s1.entity_id.values[out.s.values] == ts.values[out.q.values]
        msg += f" | true pairs kept {int(y.sum()):,} of {int(ts.notna().sum()):,} ({y.sum() / ts.notna().sum():.4f})"
        split = er.split_name(er.id_bucket(ts.fillna(r.entity_id).values))[out.q.values]
        cm = (r.country.values[out.q.values] == "US") & np.isin(split, ["A", "B"])   # French-style copy rows (E14)
        np.save(f"{W}/train_copyqs.npy", np.stack([out.q.values[cm].astype(np.int32), out.s.values[cm].astype(np.int32)]))
        msg += f" | copy pairs {int(cm.sum()):,}"
    log(msg)


# ---------------------------------------------------------------- features
def _fr_chunk(rows):
    return [normalize_record(n, a, "France") for n, a in rows]


def french_copy(s1, r):
    """E14: US records and S1 rewritten French-style (same functions, same seeds), cleaned by the France path."""
    out = []
    for d, is_r in ((s1, False), (r, True)):
        d = d.copy()
        rows = np.flatnonzero(d.country.values == "US")
        names = [fr_name(d.raw_name.values[i], i * 2 + is_r) for i in rows]
        addrs = [fr_addr(d.raw_addr.values[i], i * 3 + is_r) for i in rows]
        pairs = list(zip(names, addrs))
        with Pool() as pool:
            norm = [x for c in pool.map(_fr_chunk, [pairs[i:i + 50000] for i in range(0, len(pairs), 50000)]) for x in c]
        cols = [c for c in er.NORM_COLS if c in d.columns]
        upd = pd.DataFrame(norm, columns=er.NORM_COLS)[cols]
        for c in cols:
            d.loc[d.index[rows], c] = upd[c].values
        d.loc[d.index[rows], "raw_name"], d.loc[d.index[rows], "raw_addr"] = names, addrs
        out.append(d)
    return out


def stats_like_e14(ss, rr):
    """E14 stats(per_country=False): split-wide generic words, IDF, context and core names on the given text."""
    df = Counter()
    for x in list(ss.name.values) + list(rr.name.values):
        df.update(set(x.split()))
    n = len(ss) + len(rr)
    gen = {w for w, k in df.items() if k / n > er.GENERIC_DF}
    idf = {"idf": {w: math.log(n / c) for w, c in df.items() if c > 1}, "default": math.log(n)}
    ss, rr = ss.copy(), rr.copy()
    ss["name_core"] = [core_name(x, gen) for x in ss.name.values]
    rr["name_core"] = [core_name(x, gen) for x in rr.name.values]
    return ss, rr, idf, er.split_context(ss, rr)


def feats(args):
    sp, W = args.split, args.work
    s1, r = er.load_split(args.data, sp, W)
    p = pd.read_parquet(f"{W}/{sp}_pairs61.parquet")
    if args.smoke:                                           # 1 in 100 records, into a separate work folder
        p = p[p.q.values % 100 == 0].reset_index(drop=True)
    C = None
    if not (sp == "train" and args.what == "copies"):
        cos = np.load(f"{W}/{sp}_cos.npy")
        if args.smoke:
            cos = cos[pd.read_parquet(f"{W}/{sp}_pairs61.parquet", columns=["q"]).q.values % 100 == 0]
        assert len(cos) == len(p), "cosines must follow {split}_pairs61 order"
        C = cos_feats(p.q.values, cos)
    OUTW = args.outwork or W
    if sp == "test":
        lt = json.load(open(f"{W}/label_tables.json"))
        idf, ctx = er.fit_idf(s1, r), er.split_context(s1, r)
        parts = []
        for a in range(0, len(p), 10_000_000):
            X = er.features(p.iloc[a:a + 10_000_000].reset_index(drop=True), s1, r, lt["lo"], idf, lt["loc"], ctx)
            parts.append(pd.concat([X, C.iloc[a:a + 10_000_000].reset_index(drop=True)], axis=1))
            log(f"  test features {min(a + 10_000_000, len(p)):,}/{len(p):,}")
        pd.concat(parts, ignore_index=True).to_parquet(f"{OUTW}/test_X.parquet")
        log("test features written")
        return
    truth, r2s = er.load_truth(args.data)
    true_s = r.entity_id.map(r2s)
    y = (s1.entity_id.values[p.s.values] == true_s.values[p.q.values]).astype(np.int8)
    split = er.split_name(er.id_bucket(true_s.fillna(r.entity_id).values))[p.q.values]
    idf, ctx = er.fit_idf(s1, r), er.split_context(s1, r)
    rn, sn = r.name.values[p.q.values], s1.name.values[p.s.values]
    rt, st = r["tail"].values[p.q.values], s1["tail"].values[p.s.values]
    from rapidfuzz import fuzz
    from rapidfuzz.process import cpdist
    tset = cpdist(list(rn), list(sn), scorer=fuzz.token_set_ratio, workers=-1)
    tables = {}
    for part, src in (("A", "B"), ("B", "A"), ("H", "AB")):   # rows of `part` get tables fitted on `src` (v4)
        m = np.isin(split, list(src))
        tables[part] = er._label_tables(rn[m], sn[m], rt[m], st[m], y[m], tset[m])
    if args.what in ("all", "normal"):
        normal_part(args, p, s1, r, y, split, tables, idf, ctx, C, OUTW)
    if args.what in ("all", "copies"):
        copies_part(args, p, s1, r, y, split, tables, OUTW)


def normal_part(args, p, s1, r, y, split, tables, idf, ctx, C, W):
    json.dump({"lo": tables["H"][0], "loc": tables["H"][1]}, open(f"{W}/label_tables.json", "w"))
    X = pd.concat([er.features(p[split == g], s1, r, tables[g][0], idf, tables[g][1], ctx).set_index(np.flatnonzero(split == g))
                   for g in "ABH"]).sort_index()
    X = pd.concat([X, C], axis=1)
    X.assign(y=y, split=split, q=p.q.values, s=p.s.values, p0=p.p0.values).to_parquet(f"{W}/train_X.parquet")
    json.dump({"prefilter_tau": json.load(open(args.v4meta))["prefilter_tau"], "features": [c for c in X.columns],
               "splits": [er.SPLIT_H, er.SPLIT_A]}, open(f"{W}/meta.json", "w"))
    log(f"train features {X.shape} written")


def copies_part(args, p, s1, r, y, split, tables, W):
    # French-style copies of the US training pairs of A and B (E14): same pairs, labels and channel scores;
    # text features on the rewritten text with its own statistics; label tables = the normal ones of the same part
    cm = (r.country.values[p.q.values] == "US") & np.isin(split, ["A", "B"])
    s1f, rf = french_copy(s1, r)
    s1f, rf, idf_f, ctx_f = stats_like_e14(s1f, rf)
    cosc = np.load(f"{args.work}/train_cos_fr.npy")
    if args.smoke:                                           # copy rows of the full run, restricted to the sample
        full = pd.read_parquet(f"{args.work}/train_pairs61.parquet", columns=["q"]).q.values
        qs_c = np.load(f"{args.work}/train_copyqs.npy")[0]
        cosc = cosc[qs_c % 100 == 0]
        del full
    assert len(cosc) == int(cm.sum()), "French-style cosines must follow the copy rows"
    rows_c = np.flatnonzero(cm)
    Cc = cos_feats(p.q.values[rows_c], cosc).set_index(rows_c)
    Xc = pd.concat([er.features(p[cm & (split == g)], s1f, rf, tables[g][0], idf_f, tables[g][1], ctx_f)
                    .set_index(np.flatnonzero(cm & (split == g))) for g in "AB"]).sort_index()
    Xc = pd.concat([Xc, Cc.loc[Xc.index]], axis=1)
    Xc.assign(y=y[Xc.index.values], split=split[Xc.index.values], q=p.q.values[Xc.index.values]).to_parquet(f"{W}/train_Xc.parquet")
    log(f"French-style copies {Xc.shape} written")


# ---------------------------------------------------------------- stage 1
def stage1(args):
    W, P = args.work, args.part
    other = "B" if P == "A" else "A"
    meta = json.load(open(f"{W}/meta.json"))
    fcols = meta["features"]
    X = pd.read_parquet(f"{W}/train_X.parquet")
    Xc = pd.read_parquet(f"{W}/train_Xc.parquet")
    r = er.load_split(args.data, "train", W)[1]
    mP = X.split.values == P
    Xc = Xc[Xc.split.values == P]
    Xtr = pd.concat([X.loc[mP, fcols], Xc[fcols]], ignore_index=True)
    ytr = np.r_[X.y.values[mP], Xc.y.values]
    inner = er.id_bucket(r.entity_id.values[np.r_[X.q.values[mP], Xc.q.values]]) % 10 == 0   # E14: copies held out alike
    mo, mh = X.split.values == other, X.split.values == "H"
    for k in [int(x) for x in args.seeds.split(",")]:
        params = dict(er.LGB_PARAMS, **REG, seed=k, bagging_seed=k, feature_fraction_seed=k)
        m = er.train_lgb(Xtr, ytr, inner, params=params)
        m.save_model(f"{W}/stage1_{P}_s{k}.txt")
        np.save(f"{W}/p1_from_{P}_s{k}.npy", m.predict(X.loc[mo, fcols], num_threads=0))
        np.save(f"{W}/p1H_from_{P}_s{k}.npy", m.predict(X.loc[mh, fcols], num_threads=0))
        log(f"stage 1 part {P} seed {k}: {m.num_trees()} trees")


def finish(args):
    W = args.work
    X = pd.read_parquet(f"{W}/train_X.parquet")
    p1 = np.zeros(len(X), np.float32)
    L = lambda pat: [np.load(f"{W}/{pat}_s{k}.npy") for k in SEEDS]      # all seeds must be present
    p1[X.split.values == "B"] = np.mean(L("p1_from_A"), axis=0)
    p1[X.split.values == "A"] = np.mean(L("p1_from_B"), axis=0)
    p1[X.split.values == "H"] = (np.mean(L("p1H_from_A"), axis=0) + np.mean(L("p1H_from_B"), axis=0)) / 2
    out = X[["q", "s"]].assign(p0=X.p0.values, p1=p1, y=X.y.values, split=X.split.values)
    pd.concat([out, X[EXTRA].reset_index(drop=True)], axis=1).to_parquet(f"{W}/train_scored.parquet")
    log(f"train_scored.parquet written ({len(out):,} pairs)")


def score(args):
    W = args.work
    meta = json.load(open(f"{W}/meta.json"))
    models = [lgb.Booster(model_file=f) for f in sorted(glob.glob(f"{W}/stage1_[AB]_s*.txt"))]
    assert len(models) == 2 * len(SEEDS), f"expected {2 * len(SEEDS)} stage-1 models, found {len(models)}"
    s1, r = er.load_split(args.data, "test", W)
    p = pd.read_parquet(f"{W}/test_pairs61.parquet", columns=["q", "s", "p0"])
    X = pd.read_parquet(f"{W}/test_X.parquet")
    p1 = np.mean([m.predict(X[meta["features"]], num_threads=0) for m in models], axis=0).astype(np.float32)
    out = pd.concat([p.assign(p1=p1), X[EXTRA].reset_index(drop=True)], axis=1)
    out.to_parquet(f"{W}/test_scored.parquet")
    os.makedirs(args.out, exist_ok=True)
    er.write_lists(f"{args.out}/candidate_pairs.tsv", "candidate_entity_ids", s1.entity_id.values, er.to_sets(out, s1, r))
    log(f"test_scored.parquet + candidate_pairs.tsv written ({len(out):,} pairs)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["union", "feats", "stage1", "finish", "score"])
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", default="output")
    ap.add_argument("--split", choices=["train", "test"])
    ap.add_argument("--part", choices=["A", "B"])
    ap.add_argument("--v4", help="folder with v4's {split}_pairs.parquet")
    ap.add_argument("--prefilter", help="v4 prefilter model")
    ap.add_argument("--v4meta", help="v4 meta.json (prefilter_tau)")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--what", choices=["all", "normal", "copies"], default="all", help="train feats: which part")
    ap.add_argument("--seeds", default="0,1,2", help="stage1: which seeds this machine trains")
    ap.add_argument("--outwork", default="", help="feats: write outputs here (default --work)")
    ap.add_argument("--smoke", action="store_true", help="1 in 50 records, for a quick correctness run")
    a = ap.parse_args()
    {"union": union, "feats": feats, "stage1": stage1, "finish": finish, "score": score}[a.stage](a)
