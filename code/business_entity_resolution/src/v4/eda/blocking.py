"""EDA: blocking recall at FULL S1 density for a 3% sample of S2/S3 queries.
Recall@K per channel (name char-3gram, address words, name+address words) and their union,
per country / source / script / name-visibility stratum; plus how optimistic a 10%-S1 index is."""
import os, sys
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from er import read_tsv, id_bucket, log

W, KMAX = "work_eda", 50
CH = {
    "name": ("name_ns", dict(analyzer="char", ngram_range=(3, 3), max_df=0.05)),
    "addr": ("addr", dict(analyzer="word", token_pattern=r"\S+", max_df=0.02)),
    "both": ("both", dict(analyzer="word", token_pattern=r"\S+", max_df=0.02)),
}

def ranks(s1, q, field, kw):
    """Rank (1-based) of each query's true S1 in the top-KMAX, or 999 if absent."""
    v = TfidfVectorizer(dtype=np.float32, sublinear_tf=True, **kw).fit(pd.concat([s1[field], q[field]]))
    S, Q = v.transform(s1[field]), v.transform(q[field])
    C = sp_matmul_topn(Q, S.T.tocsr(), top_n=KMAX, threshold=0.0, sort=True, n_threads=-1)
    out = np.full(len(q), 999, np.int16)
    tpos = q.true_pos.values
    for i in range(len(q)):
        if tpos[i] < 0: continue
        cols = C.indices[C.indptr[i]:C.indptr[i + 1]]
        hit = np.flatnonzero(cols == tpos[i])
        if len(hit): out[i] = hit[0] + 1
    return out

def evaluate(s1, q):
    res = []
    for c in sorted(q.country.unique()):
        sc, qc = s1[s1.country == c].reset_index(drop=True), q[q.country == c].reset_index(drop=True)
        pos = pd.Series(np.arange(len(sc)), index=sc.entity_id)
        qc["true_pos"] = qc.true_s1.map(pos).fillna(-1).astype(int).values
        for ch, (field, kw) in CH.items():
            qc["r_" + ch] = ranks(sc, qc, field, kw)
        res.append(qc)
        log("  ranked", c, len(sc), len(qc))
    return pd.concat(res, ignore_index=True)

def table(q, by=None):
    q = q[q.true_pos >= 0]
    rows = {}
    groups = [("all", q)] if by is None else list(q.groupby(by))
    for g, d in groups:
        r = {f"{ch}@{k}": (d["r_" + ch] <= k).mean() for ch in CH for k in (1, 5, 10, 20, 50)}
        for k in (5, 10):
            r[f"name|addr@{k}"] = ((d.r_name <= k) | (d.r_addr <= k)).mean()
            r[f"ALL3@{k}"] = ((d.r_name <= k) | (d.r_addr <= k) | (d.r_both <= k)).mean()
        r["n"] = len(d)
        rows[g] = r
    return pd.DataFrame(rows).T

if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    s1 = pd.read_parquet(f"{W}/train_s1.parquet", columns=["entity_id", "country", "name", "name_ns", "addr"])
    q = pd.read_parquet(f"{W}/train_q.parquet", columns=["entity_id", "country", "name", "name_ns", "addr", "nonlatin"])
    gt = read_tsv("dataset/train/train_ground_truth.tsv")
    r2s = {m: s for s, ms in zip(gt.source1_entity_id, gt.matched_entity_ids) for m in ms.split(",") if m}
    q["true_s1"] = q.entity_id.map(r2s)
    for d in (s1, q):
        d["both"] = d.name + " " + d.addr
    # name-visibility stratum vs the true S1 name
    s1n = s1.set_index("entity_id").name
    tn = q.true_s1.map(s1n).fillna("")
    q["stratum"] = [("no_match" if not t else "identical" if a == t else
                     "share_word" if set(a.split()) & set(t.split()) else "no_shared_word")
                    for a, t in zip(q.name, tn)]
    q["source"] = q.entity_id.str[:2]
    log(f"queries {len(q):,}, with a true match {q.true_s1.notna().sum():,}")
    print("\nshare of matched queries per name stratum:\n", q[q.true_s1.notna()].stratum.value_counts(normalize=True).round(4))

    full = evaluate(s1, q)
    full.to_parquet(f"{W}/ranks_full.parquet")
    print("\n=== FULL-density recall ===\n", table(full).round(4).to_string())
    for by in ("country", "source", "nonlatin", "stratum"):
        print(f"\n--- by {by} ---\n", table(full, by)[["name@5", "addr@5", "both@5", "name|addr@5", "ALL3@5", "ALL3@10", "n"]].round(4).to_string())

    # optimism check: index only 10% of S1 (like a small sample), queries whose true S1 is in that 10%
    keep = id_bucket(s1.entity_id.values) < 10
    s1s = s1[keep]
    qs = q[q.true_s1.isin(set(s1s.entity_id)) | q.true_s1.isna()]
    small = evaluate(s1s, qs)
    both = full.set_index("entity_id").loc[small.entity_id]
    cmp = pd.DataFrame({"full_density": table(both.reset_index()).iloc[0], "10pct_index": table(small).iloc[0]})
    print("\n=== sample optimism (same queries) ===\n", cmp.loc[["name@1", "name@5", "addr@5", "name|addr@5", "ALL3@5"]].round(4).to_string())
