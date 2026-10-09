"""Time + recall of each blocking channel on India (full 883k S1, 20k sampled matched queries)."""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from er import read_tsv, _vectorize
from normalize import skeleton
from sparse_dot_topn import sp_matmul_topn

s1 = pd.read_parquet("work_eda/train_s1.parquet"); q = pd.read_parquet("work_eda/train_q.parquet")
gt = read_tsv("dataset/train/train_ground_truth.tsv")
r2s = {m: s for s, ms in zip(gt.source1_entity_id, gt.matched_entity_ids) for m in ms.split(",") if m}
q["t"] = q.entity_id.map(r2s)
s1, q = s1[s1.country == "India"].reset_index(drop=True), q[(q.country == "India") & q.t.notna()].sample(20000, random_state=0)
for d in (s1, q): d["both"] = d.name + " " + d.addr
s1["name_sk"], q["name_sk"] = s1.business_name.map(skeleton), q.business_name.map(skeleton)
tcol = q.t.map(pd.Series(np.arange(len(s1)), index=s1.entity_id)).values
hits = {}
for label, field, k, kw in [
    ("name char3 max_df=.005", "name_ns", 5, dict(analyzer="char", ngram_range=(3, 3), max_df=0.005)),
    ("addr word max_df=.005", "addr", 5, dict(analyzer="word", token_pattern=r"\S+", max_df=0.005)),
    ("addr word max_df=.002", "addr", 5, dict(analyzer="word", token_pattern=r"\S+", max_df=0.002)),
    ("both word max_df=.01", "both", 10, dict(analyzer="word", token_pattern=r"\S+", max_df=0.01)),
    ("both word max_df=.005", "both", 10, dict(analyzer="word", token_pattern=r"\S+", max_df=0.005)),
    ("both word2 max_df=.005", "both", 10, dict(analyzer="word", token_pattern=r"\S+", ngram_range=(1, 2), max_df=0.005)),
    ("skel char3 max_df=.005", "name_sk", 5, dict(analyzer="char", ngram_range=(3, 3), max_df=0.005)),
]:
    t0 = time.time(); S, Q = _vectorize(s1[field], q[field], kw); t1 = time.time()
    C = sp_matmul_topn(Q, S.T.tocsr(), top_n=k, threshold=0.05, n_threads=-1); t2 = time.time()
    hit = np.array([t in C.indices[C.indptr[i]:C.indptr[i + 1]] for i, t in enumerate(tcol)])
    hits[label] = hit
    work = (Q.astype(bool) @ S.T.astype(bool).sum(axis=1)).sum() if False else None
    print(f"{label:26s} recall@{k} {hit.mean():.4f} | vectorize {t1-t0:5.1f}s  matmul {t2-t1:6.1f}s "
          f"-> est. matmul for 4.1M India queries: {(t2-t1)*4.1e6/2e4/60:6.1f} min", flush=True)
u = lambda *ls: np.logical_or.reduce([hits[l] for l in ls]).mean()
for combo in [("name char3 max_df=.005", "addr word max_df=.005", "both word max_df=.01", "skel char3 max_df=.005"),
              ("name char3 max_df=.005", "addr word max_df=.005", "both word max_df=.005", "skel char3 max_df=.005"),
              ("name char3 max_df=.005", "addr word max_df=.002", "both word2 max_df=.005", "skel char3 max_df=.005"),
              ("name char3 max_df=.005", "addr word max_df=.005", "both word2 max_df=.005", "skel char3 max_df=.005")]:
    print(round(u(*combo), 4), " <- ", " + ".join(combo))
