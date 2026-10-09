"""E13: does the bge-m3 retrieval channel raise held-out F0.5 end to end? (train labels only)
Held-out H records (entity buckets [0,2): true S1 in H, plus unmatched records hashed there) against the FULL S1 index
of their country. Two shortlists built the same way: v4's channels only, and v4's channels + bge-m3 top-20
(name | address). Both go through v4's own prefilter (same tau), 74 features (H label tables) and stage-1 models (A/B
average), then one threshold per variant tuned on H half A; reported on half B, paired per S1 (bootstrap 95% CI).
Stage-1 level (no cross-encoder / stage 2 for the new pairs), so it measures what the shortlist change can deliver.
Usage (GPU machine with ~/w data + ~/src code): python e13_retrieval.py <work> <src>"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

W = sys.argv[1]
sys.path.insert(0, sys.argv[2])
import er  # noqa: E402
import stage2 as S  # noqa: E402

t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:5.0f}s]", *a, flush=True)
s1, r = er.load_split("dataset", "train", W)
truth, r2s = er.load_truth("dataset")
meta = json.load(open(f"{W}/v4_meta.json"))
lt = json.load(open(f"{W}/v4_label_tables.json"))
pre = lgb.Booster(model_file=f"{W}/v4_prefilter.txt")
models = [lgb.Booster(model_file=f"{W}/v4_stage1_{k}.txt") for k in "AB"]
rb = er.id_bucket(r.entity_id.map(r2s).fillna(r.entity_id).values)
rh = r[rb < 2].reset_index(drop=True)                                 # held-out records
idx_h = np.flatnonzero(rb < 2)
log(f"held-out records {len(rh):,}")

# ---- bge-m3 top-20 per held-out record within its country (name | address)
import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
m = SentenceTransformer("BAAI/bge-m3", device="cuda"); m.half(); m.max_seq_length = 64
txt = lambda d: (d.raw_name.fillna("") + " | " + d.raw_addr.fillna("")).tolist()
dense = []
for c in rh.country.unique():
    si = np.flatnonzero(s1.country.values == c); qi = np.flatnonzero(rh.country.values == c)
    Sm = torch.tensor(m.encode(txt(s1.iloc[si]), batch_size=2048, normalize_embeddings=True), device="cuda").half()
    Qm = torch.tensor(m.encode(txt(rh.iloc[qi]), batch_size=2048, normalize_embeddings=True), device="cuda").half()
    for a in range(0, len(qi), 4096):
        v, i = (Qm[a:a + 4096] @ Sm.T).float().topk(20, dim=1)
        dense.append(pd.DataFrame({"q": np.repeat(qi[a:a + 4096], 20), "s": si[i.cpu().numpy().ravel()],
                                   "sc_dense": v.cpu().numpy().ravel().astype(np.float32)}))
    del Sm, Qm
dense = pd.concat(dense, ignore_index=True)
log(f"bge-m3 pairs {len(dense):,}")

# ---- shortlist (v4 channels) with / without the dense channel, then v4 prefilter / features / stage 1
idf, ctx = er.fit_idf(s1, r), er.split_context(s1, r)            # statistics of the whole train split, as in v4
ctx_h = {"s1_name_freq": ctx["s1_name_freq"], "s1_addr_freq": ctx["s1_addr_freq"], "r_name_freq": ctx["r_name_freq"][idx_h]}
out = {}
import os  # noqa: E402
for variant, dn in (("v4", None), ("v4+bge-m3", dense)):
    cache = f"{W}/e13_{variant.replace('+', '_')}.parquet"
    if os.path.exists(cache):
        out[variant] = pd.read_parquet(cache); log(f"{variant}: {len(out[variant]):,} pairs (cached)"); continue
    p = er.block(s1, rh, dn)
    if "sc_dense" in p.columns:
        p = p.drop(columns=["sc_dense"])
    p = pd.concat([p, er._competition_feats(p)], axis=1)
    p0 = pre.predict(p[er.cheap_cols(p)][pre.feature_name()], num_threads=0)
    p = p[p0 >= meta["prefilter_tau"]].reset_index(drop=True)
    X = er.features(p, s1, rh, lt["lo"], idf, lt["loc"], ctx_h)
    p["p1"] = np.mean([mm.predict(X[meta["features"]], num_threads=0) for mm in models], axis=0)
    out[variant] = p[["q", "s", "p1"]]
    out[variant].to_parquet(cache)
    log(f"{variant}: {len(p):,} pairs after prefilter")

h_ids = s1.entity_id.values[er.split_name(er.id_bucket(s1.entity_id.values)) == "H"]
half = er.id_bucket(np.array([i[::-1] for i in h_ids])) < 50
th = {i: truth[i] for i in h_ids}
s1_pos = pd.Series(np.arange(len(s1)), index=s1.entity_id.values)
true_s = rh.entity_id.map(r2s).map(s1_pos).fillna(-1).astype(np.int64).values
grid = np.round(np.arange(0.1, 0.96, 0.025), 3)
res = {}
for variant, p in out.items():
    have = set(zip(p.q.values, p.s.values))
    rec = sum((q, t) in have for q, t in zip(np.arange(len(rh)), true_s) if t >= 0) / max((true_s >= 0).sum(), 1)
    run = S._fast_decider(p, p.p1.values, rh)
    pred = lambda t: er.to_sets(run({"t": {k: t for k in S.SLICES}, "rescue": 0.0}), s1, rh)
    ids_a = h_ids[half]
    t_best = max(grid, key=lambda t: er.macro_f05(pred(t), th, ids_a))
    pb = pred(t_best)                                   # decided once (was re-decided for every S1)
    res[variant] = np.array([er.macro_f05(pb, th, [i]) for i in h_ids[~half]])
    log(f"{variant}: candidate recall of true pairs {rec:.4f} | threshold {t_best} | half B {res[variant].mean():.4f}")
rng = np.random.default_rng(0)
dd = res["v4+bge-m3"] - res["v4"]
B = rng.integers(0, len(dd), size=(1000, len(dd)))
print(f"\n=== E13: bge-m3 channel end to end (stage 1, held-out half B, {len(dd):,} S1) ===\n"
      f"v4 {res['v4'].mean():.4f} -> v4+bge-m3 {res['v4+bge-m3'].mean():.4f} | paired diff {dd.mean():+.4f} "
      f"[{np.quantile(dd[B].mean(1), .025):+.4f}, {np.quantile(dd[B].mean(1), .975):+.4f}]")
