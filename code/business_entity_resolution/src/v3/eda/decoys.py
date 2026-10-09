"""What do decoys (S2/S3 records matching no S1) look like vs true matches? US+India, full S1 index,
3% record sample. For each record: top-3 S1 by the name+addr channel, then name/addr similarity to top-1."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from er import read_tsv, _vectorize
from sparse_dot_topn import sp_matmul_topn
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

s1 = pd.read_parquet("work_eda/train_s1.parquet"); q = pd.read_parquet("work_eda/train_q.parquet")
gt = read_tsv("dataset/train/train_ground_truth.tsv")
r2s = {m: s for s, ms in zip(gt.source1_entity_id, gt.matched_entity_ids) for m in ms.split(",") if m}
q["t"] = q.entity_id.map(r2s); q["decoy"] = q.t.isna()
print("decoy share in sample:", q.decoy.mean().round(4))
out = []
for c in ("US", "India"):
    si = np.flatnonzero(s1.country.values == c); qc = q[q.country == c].sample(40000, random_state=0)
    for d in (s1, qc): d["both"] = d.name + " " + d.addr
    S, Q = _vectorize(s1.both.iloc[si], qc.both, dict(analyzer="word", token_pattern=r"\S+", ngram_range=(1, 2), max_df=0.005))
    C = sp_matmul_topn(Q, S.T.tocsr(), top_n=2, threshold=0.0, sort=True, n_threads=-1)
    top1 = np.array([si[C.indices[C.indptr[i]]] if C.indptr[i + 1] > C.indptr[i] else -1 for i in range(len(qc))])
    sc1 = np.array([C.data[C.indptr[i]] if C.indptr[i + 1] > C.indptr[i] else 0 for i in range(len(qc))])
    sc2 = np.array([C.data[C.indptr[i] + 1] if C.indptr[i + 1] - C.indptr[i] > 1 else 0 for i in range(len(qc))])
    qc = qc.assign(top1=top1, sc1=sc1, gap=sc1 - sc2)
    qc = qc[qc.top1 >= 0]
    qc["n_sim"] = cpdist(list(qc.name), list(s1.name.values[qc.top1]), scorer=fuzz.token_set_ratio, workers=-1)
    qc["a_sim"] = cpdist(list(qc.addr), list(s1.addr.values[qc.top1]), scorer=fuzz.token_set_ratio, workers=-1)
    qc["num_eq"] = cpdist(list(qc.nums), list(s1.nums.values[qc.top1]), scorer=fuzz.token_set_ratio, workers=-1)
    qc["top1_true"] = s1.entity_id.values[qc.top1] == qc.t.values
    qc["top1_id"] = s1.entity_id.values[qc.top1]
    out.append(qc)
d = pd.concat(out)
def kind(x):
    return np.select([x.decoy, x.top1_true], ["decoy", "true@top1"], "true-not-top1")
d["kind"] = kind(d)
d["pattern"] = np.select(
    [(d.n_sim >= 90) & (d.a_sim >= 90), (d.n_sim >= 90) & (d.a_sim < 60), (d.n_sim < 60) & (d.a_sim >= 90),
     (d.n_sim < 60) & (d.a_sim < 60)],
    ["name+addr both match", "same name, other address", "same address, other name", "neither"], "partial")
pd.set_option("display.width", 200)
print("\n=== nearest-S1 pattern by record kind (row %) ===")
print(pd.crosstab([d.country, d.kind], d.pattern, normalize="index").round(3).to_string())
print("\n=== median similarities ===")
print(d.groupby(["country", "kind"])[["sc1", "gap", "n_sim", "a_sim", "num_eq"]].median().round(3).to_string())
# is a decoy's nearest S1 an entity that ALSO has true matches? (decoys generated as lookalikes of real entities)
has_match = set(gt.source1_entity_id[gt.matched_entity_ids != ""])
dd = d[d.decoy]
print("\ndecoys whose nearest S1 has true matches:", dd.top1_id.isin(has_match).mean().round(3),
      "| base rate of S1 with matches:", round(len(has_match) / len(gt), 3))
sn = s1.set_index("entity_id")
for pat in ("name+addr both match", "same name, other address", "same address, other name", "partial"):
    ex = dd[dd.pattern == pat].head(6)
    print(f"\n--- decoy examples: {pat} ({(dd.pattern == pat).mean():.1%} of decoys) ---")
    for _, x in ex.iterrows():
        print(f"  decoy: {x.business_name[:40]:42s}| {x.business_address[:70]}")
        print(f"  S1   : {sn.loc[x.top1_id].business_name[:40]:42s}| {sn.loc[x.top1_id].business_address[:70]}")

# ---- house-number delta and added-name-token signatures: decoy vs its nearest S1, true record vs its true S1
import re
first = lambda s: int(m.group()) if (m := re.search(r"\d+", s)) else -1
s1i = s1.set_index("entity_id")
tr = d[~d.decoy].copy(); tr["ref"] = tr.t
dc = d[d.decoy].copy(); dc["ref"] = dc.top1_id
both = pd.concat([tr.assign(grp="true pair"), dc.assign(grp="decoy vs nearest")])
ra, sa = both.addr.map(first).values, both.ref.map(s1i.addr).map(first).values
delta = np.where((ra < 0) | (sa < 0), -1, np.abs(ra - sa))
both["num_delta"] = pd.cut(delta, [-2, -0.5, 0.5, 20.5, 100.5, 1e12],
                           labels=["missing", "equal", "1-20", "21-100", ">100"])
print("\n=== first house number: |record - S1| (row %) ===")
print(pd.crosstab([both.country, both.grp], both.num_delta, normalize="index").round(3).to_string())
# extra name tokens (in record, not in S1 name), after normalization
extra = [set(a.split()) - set(b.split()) for a, b in zip(both.name, both.ref.map(s1i.name))]
both["n_extra"] = [len(e) for e in extra]
print("\n=== # extra name tokens (row %) ===")
print(pd.crosstab(both.grp, both.n_extra.clip(upper=3), normalize="index").round(3).to_string())
from collections import Counter
for g in ("decoy vs nearest", "true pair"):
    cnt = Counter(t for e, gg in zip(extra, both.grp) if gg == g for t in e)
    print(f"\ntop extra tokens, {g}:", cnt.most_common(30))
