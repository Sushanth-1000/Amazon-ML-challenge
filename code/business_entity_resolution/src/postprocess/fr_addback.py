"""France recall add-back: unmatched French records whose best self-trained stage-1 p1 >= tau join that S1 (cap 11)."""
import sys; sys.path.insert(0, "/home/Dell/src61")
import numpy as np, pandas as pd, er
tau = float(sys.argv[1]); out = sys.argv[2]
rd = lambda p: pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=3)
s1, r = er.load_split("/home/Dell/dataset", "test", "/home/Dell/w61")
B = rd("/home/Dell/fr/B.tsv"); sets = {k: [x for x in v.split(",") if x] for k, v in zip(B.source1_entity_id, B.matched_entity_ids)}
matched = set(x for v in sets.values() for x in v)
t = pd.read_parquet("/home/Dell/w62a/test_scored.parquet", columns=["q", "s", "p1"])
t = t[r.country.values[t.q.values] == "France"]
best = t.sort_values(["q", "p1"], ascending=[True, False]).drop_duplicates("q")
best = best[best.p1.values >= tau]
rid, sid = r.entity_id.values, s1.entity_id.values
best = best[~pd.Series(rid[best.q.values]).isin(matched).values].sort_values("p1", ascending=False)
added = 0
for q, s in zip(best.q.values, best.s.values):
    k = sid[s]
    if len(sets.get(k, [])) < 11:
        sets.setdefault(k, []).append(rid[q]); added += 1
B["matched_entity_ids"] = [",".join(sets.get(k, [])) for k in B.source1_entity_id]
B.to_csv(out, sep="\t", index=False, quoting=3)
print(f"tau {tau}: candidates {len(best):,} | added {added:,}")
