"""E29: where can v6.1 still grow on US/India? Held-out half B, per-S1 macro F0.5 (singletons included), full scale.
For every missed true record and every wrong match, the error type (not shortlisted / ranked wrong / below threshold /
wrong business / orphan) x record slice (no-address / native script / other) x country x S1 size; gain = per-S1 F0.5
points recovered if exactly that cell were fixed (x100), with a 95% bootstrap CI over S1. Plus ceilings: perfect
decisions on the shortlist v6.1 has (oracle threshold per S1 / oracle ranking)."""
import argparse
import json
import sys

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--work", required=True); ap.add_argument("--run", required=True)
ap.add_argument("--src", required=True)
a = ap.parse_args()
sys.path.insert(0, a.src)
import er  # noqa: E402
import stage2 as S  # noqa: E402

s1, r = er.load_split(a.data, "train", a.work)
truth, r2s = er.load_truth(a.data)
meta = json.load(open(f"{a.run}/meta.json"))
d = pd.read_parquet(f"{a.run}/train_stage2.parquet")
col = {"stage1": "p1", "pass1": "p2a", "pass2": "p2", "pass2+noaddr": "p2n"}[meta.get("stage2_choice", "pass1")]
h_ids = s1.entity_id.values[er.split_name(er.id_bucket(s1.entity_id.values)) == "H"]
half = er.id_bucket(np.array([i[::-1] for i in h_ids])) < 50
ids_b = h_ids[~half]
th = {i: truth[i] for i in h_ids}
cty = s1.set_index("entity_id").country.reindex(ids_b).values
rid, sid = r.entity_id.values, s1.entity_id.values
slice_r = pd.Series(S.slice_of(r, np.arange(len(r))), index=rid)
P0 = er.to_sets(S._fast_decider(d, d[col].values, r)(meta["decision"]), s1, r)


def f(P, G):
    if not G: return float(not P)
    tp = len(P & G)
    return 0.0 if tp == 0 else 1.25 * (tp / len(P)) * (tp / len(G)) / (0.25 * (tp / len(P)) + tp / len(G))


f0 = np.array([f(P0.get(i, set()), th[i]) for i in ids_b])
print(f"v6.1 half B per-S1: all {f0.mean():.4f} | US {f0[cty == 'US'].mean():.4f} | India {f0[cty == 'India'].mean():.4f} "
      f"| S1 {len(ids_b):,}, perfect {np.mean(f0 == 1):.4f}", flush=True)

best = d.assign(p=d[col].values).sort_values(["q", "p"], ascending=[True, False]).drop_duplicates("q")
best_s = dict(zip(rid[best.q.values], sid[best.s.values]))
in_d = set(zip(rid[d.q.values], sid[d.s.values]))
size = lambda n: "singleton" if n == 0 else ("1-2" if n <= 2 else ("3-5" if n <= 5 else "6-11"))
cells = {}   # (error, slice) -> {s1: (add set, remove set)}
for i in ids_b:
    G, P = th[i], P0.get(i, set())
    for q in G - P:
        e = "not_shortlisted" if (q, i) not in in_d else ("ranked_wrong" if best_s.get(q) != i else "below_threshold")
        cells.setdefault((e, slice_r.get(q, "other")), {}).setdefault(i, [set(), set()])[0].add(q)
    for q in P - G:
        e = "fp_orphan" if q not in r2s else "fp_wrong_business"
        cells.setdefault((e, slice_r.get(q, "other")), {}).setdefault(i, [set(), set()])[1].add(q)

rng = np.random.default_rng(0)
Bi = {c: rng.integers(0, int((cty == c).sum()), size=(500, int((cty == c).sum()))) for c in ("US", "India")}
pos = {i: k for k, i in enumerate(ids_b)}
rows = []
for (e, sl), v in cells.items():
    g = np.zeros(len(ids_b))
    for i, (add, rem) in v.items():
        k = pos[i]; g[k] = f((P0.get(i, set()) | add) - rem, th[i]) - f0[k]
    row = dict(error=e, slice=sl, S1_affected=len(v), gain_all=100 * g.mean())
    for c in ("US", "India"):
        m = cty == c; gm = g[m]; bm = gm[Bi[c]].mean(1)
        row[f"gain_{c}"] = 100 * gm.mean(); row[f"ci_{c}"] = f"[{100 * np.quantile(bm, .025):.3f}, {100 * np.quantile(bm, .975):.3f}]"
    rows.append(row)
T = pd.DataFrame(rows).sort_values("gain_all", ascending=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print("\n=== headroom by error type x record slice (points x100 of per-S1 F0.5 if that cell were fixed perfectly) ===")
print(T.to_string(index=False, float_format=lambda x: f"{x:.3f}"), flush=True)

# by S1 size
sz = np.array([size(len(th[i])) for i in ids_b])
print("\n=== v6.1 per-S1 F0.5 by S1 size (true matches) ===")
for c in ("US", "India"):
    for s_ in ("singleton", "1-2", "3-5", "6-11"):
        m = (cty == c) & (sz == s_)
        if m.any():
            print(f"{c:5s} {s_:9s}: {m.sum():6,} S1 ({m.mean():.3f} of half B) | F0.5 {f0[m].mean():.4f} | "
                  f"points lost x100 {100 * (1 - f0[m]).sum() / len(ids_b):.3f}", flush=True)

# ceilings on v6.1's own shortlist
Pr = {}
for i in ids_b:
    G = th[i]; Pr[i] = {q for q in G if (q, i) in in_d}
fr = np.array([f(Pr[i], th[i]) for i in ids_b])
print(f"\nCEILING perfect decisions on v6.1's shortlist (every shortlisted true pair accepted, nothing else): "
      f"all {fr.mean():.4f} | US {fr[cty == 'US'].mean():.4f} | India {fr[cty == 'India'].mean():.4f}")
print(f"   -> headroom from decisions/ranking {100 * (fr - f0).mean():.3f} pts; from search (shortlist) {100 * (1 - fr).mean():.3f} pts")
print("E29 DONE", flush=True)
