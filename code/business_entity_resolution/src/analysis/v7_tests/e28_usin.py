"""E28: US/India improvement tests on v6.1's held-out data (full scale). Everything is TUNED on held-out half A and
SCORED on held-out half B (per-S1 macro F0.5, singletons included); paired bootstrap 95% CI vs v6.1's own decision.
  T1 loss breakdown of v6.1 on half B (where the remaining points go)
  T2 per-country decision: v6.1's tune() run separately for US and India records (half-A S1 of that country)
  T3 stage-2 probability variants: pass2, pass2+noaddr, and averages with pass1, each re-tuned on half A
  T4 cap per S1 (8..11) on v6.1's decision
Usage: python e28_usin.py --data dataset --work <train s1/r> --run <folder with train_stage2.parquet + meta.json>"""
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--work", required=True); ap.add_argument("--run", required=True)
ap.add_argument("--src", required=True)
a = ap.parse_args()
sys.path.insert(0, a.src)
import er  # noqa: E402
import stage2 as S  # noqa: E402

t0 = time.time()
log = lambda *x: print(f"[{time.time() - t0:6.0f}s]", *x, flush=True)
s1, r = er.load_split(a.data, "train", a.work)
truth, r2s = er.load_truth(a.data)
meta = json.load(open(f"{a.run}/meta.json"))
d = pd.read_parquet(f"{a.run}/train_stage2.parquet")
log(f"train_stage2 {len(d):,} pairs, columns {list(d.columns)[:30]}")
COL = {"stage1": "p1", "pass1": "p2a", "pass2": "p2", "pass2+noaddr": "p2n"}
base_col = COL[meta.get("stage2_choice", "pass1")]
dec0 = meta["decision"]

h_ids = s1.entity_id.values[er.split_name(er.id_bucket(s1.entity_id.values)) == "H"]
half = er.id_bucket(np.array([i[::-1] for i in h_ids])) < 50
ids_a, ids_b = h_ids[half], h_ids[~half]
th = {i: truth[i] for i in h_ids}
cty_of = s1.set_index("entity_id").country
cty_b = cty_of.reindex(ids_b).values
rc = r.country.values[d.q.values]


def per_s1(pred, ids):
    return np.array([er.macro_f05(pred, th, [i]) for i in ids])


def preds(prob, dec, dd=d):
    return er.to_sets(S._fast_decider(dd, prob, r)(dec), s1, r)


rng = np.random.default_rng(0)
B = {c: rng.integers(0, int((cty_b == c).sum()) if c != "all" else len(ids_b),
                     size=(1000, int((cty_b == c).sum()) if c != "all" else len(ids_b))) for c in ("US", "India", "all")}


def report(name, f_new, f_base):
    out = []
    for c in ("US", "India", "all"):
        m = np.ones(len(ids_b), bool) if c == "all" else cty_b == c
        dd = f_new[m] - f_base[m]; bm = dd[B[c]].mean(1)
        lo, hi = np.quantile(bm, [.025, .975])
        out.append(f"{c} {f_base[m].mean():.4f}->{f_new[m].mean():.4f} {dd.mean():+.5f} [{lo:+.5f}, {hi:+.5f}]")
    print(f"RESULT {name:34s} | " + " | ".join(out), flush=True)


# ---- baseline (v6.1 as submitted)
P0 = preds(d[base_col].values, dec0)
f0 = per_s1(P0, ids_b)
log(f"baseline v6.1 ({base_col}, {dec0}) half B per-S1: all {f0.mean():.4f} | US {f0[cty_b == 'US'].mean():.4f} | "
    f"India {f0[cty_b == 'India'].mean():.4f}")

# ---- T1 loss breakdown on half B: gain if each error type were fixed perfectly (per-S1, points x100)
best = d.assign(p=d[base_col].values).sort_values(["q", "p"], ascending=[True, False]).drop_duplicates("q")
best_s = dict(zip(r.entity_id.values[best.q.values], s1.entity_id.values[best.s.values]))
in_d = set(zip(r.entity_id.values[d.q.values], s1.entity_id.values[d.s.values]))
fix = {k: {} for k in ("not_shortlisted", "ranked_wrong", "below_threshold", "fp_wrong_business", "fp_orphan")}
for i in ids_b:
    G, P = th[i], P0.get(i, set())
    for q in G - P:
        k = "not_shortlisted" if (q, i) not in in_d else ("ranked_wrong" if best_s.get(q) != i else "below_threshold")
        fix[k].setdefault(i, set()).add(q)
    for q in P - G:
        fix["fp_orphan" if q not in r2s else "fp_wrong_business"].setdefault(i, set()).add(q)
for k, v in fix.items():
    Pk = {i: (P0.get(i, set()) | v.get(i, set())) if not k.startswith("fp") else (P0.get(i, set()) - v.get(i, set()))
          for i in ids_b}
    fk = per_s1(Pk, ids_b)
    print(f"LOSS {k:18s}: fixing it gains all {100 * (fk - f0).mean():.3f} pts | US {100 * (fk - f0)[cty_b == 'US'].mean():.3f} | "
          f"India {100 * (fk - f0)[cty_b == 'India'].mean():.3f} | S1 affected {len(v):,}", flush=True)

# ---- T2 per-country decision (tuned on half-A S1 of each country, applied to that country's records)
dec_c, parts = {}, []
for c in ("US", "India"):
    m = rc == c
    idsA_c = ids_a[cty_of.reindex(ids_a).values == c]
    dc, fa = S.tune(d[m].reset_index(drop=True), d[base_col].values[m], r, s1, truth, idsA_c)
    dec_c[c] = dc
    log(f"T2 {c} decision tuned on half A: {dc} (half A {fa:.4f})")
    parts.append(S._fast_decider(d[m].reset_index(drop=True), d[base_col].values[m], r)(dc))
P2 = er.to_sets(pd.concat(parts, ignore_index=True), s1, r)
report("T2 per-country decision", per_s1(P2, ids_b), f0)

# ---- T3 stage-2 probability variants, each re-tuned on half A (all countries)
cands = {}
for c in ("p2", "p2n", "p1"):
    if c in d.columns:
        cands[c] = d[c].values
if "p2" in d.columns:
    cands["mean(p2a,p2)"] = (d.p2a.values + d.p2.values) / 2
if "p2n" in d.columns:
    cands["mean(p2a,p2n)"] = (d.p2a.values + d.p2n.values) / 2
if "p1" in d.columns:
    cands["0.8*p2a+0.2*p1"] = 0.8 * d.p2a.values + 0.2 * d.p1.values
for name, pr in cands.items():
    dc, fa = S.tune(d, pr, r, s1, truth, ids_a)
    report(f"T3 {name} (tuned {dc['t']}, rescue {dc['rescue']})", per_s1(preds(pr, dc), ids_b), f0)

# ---- T4 cap per S1
for cap in (8, 9, 10):
    report(f"T4 cap {cap}", per_s1(preds(d[base_col].values, {**dec0, "cap": cap}), ids_b), f0)
log("E28 DONE")
