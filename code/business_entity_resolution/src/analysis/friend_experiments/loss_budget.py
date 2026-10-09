"""Diagnoses 10-12: v4 loss budget on held-out H in F0.5 points, with 95% bootstrap CIs.
For each error type, replace v4's decision by the true answer for exactly those records (oracle fix) and measure the
macro F0.5 gain: that is the most a perfect fix of that problem can add. Done overall and inside slices.
Then anatomy of shortlist misses (D11) and of no-address errors (D12). Uses v4's exact pass-1 probabilities (p2a).
Run from student_resource/:  python loss_budget.py --data dataset --work ~/w4 --src ~/src"""
import argparse
import json
import sys
import unicodedata

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="dataset")
ap.add_argument("--work", required=True)
ap.add_argument("--src", required=True)
ap.add_argument("--boot", type=int, default=1000)
args = ap.parse_args()
sys.path.insert(0, args.src)
import er  # noqa: E402
import stage2 as S  # noqa: E402

W = args.work
s1, r = er.load_split(args.data, "train", W)
truth, r2s = er.load_truth(args.data)
meta = json.load(open(f"{W}/meta.json"))
d = pd.read_parquet(f"{W}/train_stage2.parquet")
assert meta["stage2_choice"] == "pass1" and "p2a" in d.columns
a = S.decide(d, d.p2a.values, meta["decision"], r)
h_ids = s1.entity_id.values[er.split_name(er.id_bucket(s1.entity_id.values)) == "H"]
h_set = set(h_ids)
s1_pos = pd.Series(np.arange(len(s1)), index=s1.entity_id.values)
true_s = r.entity_id.map(r2s).map(s1_pos).fillna(-1).astype(np.int64).values
asg = np.full(len(r), -1, np.int64); asg[a.q.values] = a.s.values
in_h_s1 = np.zeros(len(s1), bool); in_h_s1[s1_pos[h_ids].values] = True
# records that matter for H: true S1 in H, or assigned to an H S1
recs = np.flatnonzero(((true_s >= 0) & in_h_s1[np.maximum(true_s, 0)]) | ((asg >= 0) & in_h_s1[np.maximum(asg, 0)]))
cand = d[d.q.isin(recs)]
has_true = set(zip(cand.q.values, cand.s.values))
top1 = cand.sort_values(["q", "p2a"], ascending=[True, False]).drop_duplicates("q").set_index("q").s
out = []
for q in recs:
    t, g = true_s[q], asg[q]
    if t >= 0 and g == t:
        out.append("correct")
    elif t >= 0 and (q, t) not in has_true:
        out.append("FN not in candidates" if g < 0 else "FN not in candidates + FP")
    elif t >= 0 and top1.get(q, -1) != t:
        out.append("ranking (true S1 not top-1)" + (" + FP" if g >= 0 else ""))
    elif t >= 0:
        out.append("FN rejected (below threshold)")
    elif g >= 0:
        out.append("FP unmatched record assigned")
    else:
        out.append("correct")
R = pd.DataFrame({"q": recs, "true": true_s[recs], "asg": asg[recs], "outcome": out})
R["country"] = r.country.values[recs]
R["address"] = np.where(r.addr.values[recs] == "", "empty", "present")
R["script"] = np.where(r.nonlatin.values[recs], "native", "latin")
R["source"] = r.entity_id.str[:2].values[recs]
ids = np.array(sorted(h_set))
idx = pd.Series(np.arange(len(ids)), index=ids)
T = [truth[i] for i in ids]


def per_s1(asg_map):
    """Per-S1 F0.5 for all H S1 given a record->S1 map (only records in R)."""
    pred = {}
    for q, s in asg_map.items():
        if s >= 0 and in_h_s1[s]:
            pred.setdefault(s1.entity_id.values[s], set()).add(r.entity_id.values[q])
    f = np.empty(len(ids))
    for k, i in enumerate(ids):
        P, Tt = pred.get(i, set()), T[k]
        if not Tt:
            f[k] = float(not P)
        else:
            tp = len(P & Tt)
            f[k] = 0.0 if not tp else 1.25 * (tp / len(P)) * (tp / len(Tt)) / (0.25 * tp / len(P) + tp / len(Tt))
    return f


base_map = dict(zip(R.q.values, R.asg.values))
f0 = per_s1(base_map)
rng = np.random.default_rng(0)
B = rng.integers(0, len(ids), size=(args.boot, len(ids)))
ci = lambda v: (v.mean(), np.quantile(v[B].mean(axis=1), 0.025), np.quantile(v[B].mean(axis=1), 0.975))
m, lo, hi = ci(f0)
print(f"v4 on all H ({len(ids):,} S1): macro F0.5 {m:.4f}  95% CI [{lo:.4f}, {hi:.4f}]\n")
print("=== D10: loss budget = gain if this error type were fixed perfectly (F0.5 points x100, 95% CI) ===")
rows = []
for slice_col in (None, "country", "address", "script", "source"):
    groups = [("all", R.index)] if slice_col is None else list(R.groupby(slice_col).groups.items())
    for gname, gidx in groups:
        Rg = R.loc[gidx]
        for typ in sorted(set(R.outcome) - {"correct"}):
            sel = Rg[Rg.outcome == typ]
            if not len(sel):
                continue
            mp = dict(base_map)
            for q, t in zip(sel.q.values, sel.true.values):
                mp[q] = t  # the true answer: its true S1, or no match (-1)
            gain = per_s1(mp) - f0
            g, glo, ghi = ci(gain)
            rows.append(dict(slice=slice_col or "all", group=gname, error=typ, records=len(sel), gain_pts=100 * g,
                             ci_lo=100 * glo, ci_hi=100 * ghi))
L = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(L[L.slice == "all"].sort_values("gain_pts", ascending=False).round(3).to_string(index=False))
print("\n--- by slice (top 25 by gain) ---")
print(L[L.slice != "all"].sort_values("gain_pts", ascending=False).head(25).round(3).to_string(index=False))
L.to_csv(f"{W}/loss_budget.csv", index=False)


def script_of(x):
    for ch in x:
        if ord(ch) > 0x24F and ch.isalpha():
            return unicodedata.name(ch, "?").split()[0]
    return "LATIN"


miss = R[R.outcome.str.startswith("FN not in candidates")]
print(f"\n=== D11: shortlist misses on H: {len(miss):,} records ===")
print(miss.assign(scr=[script_of(x) for x in r.raw_name.values[miss.q.values]]).groupby(["country", "address", "scr"]).size()
      .sort_values(ascending=False).head(15).to_string())
for q in miss.sample(min(12, len(miss)), random_state=0).q.values:
    t = true_s[q]
    print(f"REC: {r.raw_name.values[q]} | {r.raw_addr.values[q]}\n S1: {s1.raw_name.values[t]} | {s1.raw_addr.values[t]}\n")
na = R[(R.address == "empty") & (R.true >= 0)]
print(f"=== D12: no-address records with a true S1 on H: {len(na):,}; outcomes ===")
print(na.outcome.value_counts().to_string())
nf = s1.name.map(s1.name.value_counts()).values
print("   share whose true S1 name is unique in train S1:", round(float((nf[na.true.values] == 1).mean()), 4))
for q in na[na.outcome != "correct"].sample(min(10, int((na.outcome != 'correct').sum())), random_state=0).q.values:
    t, g = true_s[q], asg[q]
    print(f"REC: {r.raw_name.values[q]} | (no address)\n TRUE: {s1.raw_name.values[t]} | {s1.raw_addr.values[t]}"
          + (f"\n PRED: {s1.raw_name.values[g]} | {s1.raw_addr.values[g]}" if g >= 0 else "\n PRED: (none)") + "\n")
