"""Diagnosis 4 (generic words per country), built on diagnosis 2: leave-one-country-out (LOCO) on TRAIN labels, to explain v4's France drop fairly.
Train v4's stage-1 model (same features, same LightGBM) on one country and evaluate on the other, next to the
in-country model, and ablate the label-derived feature groups that cannot transfer to an unseen country.
Only train data is used. Run from student_resource/:
    python loco_diag.py --data dataset --work ~/w4 --src ~/src
Buckets are entity-hash buckets (0-99) of the record's true S1 (or the record itself): FIT [2,8), EVAL [8,11)."""
import argparse
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="dataset")
ap.add_argument("--work", required=True)
ap.add_argument("--src", required=True)
ap.add_argument("--fit", type=int, nargs=2, default=(2, 8))
ap.add_argument("--eval", type=int, nargs=2, default=(8, 11))
ap.add_argument("--generic", choices=["split", "country"], default="country")
args = ap.parse_args()
sys.path.insert(0, args.src)
from rapidfuzz import fuzz  # noqa: E402
from rapidfuzz.process import cpdist  # noqa: E402

import er  # noqa: E402
import stage2 as S  # noqa: E402

t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:7.0f}s]", *a, flush=True)
W = args.work
s1, r = er.load_split(args.data, "train", W)
truth, r2s = er.load_truth(args.data)
rb = er.id_bucket(r.entity_id.map(r2s).fillna(r.entity_id).values)
if args.generic == "country":  # generic = in more than GENERIC_DF of the names of the record's own country
    from collections import Counter
    from normalize import core_name
    for c in sorted(set(s1.country) | set(r.country)):
        ms, mr = s1.country.values == c, r.country.values == c
        df = Counter()
        for x in list(s1.name.values[ms]) + list(r.name.values[mr]):
            df.update(set(x.split()))
        n = int(ms.sum() + mr.sum())
        gen = {w for w, k in df.items() if k / n > er.GENERIC_DF}
        s1.loc[ms, "name_core"] = [core_name(x, gen) for x in s1.name.values[ms]]
        r.loc[mr, "name_core"] = [core_name(x, gen) for x in r.name.values[mr]]
        print(f"{c}: {len(gen)} generic words (per country)", flush=True)
    s1["core_ns"] = s1.name_core.str.replace(" ", "", regex=False)
    r["core_ns"] = r.name_core.str.replace(" ", "", regex=False)
F0, F1 = args.fit
E0, E1 = args.eval
want = (rb >= F0) & (rb < E1)

# ---- post-prefilter pairs (train_scored) with their channel scores (train_pairs) for the sampled records
import os
CACHE = f"{W}/loco_pairs_{F0}_{E1}.parquet"
if os.path.exists(CACHE):
    p = pd.read_parquet(CACHE)
else:
    ts = pd.read_parquet(f"{W}/train_scored.parquet", columns=["q", "s", "y"])
    ts = ts[want[ts.q.values]]
    pf = pq.ParquetFile(f"{W}/train_pairs.parquet")
    parts = []
    for g in range(pf.num_row_groups):
        t = pf.read_row_group(g).to_pandas()
        parts.append(t[want[t.q.values]])
    p = pd.concat(parts, ignore_index=True).merge(ts, on=["q", "s"], how="inner").sort_values(["q", "s"], ignore_index=True)
    del parts
    p.to_parquet(CACHE)
y = p.y.values.astype(np.int8)
country = r.country.values[p.q.values]
bucket = rb[p.q.values]
role = np.where(bucket < F1, "FIT", "EVAL")
half = bucket % 2
log(f"pairs {len(p):,} | FIT {int((role == 'FIT').sum()):,} | EVAL {int((role == 'EVAL').sum()):,} | "
    f"countries {pd.Series(country).value_counts().to_dict()}")

rn, sn = r.name.values[p.q.values], s1.name.values[p.s.values]
rt, st = r["tail"].values[p.q.values], s1["tail"].values[p.s.values]
tset = cpdist(list(rn), list(sn), scorer=fuzz.token_set_ratio, workers=-1)
idf, ctx = er.fit_idf(s1, r), er.split_context(s1, r)
tables = lambda m: er._label_tables(rn[m], sn[m], rt[m], st[m], y[m], tset[m])


def feats(mask, lo, loc):
    return er.features(p[mask].reset_index(drop=True), s1, r, lo, idf, loc, ctx).set_index(np.flatnonzero(mask))


ABLATE = {"full": [], "-both label tables": ["extra_tok_logodds_sum", "extra_tok_logodds_max", "loc_sub_logodds"]} if args.generic == "country" else {"full": [], "-wordlogodds": ["extra_tok_logodds_sum", "extra_tok_logodds_max"],
          "-locality": ["loc_sub_logodds"],
          "-both label tables": ["extra_tok_logodds_sum", "extra_tok_logodds_max", "loc_sub_logodds"],
          "-colocation ctx": ["s1_addr_freq", "s1_name_freq", "r_name_freq"]}
evm = role == "EVAL"
res = []
for src in ("US", "India"):
    fitm = (role == "FIT") & (country == src)
    Xfit = pd.concat([feats(fitm & (half == h), *tables(fitm & (half != h))) for h in (0, 1)]).sort_index()
    Xev = feats(evm, *tables(fitm))
    log(f"source {src}: features fit {Xfit.shape} eval {Xev.shape}")
    inner = er.id_bucket(r.entity_id.values[p.q.values[fitm]]) % 10 == 0
    for name, drop in ABLATE.items():
        cols = [c for c in Xfit.columns if c not in drop]
        m = er.train_lgb(Xfit[cols], y[fitm], inner)
        pe = m.predict(Xev[cols], num_threads=0)
        prob = np.zeros(len(p)); prob[evm] = pe
        f_at = {}
        for tgt in ("US", "India"):
            tm = evm & (country == tgt)
            d = p.loc[tm, ["q", "s"]]
            ids = s1.entity_id.values[(s1.country.values == tgt) & (er.id_bucket(s1.entity_id.values) >= E0)
                                      & (er.id_bucket(s1.entity_id.values) < E1)]
            th = {i: truth[i] for i in ids}
            run = S._fast_decider(d, prob[tm], r)
            keep = s1.entity_id.isin(set(ids)).values
            curve = {}
            for t in np.arange(0.1, 0.96, 0.025):
                a = run({"t": {k: float(t) for k in S.SLICES}, "rescue": 0.0})
                curve[round(float(t), 3)] = er.macro_f05(er.to_sets(a[keep[a.s.values]], s1, r), th, ids)
            f_at[tgt] = curve
        for tgt in ("US", "India"):
            t_src = max(f_at[src], key=f_at[src].get)  # threshold chosen on the source country
            t_best = max(f_at[tgt], key=f_at[tgt].get)
            res.append(dict(source=src, target=tgt, config=name, trees=m.num_trees(), f_src_threshold=f_at[tgt][t_src],
                            f_oracle=f_at[tgt][t_best], t_src=t_src, t_oracle=t_best))
        log(f"  {name}: " + " | ".join(f"{x['target']} {x['f_src_threshold']:.4f}" for x in res[-2:]))

R = pd.DataFrame(res)
pd.set_option("display.width", 220)
print("\n=== LOCO stage-1 results (macro F0.5 on EVAL S1; threshold chosen on the source country / best possible) ===")
print(R.round(4).to_string(index=False))
R.to_csv(f"{W}/loco_diag_{args.generic}.csv", index=False)
