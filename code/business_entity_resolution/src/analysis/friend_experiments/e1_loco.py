"""E1: full-scale leave-one-country-out for v6 candidate fixes, one SOURCE country per machine.
Train on all of split A of the source country (buckets [2,51)), evaluate on all of split B (buckets [51,100)) of both
countries. Stage-1 LightGBM, v4 features (v5 parallel code, proven identical). Pair-level F0.5 per target.
Configs (each vs 'base' seed 0, paired bootstrap over entity clusters):
  base (seeds 0,1,2)  v4 features, v4 parameters                       -> seed spread
  reg                 stronger regularization (full-scale pass on 26 Sep)
  fmt                 format-independent address/number features (no raw lengths / raw number gaps)
  inv                 per-country word rarity (IDF), relative counts, per-country generic words
  inv+fmt+reg         all three together
Saves, for offline analysis (threshold rule, structural filter): eval pair ids, labels, structural columns and every
config's probabilities.  Run from student_resource/:
  python e1_loco.py --data dataset --work ~/w4 --src ~/src5 --source US --out ~/e1"""
import argparse
import math
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="dataset")
ap.add_argument("--work", required=True)
ap.add_argument("--src", required=True)
ap.add_argument("--source", choices=["US", "India"], required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--boot", type=int, default=300)
ap.add_argument("--configs", default="base,base_s1,base_s2,reg,fmt,inv,inv+fmt+reg")
ap.add_argument("--base-draws", default="", help="E1 draws_<SRC>.npz: paired diffs vs its saved base (same resamples)")
ap.add_argument("--tag", default="")
args = ap.parse_args()
sys.path.insert(0, args.src)
from rapidfuzz import fuzz  # noqa: E402
from rapidfuzz.process import cpdist  # noqa: E402

import er  # noqa: E402
import stage2 as S  # noqa: E402
from normalize import core_name  # noqa: E402

t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
os.makedirs(args.out, exist_ok=True)
W, SRC = args.work, args.source
F0, F1, E0, E1 = 2, 51, 51, 100
s1, r = er.load_split(args.data, "train", W)
truth, r2s = er.load_truth(args.data)
rkey = r.entity_id.map(r2s).fillna(r.entity_id).values
rb = er.id_bucket(rkey)
p = pd.read_parquet(f"{W}/loco_pairs_{F0}_{E1}.parquet")  # all post-prefilter pairs of buckets [2,100), cached 26 Sep
y = p.y.values.astype(np.int8)
country = r.country.values[p.q.values]
role = np.where(rb[p.q.values] < F1, "FIT", "EVAL")
half = rb[p.q.values] % 2
fitm = (role == "FIT") & (country == SRC)
evm = role == "EVAL"
log(f"pairs {len(p):,} | fit ({SRC}) {int(fitm.sum()):,} | eval {int(evm.sum()):,}")
rn, sn = r.name.values[p.q.values], s1.name.values[p.s.values]
rt, st = r["tail"].values[p.q.values], s1["tail"].values[p.s.values]
tset = cpdist(list(rn), list(sn), scorer=fuzz.token_set_ratio, workers=-1)
tables = lambda m: er._label_tables(rn[m], sn[m], rt[m], st[m], y[m], tset[m])

# ---- statistics variants (all unsupervised; the same computation would run on test)
def idf_of(ms, mr):
    df = Counter()
    for col in (s1.name.values[ms], r.name.values[mr]):
        for x in col:
            df.update(set(x.split()))
    n = int(ms.sum() + mr.sum())
    return {"idf": {w: math.log(n / c) for w, c in df.items() if c > 1}, "default": math.log(n)}


def ctx_rel():
    out = {}
    for name, d, col in (("s1_name_freq", s1, "name"), ("r_name_freq", r, "name"), ("s1_addr_freq", s1, "addr")):
        key = d.country + "|" + d[col]
        v = key.map(key.value_counts()).values.astype(np.float32)
        out[name] = v / d.country.map(d.country.value_counts()).values.astype(np.float32) * 1e6
    return out


BASE_STATS = (er.fit_idf(s1, r), er.split_context(s1, r), s1.name_core.values.copy(), r.name_core.values.copy())
ALL_S, ALL_R = np.ones(len(s1), bool), np.ones(len(r), bool)
IDF_C = {c: idf_of(s1.country.values == c, r.country.values == c) for c in ("US", "India")}
core_c_s, core_c_r = s1.name_core.values.copy(), r.name_core.values.copy()
for c in ("US", "India"):  # generic words decided within each country
    ms, mr = s1.country.values == c, r.country.values == c
    df = Counter()
    for x in list(s1.name.values[ms]) + list(r.name.values[mr]):
        df.update(set(x.split()))
    n_c = int(ms.sum() + mr.sum())  # computed once (it was re-summed for every word)
    gen = {w for w, k in df.items() if k / n_c > er.GENERIC_DF}
    core_c_s[ms] = [core_name(x, gen) for x in s1.name.values[ms]]
    core_c_r[mr] = [core_name(x, gen) for x in r.name.values[mr]]
INV_STATS = (IDF_C, ctx_rel(), core_c_s, core_c_r)
log("statistics ready")


def feats(m, lo, loc, stats):
    idf, ctx, cs, cr = stats
    s1["name_core"], r["name_core"] = cs, cr
    rows = np.flatnonzero(m)
    if "idf" in idf:
        X = er.features(p.iloc[rows].reset_index(drop=True), s1, r, lo, idf, loc, ctx).set_index(rows)
    else:  # per-country IDF: compute each country's rows with its own table
        X = pd.concat([er.features(p.iloc[rc].reset_index(drop=True), s1, r, lo, idf[c], loc, ctx).set_index(rc)
                       for c in ("US", "India") for rc in [rows[country[rows] == c]] if len(rc)]).sort_index()
    s1["name_core"], r["name_core"] = BASE_STATS[2], BASE_STATS[3]
    return X


def add_fmt(X):
    """Format-independent versions: no raw lengths or raw number gaps (these separate France from US train perfectly)."""
    X = X.copy()
    X["addr_len_ratio"] = (np.minimum(X.r_addr_len, X.s_addr_len) / np.maximum(np.maximum(X.r_addr_len, X.s_addr_len), 1)).astype(np.float32)
    d = X.num_first_delta.values
    X["num_delta_class"] = np.where(np.isnan(d), -1, np.where(d == 0, 0, np.where(d <= 20, 1, 2))).astype(np.float32)
    return X.drop(columns=["r_addr_len", "s_addr_len", "r_name_len", "s_name_len", "num_first_delta", "num_min_delta"])


REG = dict(num_leaves=31, min_data_in_leaf=1000, feature_fraction=0.5, lambda_l2=10.0)
CFG = {"base": ("base", False, {}, 0), "base_s1": ("base", False, {}, 1), "base_s2": ("base", False, {}, 2),
       "reg": ("base", False, REG, 0), "fmt": ("base", True, {}, 0), "inv": ("inv", False, {}, 0),
       "inv+fmt+reg": ("inv", True, REG, 0), "inv+reg": ("inv", False, REG, 0)}
configs = args.configs.split(",")

# ---- evaluation: pair-level outcome per EVAL record of each target country; cluster bootstrap
s1_pos = pd.Series(np.arange(len(s1)), index=s1.entity_id.values)
true_s = r.entity_id.map(r2s).map(s1_pos).fillna(-1).astype(np.int64).values
ev_rec = {c: np.flatnonzero((r.country.values == c) & (rb >= E0) & (rb < E1)) for c in ("US", "India")}
clusters = {c: pd.factorize(rkey[ev_rec[c]])[0] for c in ev_rec}
rng = np.random.default_rng(0)
WB = {c: rng.poisson(1.0, size=(args.boot, clusters[c].max() + 1)).astype(np.float32) for c in ev_rec}
grid = np.round(np.arange(0.02, 0.99, 0.02), 2)


def outcomes(tgt, prob):
    m = evm & (country == tgt)
    run = S._fast_decider(p.loc[m, ["q", "s"]], prob[m], r)
    res = {}
    for t in grid:
        a = run({"t": {k: float(t) for k in S.SLICES}, "rescue": 0.0})
        asg = np.full(len(r), -2, np.int64); asg[a.q.values] = a.s.values
        got, tru = asg[ev_rec[tgt]], true_s[ev_rec[tgt]]
        tp = (got >= 0) & (got == tru)
        res[t] = (tp, (got >= 0) & ~tp, (tru >= 0) & ~tp)
    return res


def f05(tp, fp, fn):
    P, R = tp / np.maximum(tp + fp, 1e-9), tp / np.maximum(tp + fn, 1e-9)
    return 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12)


def draws(tgt, o):
    cl = clusters[tgt]
    tp, fp, fn = (WB[tgt] @ np.bincount(cl, weights=x.astype(np.float32), minlength=cl.max() + 1) for x in o)
    return f05(tp, fp, fn)


# ---- saved once: eval pairs + structural columns used by the offline rules
Xcache = {}
rows = []
D = {}
for name in configs:
    stats_name, fmt, reg, seed = CFG[name]
    if stats_name not in Xcache:
        stats = BASE_STATS[:2] + BASE_STATS[2:] if stats_name == "base" else INV_STATS
        Xf = pd.concat([feats(fitm & (half == h), *tables(fitm & (half != h)), stats) for h in (0, 1)]).sort_index()
        Xe = feats(evm, *tables(fitm), stats)
        Xcache[stats_name] = (Xf, Xe)
        log(f"features [{stats_name}] fit {Xf.shape} eval {Xe.shape}")
        if stats_name == "base" and not os.path.exists(f"{args.out}/eval_pairs.parquet"):
            ev = p.loc[evm, ["q", "s"]].assign(y=y[evm], country=country[evm])
            for c in ("num_first_eq", "num_small_shift", "pcode_eq", "codes_eq", "a_tset", "n_tset", "sc_addr", "sc_name"):
                ev[c] = Xe[c].values
            ev.to_parquet(f"{args.out}/eval_pairs.parquet")
    Xf, Xe = Xcache[stats_name]
    if fmt:
        Xf, Xe = add_fmt(Xf), add_fmt(Xe)
    params = dict(er.LGB_PARAMS, **reg, seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
    inner = er.id_bucket(r.entity_id.values[p.q.values[fitm]]) % 10 == 0
    m = er.train_lgb(Xf, y[fitm], inner, params=params)
    prob = np.zeros(len(p), np.float32); prob[evm] = m.predict(Xe[Xf.columns], num_threads=0)
    np.save(f"{args.out}/prob{args.tag}_{SRC}_{name.replace('+', '_')}.npy", prob[evm])
    oc = {tgt: outcomes(tgt, prob) for tgt in ("US", "India")}
    curve = {tgt: {t: f05(*[x.sum() for x in oc[tgt][t]]) for t in grid} for tgt in oc}
    t_src = max(grid, key=lambda t: curve[SRC][t])
    for tgt in ("US", "India"):
        t_or = max(grid, key=lambda t: curve[tgt][t])
        D[(tgt, name)] = draws(tgt, oc[tgt][t_src])
        tp, fp, fn = (x.sum() for x in oc[tgt][t_src])
        rows.append(dict(source=SRC, target=tgt, config=name, trees=m.num_trees(), F05=curve[tgt][t_src],
                         ci_lo=np.quantile(D[(tgt, name)], .025), ci_hi=np.quantile(D[(tgt, name)], .975),
                         precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1), t_src=t_src,
                         F05_oracle=curve[tgt][t_or], t_oracle=t_or))
    log(f"{name}: " + " | ".join(f"{x['target']} {x['F05']:.4f} [{x['ci_lo']:.4f}, {x['ci_hi']:.4f}] t={x['t_src']}" for x in rows[-2:]))
    pd.DataFrame(rows).to_csv(f"{args.out}/e1{args.tag}_{SRC}.csv", index=False)

pd.set_option("display.width", 250)
print(f"\n=== E1 source {SRC}: pair-level F0.5 (95% CI), threshold chosen on {SRC} ===")
print(pd.DataFrame(rows).round(4).to_string(index=False))
if args.base_draws:  # base from the saved E1 run (identical data, clusters and bootstrap weights)
    z = np.load(args.base_draws)
    for tgt in ("US", "India"):
        D[(tgt, "base")] = z[f"{tgt}|base"]
    configs = ["base"] + configs
print("\n=== paired differences vs base (seed 0), 95% CI ===")
for tgt in ("US", "India"):
    for name in configs[1:]:
        dd = D[(tgt, name)] - D[(tgt, "base")]
        print(f"{SRC}->{tgt:5s} {name:12s}: {dd.mean():+.4f}  CI [{np.quantile(dd, .025):+.4f}, {np.quantile(dd, .975):+.4f}]")
seeds = [n for n in ("base", "base_s1", "base_s2") if n in configs and any(x["config"] == n for x in rows)]
for tgt in ("US", "India"):
    v = [rows_[ "F05"] for rows_ in rows if rows_["target"] == tgt and rows_["config"] in seeds]
    if not v:
        continue
    print(f"seed spread {SRC}->{tgt}: {', '.join(f'{x:.4f}' for x in v)}  (range {max(v) - min(v):.4f})")
np.savez(f"{args.out}/draws{args.tag}_{SRC}.npz", **{"|".join(k): v for k, v in D.items()})
log("E1 done")
