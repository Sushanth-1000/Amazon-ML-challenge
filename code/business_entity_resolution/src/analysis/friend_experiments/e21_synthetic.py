"""E21: synthetic match examples built from the UNLABELLED country's own S1 records (full-scale leave-one-country-out,
same data/splits/bootstrap as E1/E19). The SOURCE country is labelled; the TARGET country plays France: its labels are
never used for training, table fitting, threshold choice or early stopping, only for the final score.
  reg : v4 features + REG parameters, source only (= E1/E19 'reg', reproduced)
  syn : + synthetic target records. For N target S1 (EVAL buckets: the evaluated S1, as all French test S1 would be),
        one variant record each, made by edits whose frequencies are MEASURED on the source country's true matches
        (name exact / legal form / dropped word / added word / word swap / typo; address empty / exact / cut / house
        number dropped / typo). Added words and legal forms are sampled from the target's own S1 names (unsupervised).
        Variants go through the real normalizer, candidate search (reverse-S1 step off: it finds nothing on a full
        split), v4 prefilter and features; label 1 for the source S1, 0 for the other candidates. Decoy/locality label
        tables are re-learned with the synthetic labels (cross-fitted by S1 halves); target pairs get tables from source
        + all synthetic rows. Early stopping on source labels only.
Threshold = best on the SOURCE. Paired bootstrap vs reg (same resamples) and vs E1 base (--base-draws).
Usage: python e21_synthetic.py --data dataset --work ~/w --src ~/src --source US --out ~/e21 --base-draws draws_US.npz"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="dataset")
ap.add_argument("--work", required=True)
ap.add_argument("--src", required=True)
ap.add_argument("--source", choices=["US", "India"], required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--boot", type=int, default=300)
ap.add_argument("--base-draws", default="")
ap.add_argument("--nsyn", type=int, default=200000)
ap.add_argument("--hi", type=float, default=0.9)
ap.add_argument("--lo", type=float, default=0.05)
ap.add_argument("--margin", type=float, default=0.3)
ap.add_argument("--w", type=float, default=0.5)
ap.add_argument("--smoke", action="store_true", help="5%% of entities, for a quick correctness run")
args = ap.parse_args()
sys.path.insert(0, args.src)
from rapidfuzz import fuzz  # noqa: E402
from rapidfuzz.process import cpdist  # noqa: E402

import er  # noqa: E402
import stage2 as S  # noqa: E402

t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
os.makedirs(args.out, exist_ok=True)
W, SRC = args.work, args.source
TGT = "India" if SRC == "US" else "US"
F0, F1, E0, E1 = 2, 51, 51, 100
s1, r = er.load_split(args.data, "train", W)
truth, r2s = er.load_truth(args.data)
rkey = r.entity_id.map(r2s).fillna(r.entity_id).values
rb = er.id_bucket(rkey)
p = pd.read_parquet(f"{W}/loco_pairs_{F0}_{E1}.parquet")
RSEL = np.isin(rb, [20, 21, 22, 23, 60, 61, 62, 63]) if args.smoke else np.ones(len(r), bool)  # both halves/quarters
if args.smoke:
    p = p[RSEL[p.q.values]].reset_index(drop=True)
y = p.y.values.astype(np.int8)
country = r.country.values[p.q.values]
role = np.where(rb[p.q.values] < F1, "FIT", "EVAL")
half = rb[p.q.values] % 2
quarter = (rb[p.q.values] // 2) % 2
fitm = (role == "FIT") & (country == SRC)
evm = role == "EVAL"
mt = evm & (country == TGT)          # the "unlabelled country" pairs
ms = evm & (country == SRC)
log(f"pairs {len(p):,} | fit ({SRC}) {int(fitm.sum()):,} | eval {int(evm.sum()):,} | target {TGT} eval {int(mt.sum()):,}")
rn, sn = r.name.values[p.q.values], s1.name.values[p.s.values]
rt, st = r["tail"].values[p.q.values], s1["tail"].values[p.s.values]
tset = cpdist(list(rn), list(sn), scorer=fuzz.token_set_ratio, workers=-1)
IDF, CTX = er.fit_idf(s1, r), er.split_context(s1, r)
log("statistics ready")


def tables(m, lab):
    return er._label_tables(rn[m], sn[m], rt[m], st[m], lab[m], tset[m])


def feats(m, lo, loc):
    rows = np.flatnonzero(m)
    return er.features(p.iloc[rows].reset_index(drop=True), s1, r, lo, IDF, loc, CTX).set_index(rows)


# ---- evaluation (identical to E1)
s1_pos = pd.Series(np.arange(len(s1)), index=s1.entity_id.values)
true_s = r.entity_id.map(r2s).map(s1_pos).fillna(-1).astype(np.int64).values
ev_rec = {c: np.flatnonzero((r.country.values == c) & (rb >= E0) & (rb < E1) & RSEL) for c in ("US", "India")}
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


rows_out, D = [], {}


def report(name, prob, extra=""):
    oc = {tgt: outcomes(tgt, prob) for tgt in ("US", "India")}
    curve = {tgt: {t: f05(*[x.sum() for x in oc[tgt][t]]) for t in grid} for tgt in oc}
    t_src = max(grid, key=lambda t: curve[SRC][t])
    for tgt in ("US", "India"):
        t_or = max(grid, key=lambda t: curve[tgt][t])
        D[(tgt, name)] = draws(tgt, oc[tgt][t_src])
        tp, fp, fn = (x.sum() for x in oc[tgt][t_src])
        rows_out.append(dict(source=SRC, target=tgt, config=name, F05=curve[tgt][t_src],
                             ci_lo=np.quantile(D[(tgt, name)], .025), ci_hi=np.quantile(D[(tgt, name)], .975),
                             precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1), t_src=t_src,
                             F05_oracle=curve[tgt][t_or], t_oracle=t_or))
    log(f"{name}{extra}: " + " | ".join(f"{x['target']} {x['F05']:.4f} [{x['ci_lo']:.4f}, {x['ci_hi']:.4f}] t={x['t_src']}"
                                         for x in rows_out[-2:]))
    pd.DataFrame(rows_out).to_csv(f"{args.out}/e21_{SRC}.csv", index=False)
    np.save(f"{args.out}/prob_{SRC}_{name}.npy", prob[evm])


# ---- round 0: source-only model (E1 'reg')
REG = dict(num_leaves=31, min_data_in_leaf=1000, feature_fraction=0.5, lambda_l2=10.0)
params = dict(er.LGB_PARAMS, **REG, seed=0, bagging_seed=0, feature_fraction_seed=0)
Xf = pd.concat([feats(fitm & (half == h), *tables(fitm & (half != h), y)) for h in (0, 1)]).sort_index()
Xe = feats(evm, *tables(fitm, y))
log(f"features fit {Xf.shape} eval {Xe.shape}")
inner_f = er.id_bucket(r.entity_id.values[p.q.values[fitm]]) % 10 == 0   # early stopping on SOURCE labels only
m0 = er.train_lgb(Xf, y[fitm], inner_f, params=params)
prob = np.zeros(len(p), np.float32); prob[evm] = m0.predict(Xe[Xf.columns], num_threads=0)
report("reg", prob)
cols = list(Xf.columns)



# ---- measured edit frequencies on the SOURCE country's true matches (FIT rows only)
import json  # noqa: E402
import random  # noqa: E402
from collections import Counter  # noqa: E402

import lightgbm as lgb  # noqa: E402
import scipy.sparse as sps  # noqa: E402
from normalize import LEGAL, core_name  # noqa: E402

tp = np.flatnonzero(fitm & (y == 1))
A_n, B_n = sn[tp], rn[tp]
A_a, B_a = s1.addr.values[p.s.values[tp]], r.addr.values[p.q.values[tp]]


def name_kind(a, b):
    if a == b:
        return "exact"
    ta, tb = [w for w in a.split() if w not in LEGAL], [w for w in b.split() if w not in LEGAL]
    if ta == tb:
        return "legal"
    sa, sb = set(ta), set(tb)
    if sb < sa:
        return "drop"
    if sa < sb:
        return "add"
    if sorted(ta) == sorted(tb):
        return "swap"
    if len(ta) == len(tb) and fuzz.ratio(" ".join(ta), " ".join(tb)) >= 80:
        return "typo"
    return "other"


def addr_kind(a, b):
    if b == "":
        return "empty"
    if a == b:
        return "exact"
    if a.startswith(b) or b.startswith(a):
        return "cut"
    na, nb = [w for w in a.split() if w.isdigit()], [w for w in b.split() if w.isdigit()]
    if na and na[0] not in nb:
        return "nonum"
    return "other"


NK = pd.Series([name_kind(a, b) for a, b in zip(A_n, B_n)]).value_counts(normalize=True)
AK = pd.Series([addr_kind(a, b) for a, b in zip(A_a, B_a)]).value_counts(normalize=True)
log(f"source true pairs {len(tp):,} | name edits {NK.round(4).to_dict()} | address edits {AK.round(4).to_dict()}")

# ---- target S1 to copy: the evaluated target S1 (EVAL buckets; smoke: only its sample buckets)
sb_all = er.id_bucket(s1.entity_id.values)
in_pool = (s1.country.values == TGT) & (sb_all >= E0) & (sb_all < E1)
if args.smoke:
    in_pool &= np.isin(sb_all, [60, 61, 62, 63])
pool = np.flatnonzero(in_pool)
src_rows = np.sort(np.random.default_rng(1).choice(pool, size=min(args.nsyn, len(pool)), replace=False))
tok_df = pd.Series([w for x in s1.raw_name.values[s1.country.values == TGT] for w in set(str(x).split())]).value_counts()
is_legal = np.array([w.lower().strip(".,") in LEGAL for w in tok_df.index])
legal_t, gen_t = tok_df[is_legal].head(20), tok_df[~is_legal].head(50)
rnd = random.Random(7)
pick = lambda ser: rnd.choices(list(ser.index), weights=list(ser.values))[0]


def typo(w):
    if len(w) < 4:
        return w
    i, op = rnd.randrange(1, len(w) - 2), rnd.randrange(3)
    if op == 0:
        return w[:i] + w[i + 1:]                       # deletion
    if op == 1:
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]     # swap of neighbours
    return w[:i] + w[i] + w[i:]                        # doubled letter


def edit_name(x, kind):
    t = str(x).split()
    if not t:
        return str(x)
    if kind == "legal":
        keep = [w for w in t if w.lower().strip(".,") not in LEGAL]
        t = keep if (keep and len(keep) < len(t)) else t + [pick(legal_t)]
    elif kind == "drop" and len(t) >= 2:
        del t[rnd.randrange(1, len(t))]
    elif kind == "add":
        t.insert(rnd.randrange(len(t) + 1), pick(gen_t))
    elif kind == "swap" and len(t) >= 2:
        i = rnd.randrange(len(t) - 1)
        t[i], t[i + 1] = t[i + 1], t[i]
    elif kind == "typo":
        i = rnd.randrange(len(t))
        t[i] = typo(t[i])
    elif kind == "other":
        for k in rnd.sample(["drop", "add", "swap", "typo"], 2):
            t = edit_name(" ".join(t), k).split()
    return " ".join(t)


def edit_addr(x, kind):
    x = str(x)
    if kind == "empty":
        return ""
    if kind == "cut" and "," in x:
        return x.rsplit(",", 1)[0]
    if kind == "nonum":
        t = x.split()
        i = next((j for j, w in enumerate(t) if any(ch.isdigit() for ch in w)), None)
        return " ".join(t[:i] + t[i + 1:]) if i is not None else x
    if kind == "other":
        t = x.split()
        if t:
            i = rnd.randrange(len(t))
            t[i] = typo(t[i])
        return " ".join(t)
    return x


nk = rnd.choices(list(NK.index), weights=list(NK.values), k=len(src_rows))
ak = rnd.choices(list(AK.index), weights=list(AK.values), k=len(src_rows))
raw = pd.DataFrame({"entity_id": [f"SYN-{i}" for i in range(len(src_rows))],
                    "business_name": [edit_name(x, k) for x, k in zip(s1.raw_name.values[src_rows], nk)],
                    "business_address": [edit_addr(x, k) for x, k in zip(s1.raw_addr.values[src_rows], ak)],
                    "country": TGT})
rs = er._normalize(raw)
gdf = Counter()
for col in (s1.name.values, r.name.values):
    for x in col:
        gdf.update(set(x.split()))
generic = {w for w, c in gdf.items() if c / (len(s1) + len(r)) > er.GENERIC_DF}   # same rule as load_split
rs["name_core"] = [core_name(x, generic) for x in rs.name.values]
rs["core_ns"] = rs.name_core.str.replace(" ", "", regex=False)
for i in range(3):
    log(f"example: {s1.raw_name.values[src_rows[i]]} | {s1.raw_addr.values[src_rows[i]]}  ->  "
        f"{raw.business_name[i]} | {raw.business_address[i]}  ({nk[i]}, {ak[i]})")
log(f"synthetic records {len(rs):,} from {len(pool):,} target S1")

# ---- same candidate search / prefilter as real records (reverse-S1 search off: it finds nothing on a full split)
_orig = er.sp_matmul_topn
# only the reverse search multiplies S1 rows by the QUERY matrix (output columns = queries); forward calls return S1 columns
er.sp_matmul_topn = lambda A, B, **kw: sps.coo_matrix((A.shape[0], B.shape[1])) if B.shape[1] == len(rs) else _orig(A, B, **kw)
ps = er.block(s1, rs, None)
er.sp_matmul_topn = _orig
ps = pd.concat([ps, er._competition_feats(ps)], axis=1)
pre = lgb.Booster(model_file=f"{W}/v4_prefilter.txt")
tau = json.load(open(f"{W}/v4_meta.json"))["prefilter_tau"]
ps = ps[pre.predict(ps[er.cheap_cols(ps)][pre.feature_name()], num_threads=0) >= tau].reset_index(drop=True)
ys = (ps.s.values == src_rows[ps.q.values]).astype(np.int8)
found = np.bincount(ps.q.values[ys == 1], minlength=len(rs)) > 0
log(f"synthetic pairs after prefilter {len(ps):,} | source S1 kept for {found.mean():.4f} of synthetic records")

# ---- synthetic rows: label tables re-learned with synthetic labels (cross-fitted by source-S1 halves)
srn, ssn = rs.name.values[ps.q.values], s1.name.values[ps.s.values]
srt, sst = rs["tail"].values[ps.q.values], s1["tail"].values[ps.s.values]
stset = cpdist(list(srn), list(ssn), scorer=fuzz.token_set_ratio, workers=-1)
shalf = sb_all[src_rows[ps.q.values]] % 2
rkey_all = pd.concat([r.country + "|" + r.name, rs.country + "|" + rs.name], ignore_index=True)
ctx_s = {"s1_name_freq": CTX["s1_name_freq"], "s1_addr_freq": CTX["s1_addr_freq"],
         "r_name_freq": rkey_all.map(rkey_all.value_counts()).values[len(r):].astype(np.float32)}


def tables_mix(m_real, m_syn):
    cat = lambda a, b: np.concatenate([a[m_real], b[m_syn]])
    return er._label_tables(cat(rn, srn), cat(sn, ssn), cat(rt, srt), cat(st, sst), cat(y, ys), cat(tset, stset))


parts = []
for h in (0, 1):
    lo_h, loc_h = tables_mix(fitm, shalf != h)
    rows_h = np.flatnonzero(shalf == h)
    parts.append(er.features(ps.iloc[rows_h].reset_index(drop=True), s1, rs, lo_h, IDF, loc_h, ctx_s).set_index(rows_h))
Xs = pd.concat(parts).sort_index()
Xt = feats(mt, *tables_mix(fitm, np.ones(len(ps), bool)))           # target pairs: tables from source + all synthetic
log(f"features synthetic {Xs.shape} | target {Xt.shape}")
X = pd.concat([Xf, Xs[cols]], ignore_index=True)
yy = np.r_[y[fitm], ys]
w = np.r_[np.ones(int(fitm.sum()), np.float32), np.full(len(Xs), args.w, np.float32)]
hold = np.r_[inner_f, np.zeros(len(Xs), bool)]                     # early stopping on SOURCE labels only
mdl = er.train_lgb(X, yy, hold, params=params, w=w)
new = prob.copy()
new[Xt.index.values] = mdl.predict(Xt[cols], num_threads=0)
new[np.flatnonzero(ms)] = mdl.predict(Xe.loc[np.flatnonzero(ms), cols], num_threads=0)
log(f"syn model {mdl.num_trees()} trees")
report("syn", new)

pd.set_option("display.width", 250)
print(f"\n=== E21 synthetic target examples, source {SRC} -> unlabelled {TGT}: pair-level F0.5 (95% CI), "
      f"threshold chosen on {SRC} ===")
print(pd.DataFrame(rows_out).round(4).to_string(index=False))
if args.base_draws:
    z = np.load(args.base_draws)
    for tgt in ("US", "India"):
        D[(tgt, "base")] = z[f"{tgt}|base"]
print("\n=== paired differences, 95% CI ===")
for tgt in ("US", "India"):
    for ref in (["base"] if args.base_draws else []) + ["reg"]:
        for n in ("reg", "syn"):
            if n == ref:
                continue
            dd = D[(tgt, n)] - D[(tgt, ref)]
            print(f"{SRC}->{tgt:5s} {n:4s} vs {ref:4s}: {dd.mean():+.4f}  CI [{np.quantile(dd, .025):+.4f}, "
                  f"{np.quantile(dd, .975):+.4f}]")
np.savez(f"{args.out}/draws21_{SRC}.npz", **{"|".join(k): v for k, v in D.items()})
log("E21 done")
