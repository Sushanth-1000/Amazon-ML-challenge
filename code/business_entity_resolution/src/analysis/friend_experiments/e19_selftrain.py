"""E19: self-training on an unlabelled country (full-scale leave-one-country-out, same data/splits/bootstrap as E1).
The SOURCE country is labelled (split A, buckets [2,51)); the TARGET country plays France: its labels are never used
for training, pseudo-labelling, table fitting, threshold choice or early stopping, only for the final score.
  reg   : v4 features + REG parameters, trained on the source only                     (= E1 'reg', reproduced)
  st<k> : round k of self-training on the target's EVAL pairs (transductive, as it would run on the French test):
          pseudo-labels from the previous round's probabilities (fixed a priori, never tuned):
            match     = record's best candidate, p >= HI and p - second best >= MARGIN
            non-match = p <= LO, or any other candidate of a record with a pseudo-match (<= 1 S1 per record)
          cross-fitted: pseudo-labels of target half 1-h (records split by entity hash) re-learn the decoy/locality
          label tables and are added (weight W) to the source training rows; that model scores target half h.
          Pseudo rows used for training get tables from the other quarter only (same cross-fitting as E1/v4).
Threshold = best on the SOURCE (as E1). Paired bootstrap vs reg (same resamples) and vs E1 base (--base-draws).
Usage: python e19_selftrain.py --data dataset --work ~/w --src ~/src --source US --out ~/e19 --base-draws draws_US.npz"""
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
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--hi", type=float, default=0.9)
ap.add_argument("--lo", type=float, default=0.05)
ap.add_argument("--margin", type=float, default=0.3)
ap.add_argument("--w", type=float, default=0.5)
ap.add_argument("--smoke", action="store_true", help="5%% of entities, for a quick correctness run")
ap.add_argument("--target-only", action="store_true", help="E25a: model + label tables from the target pseudo-labels only")
ap.add_argument("--tag", default="")
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
    pd.DataFrame(rows_out).to_csv(f"{args.out}/e19{args.tag}_{SRC}.csv", index=False)
    np.save(f"{args.out}/prob{args.tag}_{SRC}_{name}.npy", prob[evm])


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


def pseudo(pr):
    """Pseudo-labels on the target pairs from probabilities only (no target labels)."""
    rows = np.flatnonzero(mt)
    d = pd.DataFrame({"q": p.q.values[rows], "pr": pr[rows], "i": rows}).sort_values(["q", "pr"], ascending=[True, False])
    qv, pv = d.q.values, d.pr.values
    first = np.r_[True, qv[1:] != qv[:-1]]
    nxt_same = np.r_[qv[1:] == qv[:-1], False]
    sec = np.where(first & nxt_same, np.r_[pv[1:], 0.0], 0.0)
    pos = first & (pv >= args.hi) & (pv - sec >= args.margin)
    posq = np.zeros(len(r), bool); posq[qv[pos]] = True
    neg = ((~first) & posq[qv]) | (pv <= args.lo)
    lab = np.full(len(p), -1, np.int8)
    lab[d.i.values[neg & ~pos]] = 0
    lab[d.i.values[pos]] = 1
    return lab


for k in range(1, args.rounds + 1):
    lab = pseudo(prob)
    L1, L0 = mt & (lab == 1), mt & (lab == 0)
    # diagnostics only (target labels used for reporting, never for fitting)
    log(f"round {k}: pseudo matches {int(L1.sum()):,} (precision {y[L1].mean():.4f}, cover {y[L1].sum() / max(y[mt].sum(), 1):.4f} "
        f"of true target pairs) | pseudo non-matches {int(L0.sum()):,} (share truly non-matching {1 - y[L0].mean():.5f}) | "
        f"unlabelled {int((mt & (lab < 0)).sum()):,}")
    ylab = y.copy(); ylab[mt] = lab[mt]                      # target rows now carry pseudo-labels only
    new = prob.copy()
    src_pred = []
    for h in (0, 1):
        P = mt & (half == 1 - h) & (lab >= 0)
        base = np.zeros(len(p), bool) if args.target_only else fitm             # E25a: no source rows at all
        Xs = feats(mt & (half == h), *tables(base | P, ylab))                     # rows scored
        Xp = pd.concat([feats(P & (quarter == a), *tables(base | (P & (quarter != a)), ylab)) for a in (0, 1)])
        if args.target_only:                                                      # pseudo rows only; hold = 10% of them
            X = Xp[cols].reset_index(drop=True)
            yy = lab[Xp.index.values]
            w = np.ones(len(Xp), np.float32)
            hold = er.id_bucket(r.entity_id.values[p.q.values[Xp.index.values]]) % 10 == 0
        else:
            X = pd.concat([Xf, Xp[cols]], ignore_index=True)
            yy = np.r_[y[fitm], lab[Xp.index.values]]
            w = np.r_[np.ones(int(fitm.sum()), np.float32), np.full(len(Xp), args.w, np.float32)]
            hold = np.r_[inner_f, np.zeros(len(Xp), bool)]
        mdl = er.train_lgb(X, yy, hold, params=params, w=w)
        new[Xs.index.values] = mdl.predict(Xs[cols], num_threads=0)
        src_pred.append(mdl.predict(Xe.loc[np.flatnonzero(ms), cols], num_threads=0))
        log(f"round {k} half {h}: {len(Xp):,} pseudo rows, {mdl.num_trees()} trees")
        del Xs, Xp, X
    new[np.flatnonzero(ms)] = np.mean(src_pred, axis=0)     # source rows: average of the two models (threshold choice)
    prob = new
    report(f"st{k}", prob)

pd.set_option("display.width", 250)
print(f"\n=== E19 self-training, source {SRC} -> unlabelled {TGT}: pair-level F0.5 (95% CI), threshold chosen on {SRC} ===")
print(pd.DataFrame(rows_out).round(4).to_string(index=False))
names = ["reg"] + [f"st{k}" for k in range(1, args.rounds + 1)]
if args.base_draws:
    z = np.load(args.base_draws)
    for tgt in ("US", "India"):
        D[(tgt, "base")] = z[f"{tgt}|base"]
print("\n=== paired differences, 95% CI ===")
for tgt in ("US", "India"):
    for ref in (["base"] if args.base_draws else []) + ["reg"]:
        for n in names:
            if n == ref:
                continue
            dd = D[(tgt, n)] - D[(tgt, ref)]
            print(f"{SRC}->{tgt:5s} {n:4s} vs {ref:4s}: {dd.mean():+.4f}  CI [{np.quantile(dd, .025):+.4f}, {np.quantile(dd, .975):+.4f}]")
np.savez(f"{args.out}/draws19{args.tag}_{SRC}.npz", **{"|".join(k): v for k, v in D.items()})
log("E19 done")
