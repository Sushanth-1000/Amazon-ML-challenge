"""Stage 2: re-score candidate pairs using the other records that claim the same S1 entity (siblings).

Stage 1 judges each (record, S1) pair alone. But an entity's true records agree with each other and
with its house number, while decoy clones (house number shifted by a few units) agree with each other
and not with the entity. Stage 2 adds that context:
  * support    - how many other records confidently claim this S1 entity, how strongly
  * agreement  - address / name / house-number similarity of this record to those confident siblings
  * cluster    - do the confident siblings share the entity's house number (true cluster) or not
plus the stage-1 probability and its standing among the record's own candidates.

Trained on the validation pairs (stage-1 probabilities there are out-of-fold) with 2-fold cross-fitting
grouped by entity (a record's fold = hash of its true S1, or of itself if unmatched, so siblings of one
entity never sit on both sides), optionally with the cross-encoder score (src/ce.py), then the decision rule is re-tuned on the stage-2 probabilities (tune on entity half A,
report on half B). If it beats stage 1 on half B, work/meta.json switches to stage 2.

Usage (from student_resource/, after decode.py):  python src/stage2.py --work work
"""
import argparse
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from er import _first_int, decode, id_bucket, load_truth, log, macro_f05, to_sets

MIN_P1 = 0.005   # pairs below this never win an assignment; stage 2 only scores the rest
ANCHOR_P = 0.5   # a sibling counts as a confident claim of its best S1 above this stage-1 probability
PARAMS2 = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=100,
               feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=0)
ROUNDS2 = 400


def load_ce(work, split, p):
    """Cross-encoder logits aligned with pairs p (NaN where the pair was not scored); None if absent."""
    path = f"{work}/{split}_ce.parquet"
    if not os.path.exists(path):
        return None
    ce = pd.read_parquet(path)[["q", "s", "ce"]]
    return p[["q", "s"]].merge(ce, on=["q", "s"], how="left").ce.values.astype(np.float32)


def stage2_features(q, s, p1, s1, r, ce=None):
    """q, s: aligned pair arrays (rows of r / s1); p1: stage-1 probability; ce: optional cross-encoder
    logit (NaN = not scored). Returns a float32 frame."""
    d = pd.DataFrame({"q": q, "s": s, "p1": p1})
    if ce is not None:
        d["ce"] = ce
        d["ce_rank"] = d.groupby("q").ce.rank(ascending=False, method="first")
        d["ce_gap"] = d.ce - d.groupby("q").ce.transform("max")
    g = d.groupby("q").p1
    d["q_rank"] = g.rank(ascending=False, method="first")
    d["q_max"] = g.transform("max")
    d["q_n"] = g.transform("size")
    second = d.sort_values(["q", "p1"], ascending=[True, False]).groupby("q").p1.nth(1)
    second = pd.Series(second.values, index=d.q.loc[second.index].values)
    d["q_gap"] = np.where(d.q_rank == 1, d.p1 - d.q.map(second).fillna(0), d.p1 - d.q_max)

    # confident claims: each record's best S1 if its stage-1 probability clears ANCHOR_P
    best = d[(d.q_rank == 1) & (d.p1 >= ANCHOR_P)][["q", "s", "p1"]]
    claims = best.groupby("s").p1.agg(["size", "sum", "max"])
    own = np.where((d.q_rank == 1) & (d.p1 >= ANCHOR_P), d.p1, 0.0)  # exclude the pair's own claim
    own_n = (own > 0).astype(np.float32)
    d["s_claims"] = d.s.map(claims["size"]).fillna(0).values - own_n
    d["s_claim_sum"] = d.s.map(claims["sum"]).fillna(0).values - own
    d["s_all_sum"] = d.groupby("s").p1.transform("sum") - d.p1

    # sibling agreement: compare the record with every other confident claimant of the same S1
    a = d[["q", "s"]].reset_index().merge(best.rename(columns={"q": "qa", "p1": "pa"}), on="s")
    a = a[a.qa.values != a.q.values]
    rq, ra = a.q.values, a.qa.values
    a["sib_addr"] = cpdist(list(r.addr.values[rq]), list(r.addr.values[ra]), scorer=fuzz.token_set_ratio,
                           workers=-1, dtype=np.float32)
    a["sib_name"] = cpdist(list(r.name.values[rq]), list(r.name.values[ra]), scorer=fuzz.token_set_ratio,
                           workers=-1, dtype=np.float32)
    a["sib_sk"] = cpdist(list(r.name_sk.values[rq]), list(r.name_sk.values[ra]), scorer=fuzz.ratio,
                         workers=-1, dtype=np.float32)
    rnum = _first_int(r.nums.values)
    snum = _first_int(s1.nums.values)
    fq, fa, fs = rnum[rq], rnum[ra], snum[a.s.values]
    a["sib_num_eq"] = ((fq >= 0) & (fq == fa)).astype(np.float32)
    a["sib_num_is_s1"] = ((fa >= 0) & (fa == fs)).astype(np.float32)  # sibling's number = entity's number
    agg = a.groupby("index").agg(sib_addr_max=("sib_addr", "max"), sib_name_max=("sib_name", "max"),
                                 sib_sk_max=("sib_sk", "max"), sib_num_eq_n=("sib_num_eq", "sum"),
                                 sib_num_is_s1_frac=("sib_num_is_s1", "mean"))
    d = d.join(agg)
    d[agg.columns] = d[agg.columns].fillna(-1)

    fr, fs1 = rnum[q], snum[s]
    d["num_eq_s1"] = np.where((fr < 0) | (fs1 < 0), -1, fr == fs1)
    d["num_delta_s1"] = np.where((fr < 0) | (fs1 < 0), np.nan, np.abs(fr - fs1))
    d["noaddr"] = r.addr.values[q] == ""
    d["nonlatin"] = r.nonlatin.values[q]
    return d.drop(columns=["q", "s"]).astype(np.float32)


def main():
    from decode import val_predictions  # reuses the exact validation pairs + stage-1 probabilities
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--work", default="work")
    ap.add_argument("--val-pct", type=int, default=1)
    ap.add_argument("--no-ce", action="store_true", help="ignore cross-encoder scores even if present")
    args = ap.parse_args()

    s1, r, pva = val_predictions(args)
    pp = pva[pva.prob.values >= MIN_P1].reset_index(drop=True)
    truth, rid2s = load_truth(args.data)
    s_val = id_bucket(s1.entity_id.values) < args.val_pct
    val_ids = s1.entity_id.values[s_val]
    truth = {i: truth[i] for i in val_ids}
    y = pp.y.values.astype(np.int8)
    ce = load_ce(args.work, "val", pp) if not args.no_ce else None
    log(f"cross-encoder scores: {'none' if ce is None else f'{np.isfinite(ce).sum():,} pairs'}")

    X = stage2_features(pp.q.values, pp.s.values, pp.prob.values, s1, r, ce)
    log(f"stage-2 pairs {len(pp):,}, positives {y.sum():,}, features {X.shape[1]}")
    key = r.entity_id.map(rid2s).fillna(r.entity_id).values  # entity of the record (or itself)
    fold = (id_bucket(key) % 2)[pp.q.values]  # cross-fit grouped by entity
    p2 = np.zeros(len(pp))
    for k in (0, 1):
        m = lgb.train(PARAMS2, lgb.Dataset(X[fold != k], y[fold != k]), num_boost_round=ROUNDS2)
        p2[fold == k] = m.predict(X[fold == k], num_threads=0)

    noaddr = r.addr.values[pp.q.values] == ""
    half = id_bucket(np.array([i[::-1] for i in val_ids])) < 50
    ids_a, ids_b = val_ids[half], val_ids[~half]
    f = lambda prob, d, ids: macro_f05(to_sets(decode(pp, prob, d, noaddr), s1, r), truth, ids)
    meta = json.load(open(f"{args.work}/meta.json"))
    base_b = f(pp.prob.values, meta["decoder"], ids_b)
    grid = [{"kind": "threshold", "t": float(t)} for t in np.arange(0.2, 0.96, 0.025)] + \
           [{"kind": "expf", "floor": float(fl)} for fl in (0.02, 0.05, 0.1, 0.2, 0.3)]
    best = max(grid, key=lambda d: f(p2, d, ids_a))
    key = "floor" if best["kind"] == "expf" else "t"
    best = max([best] + [{**best, "t_na": float(x)} for x in np.arange(best[key], 1.0, 0.025)],
               key=lambda d: f(p2, d, ids_a))
    s2_b, s2_all = f(p2, best, ids_b), f(p2, best, val_ids)
    log(f"stage 1 (current decoder) F0.5 on B {base_b:.4f} | stage 2 {best} -> B {s2_b:.4f}, all val {s2_all:.4f}")

    final = lgb.train(PARAMS2, lgb.Dataset(X, y), num_boost_round=ROUNDS2)
    imp = pd.Series(final.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
    log("stage-2 feature importance:\n" + imp.round(0).to_string())
    if s2_b > base_b:
        final.save_model(f"{args.work}/model2.txt")
        meta.update(stage2=True, decoder=best, stage2_val_f05_B=s2_b, stage2_features=list(X.columns),
                    use_ce=ce is not None)
        log("stage 2 adopted -> work/model2.txt, meta.json updated")
    else:
        meta["stage2"] = False
        log("stage 2 NOT adopted (no gain on half B)")
    json.dump(meta, open(f"{args.work}/meta.json", "w"))


if __name__ == "__main__":
    main()
