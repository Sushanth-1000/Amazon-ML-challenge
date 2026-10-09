"""v6.2a: one round of self-training on the FRENCH test pairs (production version of E19's tested recipe).
Inputs = v6.1 outputs in --work: train_X.parquet (labelled train rows, v6.1 features), test_pairs61.parquet,
test_X.parquet, test_scored.parquet (v6.1 stage-1 p1), meta.json, train/test s1 + r tables.
  pseudo-labels (E19 rule, fixed a priori): match = record's best candidate, p1 >= 0.9 and p1 - second >= 0.3;
  non-match = p1 <= 0.05, or any other candidate of a pseudo-matched record
  cross-fitted by French record halves (record id hash): pseudo-labels of half 1-h re-learn the decoy/locality label
  tables (together with the labelled train rows of splits A+B) and are added (weight 0.5) to those train rows; that
  model (reg, seed 0) re-scores French half h. Pseudo rows used for training get tables from the other quarter only.
Only French rows change; US/India rows keep v6.1's p1.  Run once per half (two machines), then --merge.
Usage: python st_france.py --work ~/w61 --data ~/dataset --half 0|1     then     python st_france.py --work ~/w61 --merge"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import er  # noqa: E402
from er import log  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True)
ap.add_argument("--data", default="dataset")
ap.add_argument("--half", type=int, choices=[0, 1])
ap.add_argument("--merge", action="store_true")
ap.add_argument("--smoke", action="store_true", help="1 in 50 French records and 1 in 20 train rows")
args = ap.parse_args()
W = args.work
REG = dict(num_leaves=31, min_data_in_leaf=1000, feature_fraction=0.5, lambda_l2=10.0)
HI, LO, MARGIN, WPS = 0.9, 0.05, 0.3, 0.5

if args.merge:
    sc = pd.read_parquet(f"{W}/test_scored.parquet")
    new = sc.p1.values.copy()
    for h in (0, 1):
        z = np.load(f"{W}/st_fr_half{h}.npz")
        assert np.isfinite(z["p1"]).all()
        new[z["rows"]] = z["p1"]
    changed = int((new != sc.p1.values).sum())
    sc.assign(p1=new.astype(np.float32)).to_parquet(f"{W}/test_scored_v62a.parquet")
    log(f"test_scored_v62a.parquet written: {changed:,} French pairs re-scored")
    sys.exit(0)

meta = json.load(open(f"{W}/meta.json"))
fcols = meta["features"]
s1t, rt_ = er.load_split(args.data, "test", W)
pt = pd.read_parquet(f"{W}/test_pairs61.parquet")
p1 = pd.read_parquet(f"{W}/test_scored.parquet", columns=["q", "s", "p1"])
assert (p1.q.values == pt.q.values).all() and (p1.s.values == pt.s.values).all(), "test_scored must follow test_pairs61"
p1 = p1.p1.values
fr = rt_.country.values[pt.q.values] == "France"
rbk = er.id_bucket(rt_.entity_id.values)[pt.q.values]
half, quarter = rbk % 2, (rbk // 2) % 2
if args.smoke:
    fr &= (rbk % 50 == 0)
log(f"test pairs {len(pt):,} | French {int(fr.sum()):,}")

# ---- pseudo-labels on French pairs (no labels exist; identical rule to E19)
rows = np.flatnonzero(fr)
d = pd.DataFrame({"q": pt.q.values[rows], "pr": p1[rows], "i": rows}).sort_values(["q", "pr"], ascending=[True, False])
qv, pv = d.q.values, d.pr.values
first = np.r_[True, qv[1:] != qv[:-1]]
nxt = np.r_[qv[1:] == qv[:-1], False]
sec = np.where(first & nxt, np.r_[pv[1:], 0.0], 0.0)
pos = first & (pv >= HI) & (pv - sec >= MARGIN)
posq = np.zeros(len(rt_), bool); posq[qv[pos]] = True
neg = ((~first) & posq[qv]) | (pv <= LO)
lab = np.full(len(pt), -1, np.int8)
lab[d.i.values[neg & ~pos]] = 0
lab[d.i.values[pos]] = 1
# v3 agreement filter: the teammates' France rows (v3 + look-alike filter, LB 0.983297) are leaderboard-measured better
# than v4's France (+0.0011 for v3 rows, +0.00055 for the filter). Agreement is checked against that file.
# Keep a pseudo-label only where v3 agrees: match = v6.1 confident AND v3 matched the pair; non-match = v6.1 rejects AND
# v3 did not match it. Disagreements -> unlabelled. (v3 = train-label-only pipeline: ordinary self-training.)
v3p = f"{W}/v3_matching_results.tsv"
if not os.path.exists(v3p):
    import subprocess
    subprocess.run(["gcloud", "storage", "cp", "gs://mlc-gcp1-v6/v62/team_0983297_matching_results.tsv", v3p, "-q"], check=True)
v3 = pd.read_csv(v3p, sep="	", dtype=str, keep_default_na=False, quoting=3)
s_pos = pd.Series(np.arange(len(s1t)), index=s1t.entity_id.values)
r_pos = pd.Series(np.arange(len(rt_)), index=rt_.entity_id.values)
v3k = set()
for sid, ms in zip(v3.source1_entity_id.values, v3.matched_entity_ids.values):
    if ms and sid in s_pos.index:
        si = int(s_pos[sid])
        v3k.update((int(r_pos[x]) << 22) | si for x in ms.split(",") if x in r_pos.index)
kk = (pt.q.values.astype(np.int64) << 22) | pt.s.values.astype(np.int64)
in_v3 = np.fromiter((k in v3k for k in kk), bool, len(kk))
before = (int((lab == 1).sum()), int((lab == 0).sum()))
lab[(lab == 1) & ~in_v3] = -1
lab[(lab == 0) & in_v3] = -1
log(f"v3 agreement filter: pseudo matches {before[0]:,} -> {int((lab == 1).sum()):,} | non-matches {before[1]:,} -> "
    f"{int((lab == 0).sum()):,} | v3 French matched pairs among candidates {int((in_v3 & fr).sum()):,}")
np.save(f"{W}/fr_pseudo_labels.npy", lab)
log(f"French pseudo matches {int((lab == 1).sum()):,} | non-matches {int((lab == 0).sum()):,} | "
    f"unlabelled {int((fr & (lab < 0)).sum()):,}")

# ---- labelled train rows (splits A+B) with their v6.1 features
X = pd.read_parquet(f"{W}/train_X.parquet")
X = X[X.split.values != "H"].reset_index(drop=True)
if args.smoke:
    X = X[X.q.values % 20 == 0].reset_index(drop=True)
s1r, rr = er.load_split(args.data, "train", W)
from rapidfuzz import fuzz  # noqa: E402
from rapidfuzz.process import cpdist  # noqa: E402
tr_rn, tr_sn = rr.name.values[X.q.values], s1r.name.values[X.s.values]
tr_rt, tr_st = rr["tail"].values[X.q.values], s1r["tail"].values[X.s.values]
tr_ts = cpdist(list(tr_rn), list(tr_sn), scorer=fuzz.token_set_ratio, workers=-1)
te_rn, te_sn = rt_.name.values[pt.q.values], s1t.name.values[pt.s.values]
te_rt, te_st = rt_["tail"].values[pt.q.values], s1t["tail"].values[pt.s.values]
log("train arrays ready")


def tables(test_mask):
    """Label tables from all labelled train rows + the pseudo-labelled French rows in test_mask."""
    m = np.flatnonzero(test_mask & (lab >= 0))
    ts = cpdist(list(te_rn[m]), list(te_sn[m]), scorer=fuzz.token_set_ratio, workers=-1)
    return er._label_tables(np.r_[tr_rn, te_rn[m]], np.r_[tr_sn, te_sn[m]], np.r_[tr_rt, te_rt[m]],
                            np.r_[tr_st, te_st[m]], np.r_[X.y.values, lab[m]], np.r_[tr_ts, ts])


idf, ctx = er.fit_idf(s1t, rt_), er.split_context(s1t, rt_)
Xt_cos = pd.read_parquet(f"{W}/test_X.parquet", columns=["mm_cos", "mm_cos_gap", "mm_cos_rank"])


def feats(m, lo, loc):
    rr_ = np.flatnonzero(m)
    F = er.features(pt.iloc[rr_].reset_index(drop=True), s1t, rt_, lo, idf, loc, ctx)
    return pd.concat([F, Xt_cos.iloc[rr_].reset_index(drop=True)], axis=1).set_index(rr_)


h = args.half
P = fr & (half == 1 - h) & (lab >= 0)                                    # pseudo rows used for training
Xs = feats(fr & (half == h), *tables(P))                                 # French rows re-scored in this half
Xp = pd.concat([feats(P & (quarter == a), *tables(P & (quarter != a))) for a in (0, 1)])
log(f"half {h}: score {len(Xs):,} French pairs | {len(Xp):,} pseudo rows")
Xtr = pd.concat([X[fcols], Xp[fcols]], ignore_index=True)
ytr = np.r_[X.y.values, lab[Xp.index.values]]
w = np.r_[np.ones(len(X), np.float32), np.full(len(Xp), WPS, np.float32)]
hold = np.r_[er.id_bucket(rr.entity_id.values[X.q.values]) % 10 == 0, np.zeros(len(Xp), bool)]   # train labels only
m = er.train_lgb(Xtr, ytr, hold, params=dict(er.LGB_PARAMS, **REG, seed=0, bagging_seed=0, feature_fraction_seed=0), w=w)
np.savez(f"{W}/st_fr_half{h}.npz", rows=Xs.index.values, p1=m.predict(Xs[fcols], num_threads=0).astype(np.float32))
log(f"half {h} done: {m.num_trees()} trees")
