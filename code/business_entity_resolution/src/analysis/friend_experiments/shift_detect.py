"""Diagnosis 5: which stage-1 features look different on test than on train, per country (adversarial validation).
A LightGBM classifier tries to tell a random sample of test candidate pairs from train candidate pairs; if it can
(AUC well above 0.5), the features it uses most are the ones that shifted. France is compared with train US (the
closest seen country). Both sides use v4's own tables (label_tables.json) and their split's own IDF / counts, exactly as
the pipeline does. Test is only read. Run from student_resource/ after loco_generic.py (uses its cached train sample)."""
import argparse
import glob
import json
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="dataset")
ap.add_argument("--work", required=True)
ap.add_argument("--src", required=True)
ap.add_argument("--n", type=int, default=200_000)
args = ap.parse_args()
sys.path.insert(0, args.src)
import lightgbm as lgb  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import er  # noqa: E402

W = args.work
rng = np.random.default_rng(0)
lt = json.load(open(f"{W}/label_tables.json"))


def feats(split, p):
    s1, r = er.load_split(args.data, split, W)
    X = er.features(p.reset_index(drop=True), s1, r, lt["lo"], er.fit_idf(s1, r), lt["loc"], er.split_context(s1, r))
    return X, r.country.values[p.q.values]


# train sample: the cached LOCO pairs (all candidate pairs of the sampled train records)
tr = pd.read_parquet(sorted(glob.glob(f"{W}/loco_pairs_*.parquet"))[0])
tr = tr.iloc[rng.choice(len(tr), min(len(tr), 3 * args.n), replace=False)].sort_values(["q", "s"], ignore_index=True)
Xtr, ctr = feats("train", tr.drop(columns=[c for c in ("y",) if c in tr.columns]))

# test sample: random row groups of the test shortlist, kept to the pairs that survived the prefilter
ts = pd.read_parquet(f"{W}/test_scored.parquet", columns=["q", "s"])
pf = pq.ParquetFile(f"{W}/test_pairs.parquet")
parts = [pf.read_row_group(int(g)).to_pandas() for g in rng.choice(pf.num_row_groups, 60, replace=False)]
te = pd.concat(parts, ignore_index=True).merge(ts, on=["q", "s"]).sort_values(["q", "s"], ignore_index=True)
te = te.iloc[np.sort(rng.choice(len(te), min(len(te), 4 * args.n), replace=False))].reset_index(drop=True)
Xte, cte = feats("test", te)
print(f"train sample {len(Xtr):,} pairs {pd.Series(ctr).value_counts().to_dict()} | test sample {len(Xte):,} pairs "
      f"{pd.Series(cte).value_counts().to_dict()}", flush=True)

cols = [c for c in Xtr.columns if c in Xte.columns]
pd.set_option("display.width", 220)
for test_c, train_c in (("France", "US"), ("US", "US"), ("India", "India")):
    A, B = Xtr[ctr == train_c][cols], Xte[cte == test_c][cols]
    k = min(len(A), len(B), args.n)
    A, B = A.sample(k, random_state=0), B.sample(k, random_state=0)
    X = pd.concat([A, B], ignore_index=True)
    y = np.r_[np.zeros(k), np.ones(k)]
    hold = rng.random(len(y)) < 0.3
    m = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=63, verbose=-1, seed=0),
                  lgb.Dataset(X[~hold], y[~hold]), num_boost_round=300)
    auc = roc_auc_score(y[hold], m.predict(X[hold]))
    imp = pd.Series(m.feature_importance("gain"), index=cols).sort_values(ascending=False)
    top = imp.head(12).index
    desc = pd.DataFrame({"gain_share": (imp[top] / imp.sum()).round(3), f"train {train_c} mean": A[top].mean().round(3),
                         f"test {test_c} mean": B[top].mean().round(3)})
    print(f"\n=== test {test_c} vs train {train_c}: AUC {auc:.3f} (0.5 = no shift) ===\n{desc.to_string()}", flush=True)
