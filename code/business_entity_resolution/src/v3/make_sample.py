"""Build a small dev copy of the dataset (same file layout) for fast local iteration.

Train: N random S1 entities + all their matched S2/S3 records + unmatched S2/S3 decoys at the
real ~26% decoy rate. Test: N random test S1 + a proportional random slice of test S2/S3 (smoke test only).
"""
import os, sys
import pandas as pd

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
SRC, DST = "dataset", "data_sample"
rd = lambda p: pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=3)
wr = lambda df, p: df.to_csv(p, sep="\t", index=False, quoting=3)

os.makedirs(f"{DST}/train", exist_ok=True); os.makedirs(f"{DST}/test", exist_ok=True)
gt = rd(f"{SRC}/train/train_ground_truth.tsv").sample(N, random_state=0)
keep = {x for l in gt.matched_entity_ids for x in l.split(",") if x}
all_matched = set()
with open(f"{SRC}/train/train_ground_truth.tsv", encoding="utf-8") as f:
    next(f)
    for line in f:
        all_matched.update(x for x in line.rstrip("\n").split("\t")[1].split(",") if x)
wr(gt, f"{DST}/train/train_ground_truth.tsv")
wr(rd(f"{SRC}/train/train_source1.tsv").pipe(lambda d: d[d.entity_id.isin(set(gt.source1_entity_id))]),
   f"{DST}/train/train_source1.tsv")
for i in (2, 3):
    d = rd(f"{SRC}/train/train_source{i}.tsv")
    m = d[d.entity_id.isin(keep)]
    decoys = d[~d.entity_id.isin(all_matched)].sample(int(len(m) * 0.35), random_state=i)  # ~26% of kept rows
    wr(pd.concat([m, decoys]).sample(frac=1, random_state=i), f"{DST}/train/train_source{i}.tsv")

frac = N / 1732544
wr(rd(f"{SRC}/test/test_source1.tsv").sample(N, random_state=0), f"{DST}/test/test_source1.tsv")
for i in (2, 3):
    wr(rd(f"{SRC}/test/test_source{i}.tsv").sample(frac=frac, random_state=i), f"{DST}/test/test_source{i}.tsv")
print("sample written to", DST)
