"""Quick EDA over train/test: sizes, countries, match-count distribution, record reuse, noise samples."""
import re, collections
import pandas as pd

D = "dataset"
rd = lambda p: pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=3)

gt = rd(f"{D}/train/train_ground_truth.tsv")
s1, s2, s3 = (rd(f"{D}/train/train_source{i}.tsv") for i in (1, 2, 3))
t1, t2, t3 = (rd(f"{D}/test/test_source{i}.tsv") for i in (1, 2, 3))

print("rows train s1/s2/s3/gt:", len(s1), len(s2), len(s3), len(gt))
print("rows test  s1/s2/s3:", len(t1), len(t2), len(t3))
for n, df in [("train s1", s1), ("train s2", s2), ("train s3", s3), ("test s1", t1), ("test s2", t2), ("test s3", t3)]:
    print(n, df.country.value_counts().to_dict(), "| empty addr:", (df.business_address == "").mean().round(3),
          "| empty name:", (df.business_name == "").mean().round(3))

m = gt.matched_entity_ids.str.split(",").apply(lambda l: [x for x in l if x])
n2 = m.apply(lambda l: sum(x.startswith("S2-") for x in l)); n3 = m.apply(lambda l: sum(x.startswith("S3-") for x in l))
tot = n2 + n3
print("\nsingleton rate:", (tot == 0).mean().round(4))
print("matches per S1:", tot.value_counts().sort_index().head(15).to_dict())
print("S2 per S1:", n2.value_counts().sort_index().to_dict())
print("S3 per S1:", n3.value_counts().sort_index().to_dict())
c = collections.Counter(x for l in m for x in l)
print("S2/S3 ids referenced by >1 S1:", sum(v > 1 for v in c.values()))
used = set(c)
print("S2 records that match something:", round(s2.entity_id.isin(used).mean(), 4),
      "| S3:", round(s3.entity_id.isin(used).mean(), 4))
print("GT covers all S1:", set(gt.source1_entity_id) == set(s1.entity_id))

# singleton rate by country
cty = s1.set_index("entity_id").country
g = pd.DataFrame({"c": gt.source1_entity_id.map(cty), "single": tot == 0, "tot": tot})
print("\nby country:", g.groupby("c").agg(single=("single", "mean"), avg=("tot", "mean")).round(3).to_dict())

# cross-country matches?
cmap = pd.concat([s2, s3]).set_index("entity_id").country
ex = gt.assign(l=m).explode("l").dropna(subset=["l"])
ex["c1"] = ex.source1_entity_id.map(cty); ex["c2"] = ex.l.map(cmap)
print("match pairs:", len(ex), "| cross-country pairs:", (ex.c1 != ex.c2).sum())

# script mix
nonlatin = lambda s: bool(re.search(r"[^\x00-ɏ]", s))
for n, df in [("s1", s1), ("s2", s2), ("s3", s3), ("t1", t1), ("t2", t2), ("t3", t3)]:
    smp = df.sample(200000, random_state=0) if len(df) > 200000 else df
    print(n, "non-latin name:", smp.business_name.map(nonlatin).mean().round(3),
          "non-latin addr:", smp.business_address.map(nonlatin).mean().round(3))

# show examples of matched groups
nm = pd.concat([s1, s2, s3]).set_index("entity_id")
print("\n--- sample match groups ---")
for _, r in gt[tot > 0].sample(8, random_state=1).iterrows():
    print(">>", r.source1_entity_id, "|", " | ".join(nm.loc[r.source1_entity_id, ["business_name", "business_address", "country"]]))
    for x in r.matched_entity_ids.split(","):
        print("   ", x, "|", " | ".join(nm.loc[x, ["business_name", "business_address"]]))
print("\n--- sample singletons ---")
for sid in gt[tot == 0].source1_entity_id.sample(5, random_state=2):
    print(">>", sid, "|", " | ".join(nm.loc[sid, ["business_name", "business_address", "country"]]))
print("\n--- France test samples ---")
print(t1[t1.country == "France"].sample(8, random_state=0).to_string(index=False))
print(t2[t2.country == "France"].sample(5, random_state=0).to_string(index=False))
