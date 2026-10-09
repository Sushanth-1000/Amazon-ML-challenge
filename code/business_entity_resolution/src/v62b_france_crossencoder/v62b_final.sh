# v6.2b: stage 2 re-fit with the France-adapted cross-encoder scores, apply on v6.2a test scores, France look-alike filter,
# validator, paired held-out check vs v6.1, France profile vs the teammates' France
until [ -f ~/v61_done ]; do sleep 60; done
until gcloud storage ls gs://mlc-gcp1-v6/v61/v62a_scored_done.txt >/dev/null 2>&1 && gcloud storage ls gs://mlc-gcp1-v6/v61/v62b_scored_done.txt >/dev/null 2>&1; do sleep 60; done
. ~/venv/bin/activate; cd ~; mkdir -p ~/w62b ~/out62b ~/sc62b
for f in train_s1 train_r test_s1 test_r train_scored train_edges test_edges; do ln -sfn ~/w61/$f.parquet ~/w62b/$f.parquet; done
cp ~/w61/meta.json ~/w62b/meta.json; ln -sfn ~/w61/matcher.txt ~/w62b/matcher.txt
gcloud storage cp gs://mlc-gcp1-v6/v61/v62a/test_scored_v62a.parquet ~/w62b/test_scored.parquet -q
gcloud storage cp "gs://mlc-gcp1-v6/v61/v62b/scored/*_ce.parquet" ~/sc62b/ -q
gcloud storage cp gs://mlc-gcp1-v6/code/v61/france_lookalike_filter.py gs://mlc-gcp1-v6/code/v61/cmp_heldout.py ~/src61/ -q
python - <<'PY'
import glob, pandas as pd
tr = [pd.read_parquet(f) for f in glob.glob("/home/Dell/sc62b/*_train_ce.parquet")]
te = [pd.read_parquet(f) for f in sorted(glob.glob("/home/Dell/sc62b/*_test_ce.parquet"))]
assert len(tr) == 1 and len(te) == 6, (len(tr), len(te))
pd.concat(tr, ignore_index=True).to_parquet("/home/Dell/w62b/train_ce.parquet")
t = pd.concat(te, ignore_index=True); assert not t.duplicated(["q", "s"]).any(); t.to_parquet("/home/Dell/w62b/test_ce.parquet")
print("train_ce", sum(len(x) for x in tr), "test_ce", len(t))
PY
python ~/src61/stage2.py --stage fit --data ~/dataset --work ~/w62b > ~/stage2_v62b.log 2>&1 || { echo STAGE2_FAILED >> ~/stage2_v62b.log; exit 1; }
python ~/src61/stage2.py --stage apply --data ~/dataset --work ~/w62b --out ~/out62b > ~/apply_v62b.log 2>&1 || { echo APPLY_FAILED >> ~/apply_v62b.log; exit 1; }
python ~/src61/france_lookalike_filter.py --matching ~/out62b/matching_results.tsv --out ~/out62b/matching_filtered.tsv --test-dir ~/dataset/test --country France >> ~/apply_v62b.log 2>&1
mv ~/out62b/matching_filtered.tsv ~/out62b/matching_results.tsv; cp ~/out61/candidate_pairs.tsv ~/out62b/candidate_pairs.tsv
python3 ~/dataset/../utils/validate_submission.py --matching ~/out62b/matching_results.tsv --candidate ~/out62b/candidate_pairs.tsv --test-dir ~/dataset/test --check-ids > ~/validate_v62b.log 2>&1
python ~/src61/cmp_heldout.py --data ~/dataset --work ~/w61 --a ~/w61 --b ~/w62b > ~/cmp_v62b.log 2>&1
python - <<'PY' >> ~/cmp_v62b.log 2>&1
import pandas as pd
rd = lambda p: pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=3)
s1 = rd("/home/Dell/dataset/test/test_source1.tsv").set_index("entity_id").country
for name, p in (("teammates France", "/home/Dell/w61/team.tsv"), ("v6.2b France", "/home/Dell/out62b/matching_results.tsv")):
    m = rd(p).set_index("source1_entity_id").matched_entity_ids; f = m[s1.reindex(m.index).values == "France"]
    n = f.str.split(",").map(lambda x: len([y for y in x if y]))
    print(f"{name}: matches per French S1 {n.mean():.3f} | empty rows {(n == 0).mean():.4f}")
PY
md5sum ~/out62b/*.tsv >> ~/cmp_v62b.log
gcloud storage cp ~/out62b/matching_results.tsv ~/out62b/candidate_pairs.tsv ~/stage2_v62b.log ~/apply_v62b.log ~/validate_v62b.log ~/cmp_v62b.log gs://mlc-gcp1-v6/final/v62b/ -q
echo V62B_DONE >> ~/cmp_v62b.log
