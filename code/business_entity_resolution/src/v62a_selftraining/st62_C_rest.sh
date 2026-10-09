# moved from p2-v61a (stopped to save credits): merge halves -> v6.2b CE groups -> export + shard v6.2a test pairs
until grep -q C_FULL_DONE ~/st_half1.log 2>/dev/null; do sleep 30; done
. ~/venv/bin/activate; cd ~
gcloud storage cp gs://mlc-gcp1-v6/v61/v62a/st_fr_half0.npz ~/w/ -q
python ~/src/st_france.py --work ~/w --merge > ~/st_merge.log 2>&1 || { echo MERGE_FAILED >> ~/st_merge.log; exit 1; }
gcloud storage cp ~/w/test_scored_v62a.parquet ~/w/fr_pseudo_labels.npy ~/st_merge.log gs://mlc-gcp1-v6/v61/v62a/ -q && echo done > ~/m.txt && gcloud storage cp ~/m.txt gs://mlc-gcp1-v6/v61/v62a_scored_done.txt -q
gcloud storage cp gs://mlc-gcp1-v6/v61/v62in/ce_groups.parquet ~/w/ -q
python ~/src/fr_ce_groups.py --work ~/w --data ~/dataset --out ~/v62b > ~/fr_ce_groups.log 2>&1 || { echo GROUPS_FAILED >> ~/fr_ce_groups.log; exit 1; }
gcloud storage cp ~/v62b/ce_groups.parquet ~/fr_ce_groups.log gs://mlc-gcp1-v6/v61/v62b/ -q && echo done > ~/g.txt && gcloud storage cp ~/g.txt gs://mlc-gcp1-v6/v61/v62b_groups_done.txt -q
mkdir -p ~/w62a; for f in test_s1.parquet test_r.parquet meta.json; do ln -sfn ~/w/$f ~/w62a/$f; done
ln -sfn ~/w/test_scored_v62a.parquet ~/w62a/test_scored.parquet
python ~/src/ce.py export_test --data ~/dataset --work ~/w62a > ~/ce_export_v62a.log 2>&1 || { echo EXPORT_FAILED >> ~/ce_export_v62a.log; exit 1; }
python - <<'PY'
import numpy as np, pandas as pd
t = pd.read_parquet("/home/Dell/w62a/ce_score_test.parquet"); n = 6
qs = np.sort(t.q.unique()); cut = qs[[len(qs) * k // n for k in range(1, n)]]
b = np.searchsorted(cut, t.q.values, side="right")
for k in range(n): t[b == k].to_parquet(f"/home/Dell/w62a/shard_test_{k}.parquet")
print("test pairs", len(t), "shards", [int((b == k).sum()) for k in range(n)])
PY
gcloud storage cp ~/w62a/ce_score_test.parquet ~/w62a/shard_test_*.parquet gs://mlc-gcp1-v6/v61/v62b/ -q
gcloud storage cp gs://mlc-gcp1-v6/v61/ce_score_train.parquet gs://mlc-gcp1-v6/v61/v62b/ce_score_train.parquet -q
echo done > ~/p.txt && gcloud storage cp ~/p.txt gs://mlc-gcp1-v6/v61/v62b_pairs_done.txt -q
echo C_REST_DONE >> ~/ce_export_v62a.log
