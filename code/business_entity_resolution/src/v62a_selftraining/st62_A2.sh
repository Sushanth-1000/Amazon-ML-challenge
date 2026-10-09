# after v6.2a merge on p2-v61a: export v6.2a test top-3 pairs for cross-encoder scoring (v4 ce.py, as v6.1), shard for AWS
until grep -q ST62A_DONE ~/st_merge.log 2>/dev/null; do sleep 30; done
. ~/venv/bin/activate; cd ~
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
echo A2_DONE >> ~/ce_export_v62a.log
