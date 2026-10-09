set -x
mkdir -p ~/v61 ~/v61emb ~/v61out
gcloud storage cp "gs://mlc-gcp1-v6/code/v61/*" ~/v61/ -q
ls ~/w/train_s1.parquet ~/w/train_r.parquet || gcloud storage cp gs://mlc-gcp1-v6/data/train_s1.parquet gs://mlc-gcp1-v6/data/train_r.parquet ~/w/ -q
gcloud storage cp gs://mlc-gcp1-v6/p1_archive/er4-cpu/w4/test_s1.parquet gs://mlc-gcp1-v6/p1_archive/er4-cpu/w4/test_r.parquet ~/w/ -q
df -h ~ | tail -1
python3 ~/v61/v61_gpu_embed.py --work ~/w --emb ~/v61emb --out ~/v61out > ~/v61_g1.log 2>&1
gcloud storage cp ~/v61out/train_dense.parquet ~/v61out/test_dense.parquet ~/v61_g1.log gs://mlc-gcp1-v6/v61/ -q
echo G1_DONE > ~/g1_done
