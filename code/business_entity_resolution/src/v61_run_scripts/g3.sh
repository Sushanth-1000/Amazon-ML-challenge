# v6.1 GPU step 3: v4's cross-encoder scores for v6.1's top-3 pairs (train stage-2 slice + H, test)
mkdir -p ~/v61ce
gcloud storage cp -r gs://mlc-gcp1-v6/p1_archive/er-gpu/v4_ce_w4/ce_model ~/v61ce/ -q
gcloud storage cp gs://mlc-gcp1-v6/code/v61/ce.py ~/v61/ce61.py -q
until grep -q G2_DONE ~/v61_g2.log 2>/dev/null; do sleep 30; done
until gcloud storage ls gs://mlc-gcp1-v6/v61/ce_pairs_done.txt >/dev/null 2>&1; do sleep 30; done
gcloud storage cp gs://mlc-gcp1-v6/v61/ce_score_train.parquet gs://mlc-gcp1-v6/v61/ce_score_test.parquet ~/v61ce/ -q
for sp in train test; do
  python3 ~/v61/ce61.py score --split $sp --work ~/v61ce --out ~/v61ce/ce_model --score-bs 1024 >> ~/v61_g3.log 2>&1 || { echo "G3 $sp FAILED" >> ~/v61_g3.log; exit 1; }
done
gcloud storage cp ~/v61ce/train_ce.parquet ~/v61ce/test_ce.parquet gs://mlc-gcp1-v6/v61/ -q && echo done > ~/ce_scored_done.txt && gcloud storage cp ~/ce_scored_done.txt gs://mlc-gcp1-v6/v61/ -q
echo G3_DONE >> ~/v61_g3.log
