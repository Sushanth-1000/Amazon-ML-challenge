# v6.1 final chain on p2-cpu: cross-encoder pair export -> matcher (while GPU scores) -> stage 2 fit/apply -> validator
until grep -q STAGE1_ALL_DONE ~/finish.log 2>/dev/null; do sleep 30; done
. ~/venv/bin/activate; cd ~
python ~/src61/ce.py export --data ~/dataset --work ~/w61 > ~/ce_export.log 2>&1 || { echo CE_EXPORT_FAILED >> ~/ce_export.log; exit 1; }
python ~/src61/ce.py export_test --data ~/dataset --work ~/w61 > ~/ce_export_test.log 2>&1 || { echo CE_EXPORT_FAILED >> ~/ce_export_test.log; exit 1; }
gcloud storage cp ~/w61/ce_score_train.parquet ~/w61/ce_score_test.parquet gs://mlc-gcp1-v6/v61/ -q && echo done > ~/ce_pairs_done.txt && gcloud storage cp ~/ce_pairs_done.txt gs://mlc-gcp1-v6/v61/ -q
python ~/src61/matcher.py --data ~/dataset --work ~/w61 > ~/matcher.log 2>&1 || echo "MATCHER FAILED (stage 2 continues without edges)" >> ~/matcher.log
until gcloud storage ls gs://mlc-gcp1-v6/v61/ce_scored_done.txt >/dev/null 2>&1; do sleep 30; done
gcloud storage cp gs://mlc-gcp1-v6/v61/train_ce.parquet gs://mlc-gcp1-v6/v61/test_ce.parquet ~/w61/ -q
python ~/src61/stage2.py --stage fit --data ~/dataset --work ~/w61 > ~/stage2.log 2>&1 || { echo STAGE2_FAILED >> ~/stage2.log; exit 1; }
python ~/src61/stage2.py --stage apply --data ~/dataset --work ~/w61 --out ~/out61 > ~/apply.log 2>&1 || { echo APPLY_FAILED >> ~/apply.log; exit 1; }
python3 ~/dataset/../utils/validate_submission.py --matching ~/out61/matching_results.tsv --candidate ~/out61/candidate_pairs.tsv --test-dir ~/dataset/test --check-ids > ~/validate.log 2>&1
gcloud storage cp ~/out61/matching_results.tsv ~/out61/candidate_pairs.tsv ~/w61/train_stage2.parquet ~/w61/meta.json ~/stage2.log ~/apply.log ~/validate.log ~/matcher.log gs://mlc-gcp1-v6/v61/final/ -q
echo V61_DONE > ~/v61_done
