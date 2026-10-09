# p2-cpu: share the test-side inputs v6.2 needs (after v6.1 stage 1 + scoring; ce_groups after the CE export)
until grep -q STAGE1_ALL_DONE ~/finish.log 2>/dev/null; do sleep 30; done
gcloud storage cp ~/w61/test_pairs61.parquet ~/w61/test_X.parquet ~/w61/test_s1.parquet ~/w61/test_r.parquet gs://mlc-gcp1-v6/v61/v62in/ -q && echo done > ~/v62in_done.txt && gcloud storage cp ~/v62in_done.txt gs://mlc-gcp1-v6/v61/ -q
until [ -f ~/ce_pairs_done.txt ]; do sleep 30; done
gcloud storage cp ~/w61/ce_groups.parquet gs://mlc-gcp1-v6/v61/v62in/ -q && echo done > ~/cegroups_done.txt && gcloud storage cp ~/cegroups_done.txt gs://mlc-gcp1-v6/v61/ -q
echo SHARE62_DONE >> ~/share62.log
