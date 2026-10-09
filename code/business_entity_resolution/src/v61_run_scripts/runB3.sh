# v6.1 test machine (p2-cpu): finish the running test union, test features, stage 1 B seeds 0,1, then finish + score
while pgrep -f "v61.py union --split test" >/dev/null; do sleep 30; done
grep -q "after v4 prefilter" ~/union_test.log || { echo "UNION_FAILED (see log)" >> ~/union_test.log; exit 1; }
gcloud storage cp ~/w61/test_qs.npy ~/union_test.log gs://mlc-gcp1-v6/v61/ -q && echo UNION_TEST_DONE >> ~/union_test.log
. ~/venv/bin/activate; cd ~
gcloud storage cp gs://mlc-gcp1-v6/code/v61/v61.py ~/src61/v61.py -q && md5sum ~/src61/v61.py > ~/code_md5.txt
until gcloud storage ls gs://mlc-gcp1-v6/v61/test_cos.npy >/dev/null 2>&1 && gcloud storage ls gs://mlc-gcp1-v6/v61/label_tables.json >/dev/null 2>&1; do sleep 30; done
gcloud storage cp gs://mlc-gcp1-v6/v61/test_cos.npy gs://mlc-gcp1-v6/v61/label_tables.json gs://mlc-gcp1-v6/v61/meta.json ~/w61/ -q
python ~/src61/v61.py feats --split test --data ~/dataset --work ~/w61 --v4meta ~/w/v4_meta.json > ~/feats_test.log 2>&1 || { echo FEATS_FAILED >> ~/feats_test.log; exit 1; }
echo FEATS_TEST_DONE >> ~/feats_test.log
until gcloud storage ls gs://mlc-gcp1-v6/v61/featsN_done.txt >/dev/null 2>&1 && gcloud storage ls gs://mlc-gcp1-v6/v61/copies_done.txt >/dev/null 2>&1; do sleep 30; done
gcloud storage cp gs://mlc-gcp1-v6/v61/train_X.parquet gs://mlc-gcp1-v6/v61/train_Xc.parquet ~/w61/ -q
python ~/src61/v61.py stage1 --part B --seeds 0,1 --data ~/dataset --work ~/w61 > ~/stage1_B.log 2>&1 || { echo STAGE1_FAILED >> ~/stage1_B.log; exit 1; }
for k in 0 1; do gcloud storage cp ~/w61/stage1_B_s$k.txt ~/w61/p1_from_B_s$k.npy ~/w61/p1H_from_B_s$k.npy gs://mlc-gcp1-v6/v61/ -q && echo done > ~/s1_B_$k.txt && gcloud storage cp ~/s1_B_$k.txt gs://mlc-gcp1-v6/v61/s1_B_${k}_done.txt -q; done
for m in s1_A_0 s1_A_1 s1_A_2 s1_B_2; do until gcloud storage ls gs://mlc-gcp1-v6/v61/${m}_done.txt >/dev/null 2>&1; do sleep 30; done; done
gcloud storage cp "gs://mlc-gcp1-v6/v61/stage1_A_s*.txt" gs://mlc-gcp1-v6/v61/stage1_B_s2.txt "gs://mlc-gcp1-v6/v61/p1_from_A_s*.npy" "gs://mlc-gcp1-v6/v61/p1H_from_A_s*.npy" gs://mlc-gcp1-v6/v61/p1_from_B_s2.npy gs://mlc-gcp1-v6/v61/p1H_from_B_s2.npy ~/w61/ -q
ls ~/w61/stage1_*_s*.txt | wc -l
python ~/src61/v61.py finish --data ~/dataset --work ~/w61 > ~/finish.log 2>&1 && python ~/src61/v61.py score --data ~/dataset --work ~/w61 --out ~/out61 >> ~/finish.log 2>&1 || { echo SCORE_FAILED >> ~/finish.log; exit 1; }
gcloud storage cp ~/w61/train_scored.parquet ~/w61/test_scored.parquet ~/out61/candidate_pairs.tsv ~/finish.log gs://mlc-gcp1-v6/v61/ -q && echo STAGE1_ALL_DONE >> ~/finish.log
