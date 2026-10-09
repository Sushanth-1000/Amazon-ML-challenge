# v6.1 train machine (p2-v61a): share pairs -> normal features -> stage 1 part A seeds 0,1
until grep -q UNION_TRAIN_DONE ~/union_train.log 2>/dev/null; do sleep 30; done
gcloud storage cp gs://mlc-gcp1-v6/code/v61/v61.py ~/src/v61.py -q && md5sum ~/src/v61.py > ~/code_md5.txt
gcloud storage cp ~/w/train_pairs61.parquet gs://mlc-gcp1-v6/v61/ -q && echo done > ~/pairs61_done.txt && gcloud storage cp ~/pairs61_done.txt gs://mlc-gcp1-v6/v61/ -q
until gcloud storage ls gs://mlc-gcp1-v6/v61/train_cos.npy >/dev/null 2>&1; do sleep 30; done
gcloud storage cp gs://mlc-gcp1-v6/v61/train_cos.npy ~/w/ -q
. ~/venv/bin/activate; cd ~
python ~/src/v61.py feats --split train --what normal --data ~/dataset --work ~/w --v4meta ~/w/v4_meta.json > ~/feats_train.log 2>&1 || { echo FEATS_FAILED >> ~/feats_train.log; exit 1; }
gcloud storage cp ~/w/meta.json gs://mlc-gcp1-v6/v61/ -q && gcloud storage cp ~/w/label_tables.json ~/feats_train.log gs://mlc-gcp1-v6/v61/ -q
gcloud storage cp ~/w/train_X.parquet gs://mlc-gcp1-v6/v61/ -q && echo done > ~/featsN_done.txt && gcloud storage cp ~/featsN_done.txt gs://mlc-gcp1-v6/v61/ -q
until gcloud storage ls gs://mlc-gcp1-v6/v61/copies_done.txt >/dev/null 2>&1; do sleep 30; done
gcloud storage cp gs://mlc-gcp1-v6/v61/train_Xc.parquet ~/w/ -q
python ~/src/v61.py stage1 --part A --seeds 0,1 --data ~/dataset --work ~/w > ~/stage1_A.log 2>&1 || { echo STAGE1_FAILED >> ~/stage1_A.log; exit 1; }
for k in 0 1; do gcloud storage cp ~/w/stage1_A_s$k.txt ~/w/p1_from_A_s$k.npy ~/w/p1H_from_A_s$k.npy gs://mlc-gcp1-v6/v61/ -q && echo done > ~/s1_A_$k.txt && gcloud storage cp ~/s1_A_$k.txt gs://mlc-gcp1-v6/v61/s1_A_${k}_done.txt -q; done
gcloud storage cp ~/stage1_A.log gs://mlc-gcp1-v6/v61/ -q; echo A3_DONE >> ~/stage1_A.log
