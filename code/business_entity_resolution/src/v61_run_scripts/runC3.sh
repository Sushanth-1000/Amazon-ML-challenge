# v6.1 third machine (p2-v61c): smoke test of the new code, French-style copy features, stage 1 A seed 2 + B seed 2
until grep -q SETUP_DONE ~/setup.log 2>/dev/null; do sleep 20; done
. ~/venv/bin/activate; cd ~
until gcloud storage ls gs://mlc-gcp1-v6/v61/pairs61_done.txt >/dev/null 2>&1 && gcloud storage ls gs://mlc-gcp1-v6/v61/train_cos_fr.npy >/dev/null 2>&1 && gcloud storage ls gs://mlc-gcp1-v6/v61/train_cos.npy >/dev/null 2>&1; do sleep 30; done
gcloud storage cp gs://mlc-gcp1-v6/v61/train_pairs61.parquet gs://mlc-gcp1-v6/v61/train_cos_fr.npy gs://mlc-gcp1-v6/v61/train_cos.npy gs://mlc-gcp1-v6/v61/train_copyqs.npy ~/w/ -q
python ~/src/v61.py feats --split train --what all --smoke --data ~/dataset --work ~/w --outwork ~/wsmoke --v4meta ~/w/v4_meta.json > ~/smoke_feats.log 2>&1 || { echo SMOKE_FAILED >> ~/smoke_feats.log; exit 1; }
python ~/src/v61.py stage1 --part A --seeds 0 --data ~/dataset --work ~/wsmoke > ~/smoke_stage1.log 2>&1 || { echo SMOKE_FAILED >> ~/smoke_stage1.log; exit 1; }
echo SMOKE_OK >> ~/smoke_stage1.log
python ~/src/v61.py feats --split train --what copies --data ~/dataset --work ~/w --v4meta ~/w/v4_meta.json > ~/feats_copies.log 2>&1 || { echo FEATS_FAILED >> ~/feats_copies.log; exit 1; }
gcloud storage cp ~/w/train_Xc.parquet ~/feats_copies.log gs://mlc-gcp1-v6/v61/ -q && echo done > ~/copies_done.txt && gcloud storage cp ~/copies_done.txt gs://mlc-gcp1-v6/v61/ -q
until gcloud storage ls gs://mlc-gcp1-v6/v61/featsN_done.txt >/dev/null 2>&1; do sleep 30; done
gcloud storage cp gs://mlc-gcp1-v6/v61/train_X.parquet gs://mlc-gcp1-v6/v61/meta.json ~/w/ -q
python ~/src/v61.py stage1 --part A --seeds 2 --data ~/dataset --work ~/w > ~/stage1_C.log 2>&1 || { echo STAGE1_FAILED >> ~/stage1_C.log; exit 1; }
gcloud storage cp ~/w/stage1_A_s2.txt ~/w/p1_from_A_s2.npy ~/w/p1H_from_A_s2.npy gs://mlc-gcp1-v6/v61/ -q && echo done > ~/s.txt && gcloud storage cp ~/s.txt gs://mlc-gcp1-v6/v61/s1_A_2_done.txt -q
python ~/src/v61.py stage1 --part B --seeds 2 --data ~/dataset --work ~/w >> ~/stage1_C.log 2>&1 || { echo STAGE1_FAILED >> ~/stage1_C.log; exit 1; }
gcloud storage cp ~/w/stage1_B_s2.txt ~/w/p1_from_B_s2.npy ~/w/p1H_from_B_s2.npy gs://mlc-gcp1-v6/v61/ -q && echo done > ~/s.txt && gcloud storage cp ~/s.txt gs://mlc-gcp1-v6/v61/s1_B_2_done.txt -q
gcloud storage cp ~/stage1_C.log gs://mlc-gcp1-v6/v61/ -q; echo C3_DONE >> ~/stage1_C.log
