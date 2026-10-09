set -x
cd /root/v7t; D=/root/er/student_resource/dataset; PY=/root/venv/bin/python; W=/root/v7t/in/w
$PY -c "import scipy, rapidfuzz, anyascii, lightgbm, sklearn" || /root/venv/bin/pip install -q scipy rapidfuzz anyascii
mkdir -p src out && cp code/lib/*.py code/tests/*.py src/
# ---- Job 1: France profile of v6.2a
$PY code/fr/merge_by_country.py --base in/subA.tsv --part in/v62a.tsv --countries France --out out/j1_merged.tsv --test-dir $D/test
$PY code/fr/france_lookalike_filter.py --matching out/j1_merged.tsv --out out/j1_v62a_fr.tsv --test-dir $D/test --country France
$PY code/fr/france_rules.py --matching out/j1_v62a_fr.tsv --out out/j1_v62a_fr_r1.tsv --test-dir $D/test
for f in j1_v62a_fr j1_v62a_fr_r1; do (cd /root/er/student_resource && $PY utils/validate_submission.py --matching /root/v7t/out/$f.tsv --test-dir dataset/test --check-ids 2>&1 | tail -1) > out/$f.validator.txt; done
$PY j1_profile.py $D in/subA.tsv in/v62a.tsv out/j1_merged.tsv out/j1_v62a_fr.tsv out/j1_v62a_fr_r1.tsv > out/job1_profile.txt 2>&1
md5sum out/j1_*.tsv > out/job1_md5.txt
gsutil -q cp out/job1_* out/*.validator.txt out/j1_v62a_fr.tsv out/j1_v62a_fr_r1.tsv gs://mlc-v5-er-02664747/v7t/out/
# ---- Test A: self-training round 2 (US labelled -> India unlabelled)
$PY src/e19_selftrain.py --data $D --work $W --src src --source US --out out_r2 --base-draws $W/draws_US.npz --rounds 2 --tag _r2 > out/testA_e19.log 2>&1 || echo E19_FAILED >> out/testA_e19.log
gsutil -q cp out/testA_e19.log gs://mlc-v5-er-02664747/v7t/out/
$PY src/e27_pers1.py --data $D --work $W --src src --probs out_r2 --tag _r2 --configs reg,st1,st2 > out/testA_e27.log 2>&1 || echo E27_FAILED >> out/testA_e27.log
echo TESTA_DONE >> out/testA_e27.log; gsutil -q cp out/testA_e27.log gs://mlc-v5-er-02664747/v7t/out/; gsutil -q cp -r out_r2 gs://mlc-v5-er-02664747/v7t/out/
