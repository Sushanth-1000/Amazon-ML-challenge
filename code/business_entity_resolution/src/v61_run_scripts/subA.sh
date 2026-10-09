# Submission A: v6.1 US/India + teammates' France (0.983297 rows), candidates merged the same way, official validator
until [ -f ~/v61_done ]; do sleep 60; done
until gcloud storage ls gs://mlc-gcp1-v6/v62/team_0983297_candidate_pairs.tsv >/dev/null 2>&1; do sleep 60; done
. ~/venv/bin/activate; cd ~; mkdir -p ~/subA
gcloud storage cp gs://mlc-gcp1-v6/code/v61/merge_by_country.py ~/src61/ -q
gcloud storage cp gs://mlc-gcp1-v6/v62/team_0983297_candidate_pairs.tsv ~/w61/team_cand.tsv -q
python ~/src61/merge_by_country.py --base ~/out61/matching_results.tsv --part ~/w61/team.tsv --countries France --out ~/subA/matching_results.tsv --test-dir ~/dataset/test > ~/subA.log 2>&1 || { echo SUBA_FAILED >> ~/subA.log; exit 1; }
python ~/src61/merge_by_country.py --base ~/out61/candidate_pairs.tsv --part ~/w61/team_cand.tsv --countries France --out ~/subA/candidate_pairs.tsv --test-dir ~/dataset/test >> ~/subA.log 2>&1 || { echo SUBA_FAILED >> ~/subA.log; exit 1; }
python3 ~/dataset/../utils/validate_submission.py --matching ~/subA/matching_results.tsv --candidate ~/subA/candidate_pairs.tsv --test-dir ~/dataset/test --check-ids >> ~/subA.log 2>&1
md5sum ~/subA/*.tsv >> ~/subA.log
gcloud storage cp ~/subA/matching_results.tsv ~/subA/candidate_pairs.tsv ~/subA.log gs://mlc-gcp1-v6/final/subA/ -q
echo SUBA_DONE >> ~/subA.log
