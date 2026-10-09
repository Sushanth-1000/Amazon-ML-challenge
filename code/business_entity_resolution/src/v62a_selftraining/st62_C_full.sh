# half 1 full run (smoke skipped: its 1-in-50 sample had no half-1 records; code path proven by half 0's full run)
. ~/venv/bin/activate; cd ~
md5sum ~/src/st_france.py > ~/st_code_md5.txt
python ~/src/st_france.py --work ~/w --data ~/dataset --half 1 > ~/st_half1.log 2>&1 || { echo ST_FAILED >> ~/st_half1.log; exit 1; }
gcloud storage cp ~/w/st_fr_half1.npz ~/st_half1.log gs://mlc-gcp1-v6/v61/v62a/ -q && echo done > ~/sth.txt && gcloud storage cp ~/sth.txt gs://mlc-gcp1-v6/v61/st_half1_done.txt -q
echo C_FULL_DONE >> ~/st_half1.log
