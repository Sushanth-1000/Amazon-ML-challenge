. ~/venv/bin/activate; cd ~; mkdir -p ~/fr
gcloud storage cp gs://mlc-gcp1-v6/final/B/matching_results.tsv ~/fr/B.tsv -q
gcloud storage cp gs://mlc-gcp1-v6/code/v61/france_rules.py gs://mlc-gcp1-v6/code/v61/france_lookalike_filter.py ~/src61/ -q
for tau in 0.7 0.5; do
  python ~/fr_addback.py $tau ~/fr/add_$tau.tsv
  python ~/src61/france_lookalike_filter.py --matching ~/fr/add_$tau.tsv --out ~/fr/add_${tau}_f.tsv --test-dir ~/dataset/test --country France | tail -1
  python ~/src61/france_rules.py --matching ~/fr/add_${tau}_f.tsv --out ~/fr/E_fr_$tau.tsv --test-dir ~/dataset/test | tail -1
done
python - <<'PY'
import pandas as pd
rd = lambda p: pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=3)
s1 = rd("/home/Dell/dataset/test/test_source1.tsv").set_index("entity_id").country
rr = pd.concat([rd(f"/home/Dell/dataset/test/test_source{i}.tsv") for i in (2, 3)]); na = set(rr.entity_id[rr.business_address.str.strip() == ""])
for name, p in (("B", "/home/Dell/fr/B.tsv"), ("E tau 0.7", "/home/Dell/fr/E_fr_0.7.tsv"), ("E tau 0.5", "/home/Dell/fr/E_fr_0.5.tsv")):
    m = rd(p); f = m[s1.reindex(m.source1_entity_id).values == "France"]
    L = f.matched_entity_ids.map(lambda x: [y for y in x.split(",") if y]); n = L.map(len)
    print(f"{name:10s} France matches/S1 {n.mean():.4f} | empty {(n == 0).mean():.4f} | no-address share {sum(len(set(x) & na) for x in L) / n.sum():.4f}")
PY
gcloud storage cp ~/fr/E_fr_0.7.tsv ~/fr/E_fr_0.5.tsv gs://mlc-gcp1-v6/final/fr_addback/ -q; echo ADDBACK_DONE
