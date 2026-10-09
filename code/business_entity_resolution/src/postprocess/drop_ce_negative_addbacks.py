"""G2 = G minus the French add-back pairs whose France-adapted cross-encoder (v6.2b) score is < 0.
Add-back pairs = French matches present in G (= E's France) but not in D2 (= B's France).
  python drop_ce_negative_addbacks.py --g G.tsv --d2 D2.tsv --ce-dir v62b --test-dir dataset/test --out G2.tsv"""
import argparse, glob
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
for k in ('--g', '--d2', '--ce-dir', '--test-dir', '--out'):
    ap.add_argument(k, required=True)
ap.add_argument('--threshold', type=float, default=0.0)
a = ap.parse_args()
rd = lambda f, **k: pd.read_csv(f'{a.test_dir}/{f}', sep='\t', dtype=str, keep_default_na=False, quoting=3, **k)
t1 = rd('test_source1.tsv', usecols=['entity_id', 'country'])
tr = pd.concat([rd(f'test_source{i}.tsv', usecols=['entity_id']) for i in (2, 3)], ignore_index=True)
si = pd.Series(np.arange(len(t1)), index=t1.entity_id.values)  # CE files index S1 / records by row position
qi = pd.Series(np.arange(len(tr)), index=tr.entity_id.values)
fr = set(t1.entity_id[t1.country == 'France'])
load = lambda f: {s: [x for x in v.split(',') if x]
                  for s, v in zip(*pd.read_csv(f, sep='\t', dtype=str, keep_default_na=False).T.values)}
g, d2 = load(a.g), load(a.d2)
ce = pd.concat([pd.read_parquet(f, columns=['q', 's', 'ce']) for f in glob.glob(f'{a.ce_dir}/*_test_ce.parquet')])
ce = ce.drop_duplicates(['q', 's']).set_index(['q', 's']).ce
drop = set()
for s in fr:
    for x in set(g[s]) - set(d2[s]):
        v = ce.get((qi[x], si[s]))
        if v is not None and v < a.threshold:
            drop.add((s, x))
print('dropping', len(drop), 'CE-negative add-back pairs')
with open(a.out, 'w', encoding='utf-8', newline='') as fh:
    fh.write('source1_entity_id\tmatched_entity_ids\n')
    for s, v in g.items():
        fh.write(s + '\t' + ','.join(x for x in v if (s, x) not in drop) + '\n')
