"""France profile of several full matching files vs the 0.986041 file (first argument)."""
import sys
import numpy as np, pandas as pd
D = sys.argv[1]; files = sys.argv[2:]
rd = lambda f: pd.read_csv(f, sep='\t', dtype=str, keep_default_na=False, quoting=3)
t1 = rd(f'{D}/test/test_source1.tsv'); tr = pd.concat([rd(f'{D}/test/test_source2.tsv'), rd(f'{D}/test/test_source3.tsv')])
fr = set(t1.entity_id[t1.country == 'France']); noaddr = dict(zip(tr.entity_id, tr.business_address == ''))
load = lambda f: {a: frozenset(x for x in v.split(',') if x) for a, v in zip(*rd(f).T.values) if a in fr}
base = load(files[0])
for f in files:
    d = load(f); n = np.array([len(v) for v in d.values()]); m = [x for v in d.values() for x in v]
    same = np.mean([d[a] == base[a] for a in d])
    add = sum(len(d[a] - base[a]) for a in d); rem = sum(len(base[a] - d[a]) for a in d)
    print(f'{f}: France matches/S1 {n.mean():.4f} | empty {np.mean(n == 0):.4%} | no-address {np.mean([noaddr[x] for x in m]):.4%} '
          f'| identical to 0.986041 France {same:.4%} | vs 0.986041: +{add:,} / -{rem:,} matches', flush=True)
