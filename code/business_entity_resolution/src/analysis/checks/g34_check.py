"""Row changes of G3 / G4 relative to G2 (LB 0.987342), per country.
  python g34_check.py TEST_DIR G2.tsv G3.tsv G4.tsv"""
import sys
import pandas as pd

T, G2, *files = sys.argv[1:]
c = pd.read_csv(f'{T}/test_source1.tsv', sep='\t', dtype=str, keep_default_na=False, quoting=3,
                usecols=['entity_id', 'country']).set_index('entity_id').country.to_dict()
L = lambda f: {a: set(x for x in v.split(',') if x)
               for a, v in zip(*pd.read_csv(f, sep='\t', dtype=str, keep_default_na=False).T.values)}
g2 = L(G2)
for f in files:
    d = L(f)
    for k in ('US', 'India', 'France'):
        ch = [a for a in g2 if c[a] == k and d[a] != g2[a]]
        print(f'{f} vs G2 {k:6s}: {len(ch):,} S1 differ | +{sum(len(d[a] - g2[a]) for a in ch):,} / '
              f'-{sum(len(g2[a] - d[a]) for a in ch):,} matches | empty in G2 -> matched: {sum(1 for a in ch if not g2[a]):,}')
