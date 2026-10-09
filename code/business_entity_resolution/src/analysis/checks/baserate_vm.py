"""Base-rate check: share of Source-2/3 records that belong to ANY S1 -- train (ground truth) vs test (what G4 matches),
per country; and on the US/India test frame, how records split by best stage-2 probability (x_p2a)."""
import sys
import numpy as np, pandas as pd
D, G4, FRAME = sys.argv[1:4]
rd = lambda f, **k: pd.read_csv(f'{D}/{f}', sep='\t', dtype=str, keep_default_na=False, quoting=3, **k)
for split in ('train', 'test'):
    s1 = rd(f'{split}/{split}_source1.tsv', usecols=['entity_id', 'country'])
    r = pd.concat([rd(f'{split}/{split}_source{i}.tsv', usecols=['entity_id', 'country']) for i in (2, 3)])
    if split == 'train':
        gt = rd('train/train_ground_truth.tsv'); owned = {m for v in gt.matched_entity_ids for m in v.split(',') if m}
        what = 'owned (ground truth)'
    else:
        g = pd.read_csv(G4, sep='\t', dtype=str, keep_default_na=False); owned = {m for v in g.iloc[:, 1] for m in v.split(',') if m}
        what = 'matched by G4'
    r['own'] = r.entity_id.isin(owned)
    print(f'{split}: S1 {len(s1):,} | records {len(r):,} | {what} {r.own.mean():.1%} | records per S1 {len(r) / len(s1):.2f}')
    print('   by country:', {c: f'{m:.1%} of {n:,}' for c, (m, n) in r.groupby('country').own.agg(['mean', 'size']).iterrows()})
f = pd.read_parquet(FRAME, columns=['record_id', 'slice', 'x_p2a'])
best = f.groupby('record_id').agg(conf=('x_p2a', 'max'), sl=('slice', 'first'))
thr = best.sl.map({'noaddr': 0.5, 'native': 0.5, 'other': 0.625})
print('test US/India records with a stage-2 candidate:', f'{len(best):,}')
for sl in ('noaddr', 'other', 'native'):
    b = best[best.sl == sl]; t = thr[best.sl == sl]
    print(f'  {sl:6s} {len(b):>9,} | accepted {np.mean(b.conf >= t):.1%} | rejected zone [0.02, thr) {np.mean((b.conf < t) & (b.conf >= 0.02)):.1%} '
          f'| < 0.02 {np.mean(b.conf < 0.02):.1%} | accepted just above thr (thr..thr+0.15) {np.mean((b.conf >= t) & (b.conf < t + 0.15)):.2%}')
